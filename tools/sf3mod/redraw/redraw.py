"""Local GPU sprite redraw pilot: original SF3 frame (pose) + mascot reference -> FLUX.2 [klein] edit -> keyed,
aligned RGBA frame at the original canvas size (+ a 4x copy).

  python sf3mod/redraw/redraw.py prep  <bank> <variant> <patterns...>   # condition images + job json for the worker
  python sf3mod/redraw/redraw.py post  <bank> <variant> <patterns...>   # key, align, downscale -> frames/
  python sf3mod/redraw/redraw.py sheet <bank> <variant> <patterns...>   # original | codex | local comparison

Variants live in VARIANTS (prompt, refs, scale). Outputs go to sf3mod/build/redraw/<bank>/<variant>/.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
SKIN = ROOT / 'sf3mod' / 'build' / 'skin'
OUT = ROOT / 'sf3mod' / 'build' / 'redraw'
JOBS = OUT / '_jobs'
BG = (128, 128, 128)
MARGIN = 64

GIGA_DESC = ('GIGA: a huge muscular bodybuilder wrestler with tanned skin, short swept-back black hair, a short black '
             'beard, shirtless, black wrestling briefs, white wrist wraps on both wrists, black lace-up combat boots, '
             'bare muscular legs')
STYLE = ('Street Fighter III 3rd Strike sprite art: hand-drawn 2D pixel art sprite, bold dark outline, few flat '
         'cel-shaded tones, crisp edges. Plain flat solid gray background, no floor shadow, no text, no border.')

VARIANTS = {
    # ref = one Codex GIGA idle cell (already in sprite style)
    'kA': dict(refs=['codex0'], prompt=(
        'Image 1 is a pixel-art fighting game sprite of a fighter. Image 2 shows the character ' + GIGA_DESC + '. '
        'Redraw the fighter of image 1 as the character from image 2, same character design and colours as image 2. '
        'Keep exactly the same pose, body position, silhouette, size and placement as image 1: every limb, hand, '
        'foot and the head stay in the same place and the same angle. ' + STYLE)),
    # ref = concept full body + portrait (not sprite style)
    'kB': dict(refs=['concept', 'portrait'], prompt=(
        'Image 1 is a pixel-art fighting game sprite of a fighter. Images 2 and 3 show the character ' + GIGA_DESC +
        '. Redraw the fighter of image 1 as this character, same face, hair, beard, body and outfit as images 2 '
        'and 3. Keep exactly the same pose, body position, silhouette, size and placement as image 1: every limb, '
        'hand, foot and the head stay in the same place and the same angle. ' + STYLE)),
    # Codex cell + concept
    'kC': dict(refs=['codex0', 'concept'], prompt=(
        'Image 1 is a pixel-art fighting game sprite of a fighter. Images 2 and 3 show the character ' + GIGA_DESC +
        '. Redraw the fighter of image 1 as this character, drawn exactly like the sprite in image 2. '
        'Keep exactly the same pose, body position, silhouette, size and placement as image 1: every limb, '
        'hand, foot and the head stay in the same place and the same angle. ' + STYLE)),
    # reference = head and chest only (no full-body pose to copy)
    'kD': dict(refs=['codexhead0'], prompt=(
        'Image 1 is a pixel-art fighting game sprite of a fighter. Image 2 is a close-up of the head and chest of the '
        'character ' + GIGA_DESC + '. Redraw the fighter of image 1 as this character, same face, hair, beard, skin '
        'and outfit, drawn in the same sprite style as image 2. The pose comes only from image 1: keep exactly the '
        'same pose, body position, silhouette, size and placement as image 1, every limb, hand, foot and the head '
        'in the same place and the same angle. ' + STYLE)),
    # pose image last: character first, pose second
    'kE': dict(refs=['codex0'], first='ref', prompt=(
        'Image 1 shows the character ' + GIGA_DESC + '. Image 2 is a pixel-art fighting game sprite of a different '
        'fighter in a specific pose. Draw the character from image 1 in exactly the pose of image 2: same pose, body '
        'position, silhouette, size and placement as image 2, every limb, hand, foot and the head in the same place '
        'and angle as image 2. Do not copy the pose of image 1. ' + STYLE)),
}
for _k, _st, _ini, _c in (('kI', 0.8, 'recolor', 'recolor'), ('kJ', 0.72, 'recolor', 'recolor'),
                          ('kK', 0.8, 'recolor', 'orig'), ('kL', 0.85, 'recolor', 'recolor'),
                          ('kM', 0.8, 'recolor', 'recolor')):
    VARIANTS[_k] = dict(VARIANTS['kD'], strength=_st, init=_ini, cond=_c)
VARIANTS['kM']['steps'] = 6
# chain: reference = this character's previous redrawn frame (raw canvas) + head close-up; frame 0 uses Codex cell
VARIANTS['kP'] = dict(VARIANTS['kL'], refs=['prev', 'codexhead0'], prompt=(
    'Image 1 is a pixel-art fighting game sprite of a fighter in a specific pose. Image 2 is the previous animation '
    'frame of the character ' + GIGA_DESC + ', already drawn as a sprite. Image 3 is a close-up of his head and '
    'chest. Redraw the fighter of image 1 as exactly this character, identical design, colours and drawing style as '
    'image 2 and image 3. The pose comes only from image 1: keep exactly the same pose, body position, silhouette, '
    'size and placement as image 1, every limb, hand, foot and the head in the same place and the same angle. '
    + STYLE))
VARIANTS['kN'] = dict(VARIANTS['kL'], strength=0.92)
for _k, _st, _ini in (('kF', 0.85, 'cond'), ('kG', 0.72, 'cond'), ('kH', 0.8, 'gray')):
    VARIANTS[_k] = dict(VARIANTS['kD'], strength=_st, init=_ini)


# Per-bank colour-block of the original frame toward the mascot (init + pose image): hue class -> (hue deg or
# None, saturation, value scale). Classes: skin, green, yellow, red, brown, blue, purple, white. Outline stays.
RECOLOR = {
    'pl01': {  # Alex -> GIGA: tan skin, green pants/gloves -> bare skin, blond hair + red band -> black, boots black
        'skin': (27, 0.55, 0.92), 'green': (27, 0.52, 1.0, 0.5), 'yellow': (0, 0.0, 0.22), 'red': (0, 0.0, 0.2),
        'brown': (0, 0.0, 0.3)},
}


def hue_class(h, s, v):
    hd = h * 360
    if v < 0.16: return 'dark'
    if s < 0.16: return 'white' if v > 0.7 else 'grey'
    if 65 <= hd < 170: return 'green'
    if 170 <= hd < 255: return 'blue'
    if 255 <= hd < 320: return 'purple'
    if hd >= 345 or hd < 8: return 'red' if s > 0.55 else 'skin'
    if 38 <= hd < 65: return 'yellow' if s > 0.45 else 'skin'
    if v < 0.5 and s > 0.4: return 'brown'
    if s > 0.75 and hd < 20: return 'red'
    return 'skin'


def recolor(fr, bank):
    import colorsys
    rules = RECOLOR.get(bank, {})
    a = np.asarray(fr).copy()
    cache = {}
    H, W = a.shape[:2]
    for y in range(H):
        for x in range(W):
            if a[y, x, 3] == 0: continue
            key = tuple(a[y, x, :3])
            if key not in cache:
                h, s_, v = colorsys.rgb_to_hsv(*(c / 255 for c in key))
                r = rules.get(hue_class(h, s_, v))
                if r is None:
                    cache[key] = key
                else:
                    th, ts, vs = r[:3]
                    vmin = r[3] if len(r) > 3 else 0.0  # lift dark shades (baggy-pants folds) toward skin
                    nh = h if th is None else th / 360
                    nv = min(1, vmin + (1 - vmin) * v * vs) if vmin else min(1, v * vs)
                    cache[key] = tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(nh, ts, nv))
            a[y, x, :3] = cache[key]
    return Image.fromarray(a)


def ref_paths(bank, variant):
    d = OUT / bank / '_refs'
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for r in VARIANTS[variant]['refs']:
        if r == 'prev':
            out.append('prev'); continue
        p = d / f'{r}.png'
        if not p.exists():
            if r.startswith('codexhead'):
                c = Image.open(ref_paths_one(bank, 'codex' + r[9:])).convert('RGB')
                c.crop((0, 0, c.width, int(c.height * 0.5))).save(p)
            elif r.startswith('codex'):
                i = int(r[5:])
                sheet = Image.open(OUT / bank / 'idle_codex.png').convert('RGB')
                cw, ch = sheet.width / 4, sheet.height / 3
                x, y = i % 4, i // 4
                cell = sheet.crop((round(x * cw) + 3, round(y * ch) + 3, round((x + 1) * cw) - 3, round((y + 1) * ch) - 3))
                cell.resize((cell.width * 2, cell.height * 2), Image.LANCZOS).save(p)
            elif r == 'concept':
                Image.open(ROOT / 'art' / 'concept' / 'giga.png').convert('RGB').save(p)
            elif r == 'portrait':
                im = Image.open(ROOT / 'sf3mod' / 'assets' / 'gen' / 'portrait_giga.png').convert('RGBA')
                bg = Image.new('RGBA', im.size, BG + (255,))
                Image.alpha_composite(bg, im).convert('RGB').save(p)
        out.append(str(p))
    return out


def ref_paths_one(bank, r):
    v = {'refs': [r], 'prompt': ''}
    VARIANTS['_tmp'] = v
    return ref_paths(bank, '_tmp')[0]


def geom(w, h, long_side=880):
    s = max(2, min(8, long_side // max(w, h)))
    W = -(-(w * s + 2 * MARGIN) // 16) * 16
    H = -(-(h * s + 2 * MARGIN) // 16) * 16
    return s, W, H, (W - w * s) // 2, (H - h * s) // 2


def frame(bank, pat):
    return Image.open(SKIN / bank / 'frames' / f'{pat:04d}.png').convert('RGBA')


def cond_image(fr):
    w, h = fr.size
    s, W, H, ox, oy = geom(w, h)
    big = fr.resize((w * s, h * s), Image.NEAREST)
    can = Image.new('RGBA', (W, H), BG + (255,))
    can.alpha_composite(big, (ox, oy))
    return can.convert('RGB'), (s, W, H, ox, oy)


def prep(bank, variant, pats, seed=7, steps=None):
    steps = steps or VARIANTS[variant].get('steps', 4)
    vd = OUT / bank / variant
    (vd / 'cond').mkdir(parents=True, exist_ok=True)
    refs = ref_paths(bank, variant)
    tasks = []
    for p in pats:
        fr = frame(bank, p)
        c, (s, W, H, ox, oy) = cond_image(fr)
        cp = vd / 'cond' / f'{p:04d}.png'
        c.save(cp)
        rc = None
        if 'recolor' in (VARIANTS[variant].get('init'), VARIANTS[variant].get('cond')):
            rc = vd / 'cond' / f'{p:04d}_recolor.png'
            cond_image(recolor(fr, bank))[0].save(rc)
        if VARIANTS[variant].get('cond') == 'recolor':
            cp = rc
        extra = {}
        if VARIANTS[variant].get('init'):
            ip = cp
            if VARIANTS[variant]['init'] == 'recolor':
                ip = rc
            elif VARIANTS[variant]['init'] == 'gray':
                ip = vd / 'cond' / f'{p:04d}_gray.png'
                g = c.convert('L').convert('RGB'); g.save(ip)
            extra = dict(init=str(ip), strength=VARIANTS[variant]['strength'])
        rr = list(refs)
        if 'prev' in VARIANTS[variant]['refs']:
            prev = tasks[-1]['out'] if tasks else ref_paths_one(bank, 'codex0')
            rr = [prev if r == 'prev' else r for r in refs]
        order = rr + [str(cp)] if VARIANTS[variant].get('first') == 'ref' else [str(cp)] + rr
        tasks.append(dict(refs=order, prompt=VARIANTS[variant]['prompt'], width=W, height=H,
                          seed=seed, steps=steps, out=str(vd / 'raw' / f'{p:04d}.png'), **extra))
    JOBS.mkdir(parents=True, exist_ok=True)
    jp = JOBS / f'{bank}_{variant}.json'
    jp.write_text(json.dumps(dict(tasks=tasks), indent=1))
    print('job', jp, len(tasks))


SHEET_PROMPT = (
    'Image 1 is a sheet of 12 animation frames (4 columns, 3 rows) of a pixel-art fighting game sprite. Image 2 is a '
    'close-up of the head and chest of the character ' + GIGA_DESC + '. Redraw every frame of the sheet as this '
    'character, same face, hair, beard, skin and outfit in every frame, drawn in the same sprite style as image 2. '
    'Keep the sheet layout: each frame stays in its own cell with exactly the same pose, body position, silhouette, '
    'size and placement as in image 1, every limb, hand, foot and the head in the same place and angle. ' + STYLE)


def prep_sheet(bank, variant, pats, cols=4, f=0.4, strength=0.85, seed=7, steps=4):
    """One generation for a whole animation: per-frame recoloured cond canvases scaled by f in a cols-wide grid."""
    vd = OUT / bank / variant
    (vd / 'cond').mkdir(parents=True, exist_ok=True)
    cells = []
    for p in pats:
        fr = frame(bank, p)
        c, g = cond_image(recolor(fr, bank))
        cells.append((p, c, g))
    W, H = cells[0][2][1], cells[0][2][2]
    cw, ch = round(W * f), round(H * f)
    rows = -(-len(cells) // cols)
    SW, SH = -(-(cw * cols) // 16) * 16, -(-(ch * rows) // 16) * 16
    sh = Image.new('RGB', (SW, SH), BG)
    for i, (p, c, g) in enumerate(cells):
        sh.paste(c.resize((cw, ch), Image.LANCZOS), ((i % cols) * cw, (i // cols) * ch))
    sp = vd / 'cond' / 'sheet.png'
    sh.save(sp)
    (vd / 'sheet.json').write_text(json.dumps(dict(pats=pats, cols=cols, cw=cw, ch=ch, W=W, H=H)))
    refs = ref_paths(bank, 'kD')
    task = dict(refs=[str(sp)] + refs, prompt=SHEET_PROMPT, width=SW, height=SH, seed=seed, steps=steps,
                out=str(vd / 'raw_sheet.png'), init=str(sp), strength=strength)
    jp = JOBS / f'{bank}_{variant}.json'
    jp.write_text(json.dumps(dict(tasks=[task]), indent=1))
    print('job', jp, SW, SH)


def split_sheet(bank, variant):
    vd = OUT / bank / variant
    m = json.loads((vd / 'sheet.json').read_text())
    raw = Image.open(vd / 'raw_sheet.png').convert('RGB')
    (vd / 'raw').mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(m['pats']):
        x, y = (i % m['cols']) * m['cw'], (i // m['cols']) * m['ch']
        raw.crop((x, y, x + m['cw'], y + m['ch'])).resize((m['W'], m['H']), Image.LANCZOS).save(
            vd / 'raw' / f'{p:04d}.png')


# ---------- post: key, align, downscale ----------

def key_mask(rgb, tol=26, hole_tol=14, min_hole=60):
    """Foreground mask of a generation on a flat background: flood fill from the border with `tol`, plus enclosed
    background-coloured pockets (gaps between limbs) bigger than `min_hole` px within `hole_tol`."""
    a = np.asarray(rgb).astype(np.int16)
    H, W, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    d = np.abs(a - bg).max(axis=2)
    near = d <= tol
    seen = np.zeros((H, W), bool)
    q = deque()
    for x in range(W):
        for y in (0, H - 1):
            if near[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    for y in range(H):
        for x in (0, W - 1):
            if near[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < H and 0 <= xx < W and near[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True; q.append((yy, xx))
    bgm = seen
    # enclosed pockets
    try:
        import cv2
        pocket = ((d <= hole_tol) & ~bgm).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(pocket, 4)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_hole:
                bgm |= lab == i
        fg = (~bgm).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n > 1:  # keep components that are not tiny specks
            big = max(st[1:, cv2.CC_STAT_AREA])
            keep = np.zeros_like(fg, bool)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] >= max(40, big * 0.002):
                    keep |= lab == i
            return keep, bg
        return fg.astype(bool), bg
    except ImportError:
        return ~bgm, bg


def best_shift(gen_m, ref_m, max_shift=64, scales=tuple(np.round(np.arange(0.84, 1.121, 0.02), 3))):
    """Scale (about the ref bbox bottom-centre) + translation that best overlays gen mask on the ref mask (IoU)."""
    import cv2
    H, W = ref_m.shape
    ys, xs = np.nonzero(ref_m)
    cx, cy = xs.mean(), ys.max()
    best = (-1, 1.0, 0, 0)
    small = 4
    rm = cv2.resize(ref_m.astype(np.uint8), (W // small, H // small), interpolation=cv2.INTER_AREA) > 0
    for sc in scales:
        M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
        g = cv2.warpAffine(gen_m.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        gs = cv2.resize(g, (W // small, H // small), interpolation=cv2.INTER_AREA) > 0
        for dy in range(-max_shift // small, max_shift // small + 1):
            for dx in range(-max_shift // small, max_shift // small + 1):
                sh = np.roll(np.roll(gs, dy, 0), dx, 1)
                inter = (sh & rm).sum(); uni = (sh | rm).sum()
                iou = inter / max(uni, 1)
                if iou > best[0]:
                    best = (iou, sc, dx * small, dy * small)
    iou, sc, dx, dy = best
    return np.float32([[sc, 0, cx * (1 - sc) + dx], [0, sc, cy * (1 - sc) + dy]]), iou


def post(bank, variant, pats, final=False):
    import cv2
    vd = OUT / bank / variant
    (vd / 'frames').mkdir(parents=True, exist_ok=True)
    (vd / 'frames4x').mkdir(parents=True, exist_ok=True)
    stats = {}
    for p in pats:
        fr = frame(bank, p)
        w, h = fr.size
        s, W, H, ox, oy = geom(w, h)
        raw = Image.open(vd / 'raw' / f'{p:04d}.png').convert('RGB')
        if raw.size != (W, H):
            raw = raw.resize((W, H), Image.LANCZOS)
        gm, bgc = key_mask(raw)
        ref_m = np.zeros((H, W), bool)
        big_a = np.asarray(fr.resize((w * s, h * s), Image.NEAREST))[:, :, 3] > 0
        ref_m[oy:oy + h * s, ox:ox + w * s] = big_a
        iou0 = (gm & ref_m).sum() / max((gm | ref_m).sum(), 1)
        M, iou = best_shift(gm, ref_m)
        rgb = cv2.warpAffine(np.asarray(raw), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=bgc.tolist())
        m = cv2.warpAffine(gm.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_LINEAR)
        rgba = np.dstack([rgb, m]).astype(np.uint8)
        crop = Image.fromarray(rgba).crop((ox, oy, ox + w * s, oy + h * s))
        # 4x copy: smooth resample of the crop
        four = crop.resize((w * 4, h * 4), Image.LANCZOS)
        fa = np.asarray(four).copy(); fa[:, :, 3] = np.where(fa[:, :, 3] >= 128, 255, 0)
        Image.fromarray(fa).save(vd / 'frames4x' / f'{p:04d}.png')
        # 1x: area-average colour of foreground pixels only, hard alpha
        ca = np.asarray(crop).astype(np.float32)
        al = ca[:, :, 3:] / 255.0
        prem = cv2.resize(ca[:, :, :3] * al, (w, h), interpolation=cv2.INTER_AREA)
        cov = cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA)[:, :, None]
        col = np.where(cov > 1e-3, prem / np.maximum(cov, 1e-3), 0)
        one = np.dstack([np.clip(col, 0, 255), np.where(cov[:, :, 0] >= 0.5, 255, 0)]).astype(np.uint8)
        Image.fromarray(one).save(vd / 'frames' / f'{p:04d}.png')
        a1 = one[:, :, 3] > 0
        a0 = np.asarray(fr)[:, :, 3] > 0
        iou1 = (a1 & a0).sum() / max((a1 | a0).sum(), 1)
        stats[p] = dict(iou_raw=round(float(iou0), 3), iou_aligned=round(float(iou), 3), iou_1x=round(float(iou1), 3),
                        scale=round(float(M[0, 0]), 3), shift=[round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)])
        print(p, stats[p])
    (vd / 'stats.json').write_text(json.dumps(stats, indent=1))
    return stats


# ---------- comparison sheet ----------

def on_gray(im, size):
    c = Image.new('RGBA', size, BG + (255,))
    im = im.convert('RGBA')
    sc = min(size[0] / im.width, size[1] / im.height)
    im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.NEAREST)
    c.alpha_composite(im, ((size[0] - im.width) // 2, size[1] - im.height - 6))
    return c


def sheet(bank, variant, pats, out=None, cell=240):
    vd = OUT / bank / variant
    codex = None
    cx = OUT / bank / 'idle_codex.png'
    if cx.exists():
        codex = Image.open(cx).convert('RGBA')
    rows = []
    for p in pats:
        cols = [on_gray(frame(bank, p), (cell, cell))]
        if codex is not None and 1 <= p <= 12:
            i = p - 1
            cw, ch = codex.width / 4, codex.height / 3
            x, y = i % 4, i // 4
            cols.append(codex.crop((round(x * cw) + 3, round(y * ch) + 3, round((x + 1) * cw) - 3,
                                    round((y + 1) * ch) - 3)).resize((cell, cell), Image.LANCZOS))
        else:
            cols.append(Image.new('RGBA', (cell, cell), (60, 60, 60, 255)))
        cols.append(on_gray(Image.open(vd / 'frames' / f'{p:04d}.png'), (cell, cell)))
        cols.append(on_gray(Image.open(vd / 'frames4x' / f'{p:04d}.png'), (cell, cell)))
        rows.append(cols)
    lab = 22
    img = Image.new('RGBA', (cell * 4 + 15, lab + len(rows) * (cell + 3)), (30, 30, 30, 255))
    d = ImageDraw.Draw(img)
    for j, t in enumerate(['original', 'codex', 'local 1x (game size)', 'local 4x']):
        d.text((j * (cell + 5) + 6, 5), t, fill=(240, 240, 240, 255))
    for i, cols in enumerate(rows):
        for j, c in enumerate(cols):
            img.paste(c, (j * (cell + 5), lab + i * (cell + 3)))
        d.text((4, lab + i * (cell + 3) + 4), str(pats[i]), fill=(255, 255, 0, 255))
    out = out or vd / 'compare.png'
    img.convert('RGB').save(out)
    print('sheet', out)


def parse_pats(args):
    out = []
    for a in args:
        if '-' in a:
            x, y = a.split('-'); out += list(range(int(x), int(y) + 1))
        else:
            out.append(int(a))
    return out


if __name__ == '__main__':
    cmd, bank, variant = sys.argv[1:4]
    pats = parse_pats(sys.argv[4:])
    {'prep': prep, 'post': post, 'sheet': sheet, 'prepsheet': prep_sheet,
     'split': lambda b, v, p: split_sheet(b, v)}[cmd](bank, variant, pats)
