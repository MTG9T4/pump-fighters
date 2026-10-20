"""Per-frame redraw through pose sheets: original SF3 frames go onto a grid sheet, Codex redraws the sheet with the
mascot in the same poses, and the redrawn sheet is cut back into frames at the original canvas size.

  python sf3mod/redraw/sheets.py make    <bank> <name> <patterns...>   # src sheet + layout for Codex
  python sf3mod/redraw/sheets.py post    <bank> <name>                 # Codex sheet -> keyed, aligned 1x frames
  python sf3mod/redraw/sheets.py compare <bank> <name> [out.png]       # original | current mesh | redraw

Files live in sf3mod/build/redraw/<bank>/sheets/<name>/ (src.png, layout.json, codex.png, frames/, stats.json).
Finished frames are copied to sf3mod/build/redraw/<bank>/final/frames/, which skin.py uses in place of the mesh.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redraw import key_mask, best_shift  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SKIN = ROOT / 'sf3mod' / 'build' / 'skin'
OUT = ROOT / 'sf3mod' / 'build' / 'redraw'
BG = (128, 128, 128)
GRID = (96, 96, 96)
PAD = 24      # old sheets (GIGA): cell padding in sheet px, no margin kept
SCALES = tuple(np.round(np.arange(0.74, 1.361, 0.02), 3))  # Codex often draws the mascot up to ~1.3x
MIN_IOU = 0.35  # below this overlap with the original the redraw is not used (mesh fallback)
MARGIN = 32   # sprite px kept around the original canvas (skin.py allows up to M = 32)


def frame(bank, p):
    return Image.open(SKIN / bank / 'frames' / f'{p:04d}.png').convert('RGBA')


def sheet_dir(bank, name):
    return OUT / bank / 'sheets' / name


def make(bank, name, pats, cols=4, cell_max=340):
    pose = json.loads((SKIN / bank / 'pose.json').read_text())
    boxes = {p: (*pose[str(p)]['origin'], *pose[str(p)]['size']) for p in pats}  # ox, oy, w, h (origin space)
    xmin = min(b[0] for b in boxes.values()); ymin = min(b[1] for b in boxes.values())
    xmax = max(b[0] + b[2] for b in boxes.values()); ymax = max(b[1] + b[3] for b in boxes.values())
    S = max(2, int(cell_max // max(xmax - xmin, ymax - ymin)))
    pad = MARGIN * S  # room for tails, capes and manes that reach past the original fighter
    cw, ch = (xmax - xmin) * S + 2 * pad, (ymax - ymin) * S + 2 * pad
    rows = -(-len(pats) // cols)
    sheet = Image.new('RGB', (cols * cw, rows * ch), BG)
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(pats):
        ox, oy, w, h = boxes[p]
        cx, cy = (i % cols) * cw, (i // cols) * ch
        im = frame(bank, p).resize((w * S, h * S), Image.NEAREST)
        sheet.paste(im, (cx + pad + (ox - xmin) * S, cy + pad + (oy - ymin) * S), im)
    for c in range(1, cols): d.line([(c * cw, 0), (c * cw, rows * ch)], fill=GRID, width=2)
    for r in range(1, rows): d.line([(0, r * ch), (cols * cw, r * ch)], fill=GRID, width=2)
    # the image model only draws a few sheet shapes (1:1, 3:2, 2:3): pad with plain grey (right/bottom) to the closest
    # one, so the redraw comes back at the same aspect and no frame gets squashed
    gw, gh = sheet.size
    sw, sh = min(((max(gw, round(gh * a)), max(gh, round(gw / a))) for a in (1.5, 1.0, 2 / 3)), key=lambda t: t[0] * t[1])
    if (sw, sh) != (gw, gh):
        padded = Image.new('RGB', (sw, sh), BG); padded.paste(sheet, (0, 0)); sheet = padded
    sd = sheet_dir(bank, name); sd.mkdir(parents=True, exist_ok=True)
    sheet.save(sd / 'src.png')
    (sd / 'layout.json').write_text(json.dumps(dict(pats=pats, cols=cols, rows=rows, S=S, cw=cw, ch=ch, pad=pad, sheet=[sw, sh],
                                                    xmin=xmin, ymin=ymin, boxes=boxes)))
    print(sd / 'src.png', sheet.size, 'scale', S)


def grid_lines(img, frac=0.5):
    """Rows and columns where Codex drew a grid line (flat dark grey across most of the sheet)."""
    a = np.asarray(img).astype(np.int16)
    v = a.mean(2); grey = (a.max(2) - a.min(2) < 14) & (v > 55) & (v < 118)
    return np.nonzero(grey.mean(1) > frac)[0], np.nonzero(grey.mean(0) > frac)[0]


def snap(nominal, lines, reach=45):
    """Offset from a nominal cell border to the nearest drawn grid line within `reach` px (0 when none)."""
    near = [int(l) - nominal for l in lines if abs(int(l) - nominal) <= reach]
    return min(near, key=abs) if near else 0


def erase_grid(img, frac=0.5):
    """Paint Codex's own grid lines (flat dark-grey rows/columns crossing most of the sheet, often drawn a few px off
    the true cell borders) with the background, so they never end up inside a cut frame."""
    a = np.asarray(img).astype(np.int16).copy()
    v = a.mean(2); grey = (a.max(2) - a.min(2) < 14) & (v > 55) & (v < 118)
    for y in np.nonzero(grey.mean(1) > frac)[0]:
        a[max(0, y - 1):y + 2][grey[max(0, y - 1):y + 2]] = BG
    for x in np.nonzero(grey.mean(0) > frac)[0]:
        sl = grey[:, max(0, x - 1):x + 2]
        blk = a[:, max(0, x - 1):x + 2]; blk[sl] = BG; a[:, max(0, x - 1):x + 2] = blk
    return Image.fromarray(a.astype(np.uint8))


def post(bank, name):
    import cv2
    sd = sheet_dir(bank, name)
    L = json.loads((sd / 'layout.json').read_text())
    cols, rows, S, cw, ch = L['cols'], L['rows'], L['S'], L['cw'], L['ch']
    raw = Image.open(sd / 'codex.png').convert('RGB')
    sw, sh = L.get('sheet', [cols * cw, rows * ch])
    if abs((raw.width / raw.height) / (sw / sh) - 1) > 0.03:
        # Codex drew a different sheet shape: stretching it would squash every frame, so reject it for a redraw
        (sd / 'codex.png').rename(sd / 'codex_badaspect.png')
        print(f'{bank} {name}: codex sheet {raw.size} has the wrong aspect for {sw}x{sh}; rejected')
        return
    raw = raw.resize((sw, sh), Image.LANCZOS)
    # Codex often draws its grid lines (and the rows/columns with them) up to ~40 px off the true cell borders:
    # move each row's/column's cut window by the offset of the line Codex actually drew
    ys, xs = grid_lines(raw)
    rowoff = [0] + [snap(r * ch, ys) for r in range(1, rows)]
    coloff = [0] + [snap(c * cw, xs) for c in range(1, cols)]
    raw = erase_grid(raw)
    big = Image.new('RGB', (cols * cw + 2 * 48, rows * ch + 2 * 48), BG); big.paste(raw, (48, 48))
    (sd / 'frames').mkdir(exist_ok=True)
    stats = {}
    for i, p in enumerate(L['pats']):
        ox, oy, w, h = L['boxes'][str(p)]
        cx, cy = (i % cols) * cw, (i // cols) * ch
        ox_, oy_ = 48 + cx + coloff[i % cols], 48 + cy + rowoff[i // cols]
        cell = np.asarray(big.crop((ox_, oy_, ox_ + cw, oy_ + ch))).copy()
        b = 6  # paint over the grid lines so the key sees one flat border
        bgc = np.array(BG)
        cell[:b], cell[-b:], cell[:, :b], cell[:, -b:] = bgc, bgc, bgc, bgc
        gm, bg = key_mask(Image.fromarray(cell))
        pad = L.get('pad', PAD); mg = min(MARGIN, pad // S)  # sprite px kept around the original canvas
        fx, fy = pad + (ox - L['xmin']) * S, pad + (oy - L['ymin']) * S
        ref = np.zeros((ch, cw), bool)
        ref[fy:fy + h * S, fx:fx + w * S] = np.asarray(frame(bank, p).resize((w * S, h * S), Image.NEAREST))[:, :, 3] > 0
        M, iou = best_shift(gm, ref, max_shift=40, scales=SCALES)
        rgb = cv2.warpAffine(cell, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=bg.tolist())
        m = cv2.warpAffine(gm.astype(np.uint8) * 255, M, (cw, ch), flags=cv2.INTER_LINEAR)
        crop = np.dstack([rgb, m])[fy - mg * S:fy + (h + mg) * S, fx - mg * S:fx + (w + mg) * S].astype(np.float32)
        al = crop[:, :, 3:] / 255.0
        W2, H2 = w + 2 * mg, h + 2 * mg
        prem = cv2.resize(crop[:, :, :3] * al, (W2, H2), interpolation=cv2.INTER_AREA)
        cov = cv2.resize(al, (W2, H2), interpolation=cv2.INTER_AREA)[:, :, None]
        col = np.where(cov > 1e-3, prem / np.maximum(cov, 1e-3), 0)
        one = np.dstack([np.clip(col, 0, 255), np.where(cov[:, :, 0] >= 0.5, 255, 0)]).astype(np.uint8)
        a0 = np.asarray(frame(bank, p))[:, :, 3] > 0
        a1 = one[mg:mg + h, mg:mg + w, 3] > 0
        iou1 = float((a0 & a1).sum() / max((a0 | a1).sum(), 1))
        stats[p] = dict(iou_aligned=round(float(iou), 3), iou_1x=round(iou1, 3), scale=round(float(M[0, 0]), 3))
        out = sd / 'frames' / f'{p:04d}.png'
        if iou1 < MIN_IOU:  # Codex left the cell (nearly) empty or drew something else: keep the mesh for this frame
            stats[p]['fallback'] = True
            if out.exists(): out.unlink()
        else:
            Image.fromarray(one).save(out)
        print(p, stats[p])
    (sd / 'stats.json').write_text(json.dumps(stats, indent=1))


def publish(bank, name):
    sd = sheet_dir(bank, name); fd = OUT / bank / 'final' / 'frames'; fd.mkdir(parents=True, exist_ok=True)
    for f in (sd / 'frames').glob('*.png'): shutil.copy2(f, fd / f.name)
    st = json.loads((sd / 'stats.json').read_text()) if (sd / 'stats.json').exists() else {}
    for p, v in st.items():
        if v.get('fallback') and (fd / f'{int(p):04d}.png').exists(): (fd / f'{int(p):04d}.png').unlink()
    print('published', len(list((sd / 'frames').glob('*.png'))), 'frames to', fd)


def compare(bank, name, out=None, z=3):
    sys.path.insert(0, str(ROOT / 'sf3mod'))
    import skin
    sd = sheet_dir(bank, name)
    L = json.loads((sd / 'layout.json').read_text())
    pats = L['pats']
    bk, pose, parts, G, cfg, hooks = skin.setup(bank)
    Js = skin.prepare_poses(bank, bk, pose, G, dict(cfg, complete=True), hooks, set(pats))
    tiles = []
    for p in pats:
        W, H = pose[str(p)]['size']
        orig = frame(bank, p)
        mesh = Image.fromarray(np.asarray(skin.render(Js[p], parts, G, cfg, (W, H)))[skin.M:skin.M + H, skin.M:skin.M + W]) \
            if p in Js else Image.new('RGBA', (W, H))
        # the redraw as the game shows it: the try build's palette-snapped preview (cropped to its bbox), else 1x
        red = Image.open(sd / 'frames' / f'{p:04d}.png').convert('RGBA')
        pv = SKIN / bank / 'preview' / f'{p:04d}.png'
        if pv.exists() and pv.stat().st_mtime > (sd / 'frames' / f'{p:04d}.png').stat().st_mtime:
            box = red.getbbox(); game = Image.new('RGBA', (W, H)); g = Image.open(pv).convert('RGBA')
            if box: game.alpha_composite(g.crop((0, 0, min(g.width, W - box[0]), min(g.height, H - box[1]))), box[:2])
            red = game
        row = Image.new('RGBA', ((W * 3 + 16) * z, H * z + 18), (40, 40, 50, 255))
        for k, im in enumerate((orig, mesh, red)):
            row.alpha_composite(im.resize((W * z, H * z), Image.NEAREST), (k * (W + 8) * z, 18))
        ImageDraw.Draw(row).text((4, 2), f'{p}  original | current build | redraw (game palette)', fill=(255, 220, 0, 255))
        tiles.append(row)
    cols = 2
    cwid = max(t.width for t in tiles); rh = [max(t.height for t in tiles[r * cols:(r + 1) * cols]) for r in range(-(-len(tiles) // cols))]
    sheet = Image.new('RGBA', (cwid * cols + 20, sum(rh) + 10 * len(rh)), (24, 24, 30, 255))
    y = 0
    for r in range(len(rh)):
        for c in range(cols):
            k = r * cols + c
            if k < len(tiles): sheet.alpha_composite(tiles[k], (c * (cwid + 20), y))
        y += rh[r] + 10
    out = Path(out) if out else sd / 'compare.png'
    sheet.save(out); print(out, sheet.size)


def plan(bank, idle_n=12, per=12, max_w=250, max_h=210):
    """Frames to redraw (posed + re-coloured frames; body pieces bigger than a head as 'pieces'), grouped by animation
    into sheets of up to `per` frames whose union box stays small. The first idle frames form sheet 'idle' (the style
    reference for the rest). Writes frames_info.json and plan.json under build/redraw/<bank>/."""
    sys.path.insert(0, str(ROOT / 'sf3mod'))
    import skin
    bk, pose, parts, G, cfg, hooks = skin.setup(bank)
    Js = skin.prepare_poses(bank, bk, pose, G, dict(cfg, complete=True), hooks, None)
    info = {}
    for p, J in Js.items():
        kind = 'piece' if J.get('_piece') is not None else 'keep' if J.get('_keep') is not None else             'treat' if J.get('_treat') else 'plain'
        info[p] = dict(kind=kind, size=pose[str(p)]['size'])
    od = OUT / bank; od.mkdir(parents=True, exist_ok=True)
    (od / 'frames_info.json').write_text(json.dumps({str(k): v for k, v in info.items()}))
    anims = json.loads((SKIN / bank / 'anims.json').read_text())
    seq = list(anims.values())[0] if len(anims) == 1 else anims
    seq = seq if isinstance(seq, list) else list(seq.values())
    want = {p for p, v in info.items() if v['kind'] in ('plain', 'treat')}
    order, seen = [], set()
    for an in seq:
        for f in an['frames']:
            p = f[0] if isinstance(f, list) else f
            if p in want and p not in seen: seen.add(p); order.append(p)
    order += sorted(want - seen)
    idle = order[:idle_n]; rest = order[idle_n:]

    def box(p):
        ox, oy = pose[str(p)]['origin']; w, h = pose[str(p)]['size']; return ox, oy, ox + w, oy + h
    sheets, cur, U = [], [], None
    for p in rest:
        b = box(p)
        nu = b if U is None else (min(U[0], b[0]), min(U[1], b[1]), max(U[2], b[2]), max(U[3], b[3]))
        if cur and (len(cur) == per or nu[2] - nu[0] > max_w or nu[3] - nu[1] > max_h):
            sheets.append(cur); cur = []; nu = b
        cur.append(p); U = nu
    if cur: sheets.append(cur)
    pieces = [p for p, v in info.items() if v['kind'] == 'piece' and max(v['size']) > 40]
    pl = {'idle': idle, **{f's{i:03d}': s for i, s in enumerate(sheets)}}
    for i in range(0, len(pieces), per): pl[f'pieces{i // per}'] = pieces[i:i + per]
    (od / 'plan.json').write_text(json.dumps(pl))
    print(bank, len(want), 'frames +', len(pieces), 'pieces ->', len(pl), 'sheets')


def review(bank, per_image=3, width=1250):
    """Side-by-side review images (source sheet | Codex sheet), a few sheets per image, in build/redraw/<bank>/review/."""
    plan = json.loads((OUT / bank / 'plan.json').read_text())
    rd = OUT / bank / 'review'; rd.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in plan:
        sd = sheet_dir(bank, n)
        if not (sd / 'codex.png').exists(): continue
        a = Image.open(sd / 'src.png').convert('RGB'); b = Image.open(sd / 'codex.png').convert('RGB').resize(a.size)
        half = (width - 10) // 2; sc = half / a.width
        a = a.resize((half, int(a.height * sc))); b = b.resize(a.size)
        r = Image.new('RGB', (width, a.height + 16), (20, 20, 26)); r.paste(a, (0, 16)); r.paste(b, (half + 10, 16))
        ImageDraw.Draw(r).text((4, 2), f'{bank} {n}', fill=(255, 220, 0)); rows.append(r)
    for k in range(0, len(rows), per_image):
        g = rows[k:k + per_image]; o = Image.new('RGB', (width, sum(r.height for r in g))); y = 0
        for r in g: o.paste(r, (0, y)); y += r.height
        o.save(rd / f'r{k // per_image:02d}.png')
    print(rd, -(-len(rows) // per_image), 'images')


if __name__ == '__main__':
    cmd, bank, *rest = sys.argv[1:]
    name = rest.pop(0) if rest and cmd not in ('plan', 'review') else None
    if cmd == 'make': make(bank, name, [int(x) for x in rest])
    elif cmd == 'post': post(bank, name)
    elif cmd == 'publish': publish(bank, name)
    elif cmd == 'compare': compare(bank, name, rest[0] if rest else None)
    elif cmd == 'plan': plan(bank)
    elif cmd == 'review': review(bank)
