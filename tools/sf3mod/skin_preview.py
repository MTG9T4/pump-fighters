"""Side-by-side review sheets for one fighter: original SF3 frame | skinned frame | skeleton overlay.

    python sf3mod/skin_preview.py pl18                       # 24 frames spread over the bank
    python sf3mod/skin_preview.py pl18 --frames 8,16,40      # chosen patterns
    python sf3mod/skin_preview.py pl18 --range 100 160       # every pattern in a range (animations are mostly contiguous)
    python sf3mod/skin_preview.py pl18 --standing            # the idle animation's frames
Writes build/skin/<bank>/review/<name>.png (default name: sheet) and prints the path. No game files are written.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD)); sys.path.insert(0, str(MOD / 'tools'))
import skin

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


def tour(bank_name, pose):
    """~40 patterns over the move set, from the fighter's animation scripts (build/skin/<bank>/anims.json)."""
    an = json.loads((MOD / 'build' / 'skin' / bank_name / 'anims.json').read_text())['anims']
    pick = dict(nmca=((0, 2, 3, 4, 5, 6, 8, 14), (0.3, 0.7)), atca=(range(0, 24, 2), (0.5,)),
                saca=((0, 2, 4, 6, 8), (0.35, 0.7)), dmca=((0, 3, 6, 10, 16, 22, 30, 40), (0.5,)))
    out = []
    for kind, (idxs, fr) in pick.items():
        an_k = [x for x in an if x['name'] == kind]
        for i in idxs:
            if i >= len(an_k): continue
            seq = [p for p, _ in an_k[i]['frames'] if str(p) in pose and 'kp' in pose[str(p)]]
            for t in fr:
                if seq:
                    p = seq[min(len(seq) - 1, int(t * len(seq)))]
                    if p not in out: out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank'); ap.add_argument('--frames'); ap.add_argument('--range', nargs=2, type=int)
    ap.add_argument('--standing', action='store_true'); ap.add_argument('--count', type=int, default=24)
    ap.add_argument('--tour', action='store_true',
                    help='a tour of the move set: idle, walks, jumps, crouch, normals, specials, hits, knockdowns')
    ap.add_argument('--name', default='sheet'); ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--compare', action='store_true',
                    help='columns original | old cut-out puppet | mesh skin (needs the mascot mesh art)')
    a = ap.parse_args()
    bank, pose, parts, G, cfg, hooks = skin.setup(a.bank)
    if a.frames: frames = [int(x) for x in a.frames.split(',')]
    elif a.range: frames = list(range(a.range[0], a.range[1] + 1))
    elif a.standing: frames = bank.standing_patterns(8)
    elif a.tour: frames = tour(a.bank, pose)
    else:
        good = [int(p) for p, r in pose.items() if 'kp' in r and r['size'][1] >= 40]
        good.sort(); frames = [good[int(i * len(good) / a.count)] for i in range(a.count)] if good else []
    Js = skin.prepare_poses(a.bank, bank, pose, G, cfg, hooks, set(frames))
    cells = []
    for p in frames:
        rec = pose[str(p)]
        orig = Image.open(MOD / 'build' / 'skin' / a.bank / 'frames' / f'{p:04d}.png').convert('RGBA') \
            if (MOD / 'build' / 'skin' / a.bank / 'frames' / f'{p:04d}.png').exists() else Image.new('RGBA', (8, 8))
        W, H = orig.size; m = skin.M
        o = Image.new('RGBA', (W + 2 * m, H + 2 * m), (52, 52, 64, 255)); o.alpha_composite(orig, (m, m))
        if p in Js:
            sk = Image.fromarray(skin.render(Js[p], parts, G, cfg, (W, H)), 'RGBA')
        else:
            sk = Image.new('RGBA', o.size, (120, 30, 30, 120))
        s2 = Image.new('RGBA', o.size, (52, 52, 64, 255)); s2.alpha_composite(sk)
        if a.compare:  # old cut-out next to the mesh skin (sk above is the active renderer: mesh)
            mesh_model, parts.mesh = parts.mesh, None
            try:
                old = Image.fromarray(skin.render(Js[p], parts, G, cfg, (W, H)), 'RGBA') if p in Js else sk
            finally:
                parts.mesh = mesh_model
            s1 = Image.new('RGBA', o.size, (52, 52, 64, 255)); s1.alpha_composite(old)
            row = Image.new('RGBA', (o.width * 3 + 8, o.height + 12), (30, 30, 36, 255))
            for k, im in enumerate((o, s1, s2)): row.alpha_composite(im, (k * (o.width + 4), 12))
            ImageDraw.Draw(row).text((2, 0), f'{p}' + ('' if p in Js else ' (colour-mapped)') +
                                     ' | orig | cut-out | mesh', fill='yellow')
            cells.append(row)
            continue
        ov = o.copy(); ov.alpha_composite(Image.fromarray((np.asarray(sk) * [1, 1, 1, 0.45]).astype(np.uint8), 'RGBA'))
        d = ImageDraw.Draw(ov)
        if 'kp' in rec:
            kp = np.array(rec['kp']) + m; sc = np.array(rec['score'])
            for i, j in BONES:
                col = (255, 80, 80) if i % 2 else (80, 160, 255)  # odd COCO ids = left side (red), even = right (blue)
                if sc[i] > .3 and sc[j] > .3: d.line([tuple(kp[i]), tuple(kp[j])], fill=col, width=1)
        if p in Js:
            J = Js[p]
            for side, col in (('l', (255, 200, 0)), ('r', (0, 255, 200))):
                pts = [J[side][k] for k in ('hip', 'kne', 'ank')]
                d.line([tuple(np.array(q) + m) for q in pts], fill=col, width=1)
            d.ellipse(tuple(np.r_[np.array(J['head']) + m - 2, np.array(J['head']) + m + 2]), outline=(255, 255, 255))
        row = Image.new('RGBA', (o.width * 3 + 8, o.height + 12), (30, 30, 36, 255))
        for k, im in enumerate((o, s2, ov)): row.alpha_composite(im, (k * (o.width + 4), 12))
        ImageDraw.Draw(row).text((2, 0), f'{p}' + ('' if p in Js else ' (colour-mapped)') +
                                 (f" face={Js[p]['face']}" if p in Js else ''), fill='yellow')
        cells.append(row)
    if not cells: print('no frames'); return
    cols = 4 if len(cells) > 8 else 2
    cw = max(c.width for c in cells); ch = max(c.height for c in cells)
    sheet = Image.new('RGBA', (cw * cols, ch * ((len(cells) + cols - 1) // cols)), (20, 20, 24, 255))
    for i, c in enumerate(cells): sheet.alpha_composite(c, ((i % cols) * cw, (i // cols) * ch))
    sheet = sheet.resize((sheet.width * a.scale, sheet.height * a.scale), Image.NEAREST)
    out = MOD / 'build' / 'skin' / a.bank / 'review'; out.mkdir(parents=True, exist_ok=True)
    path = out / f'{a.name}.png'; sheet.save(path); print(path)


if __name__ == '__main__':
    main()
