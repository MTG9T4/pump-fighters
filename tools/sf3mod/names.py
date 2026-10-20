"""Redraw the select-screen and win-screen name sprites in ef02_usa.bin with the mascot names.

Names share palette rows with every other name, so the new text is drawn in the existing colours:
heavy italic white letters with a dark outline, snapped to the row's palette, kept inside the original tiles.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from spritebank import Palettes

MOD = Path(__file__).resolve().parent
FONTS = ['C:/Windows/Fonts/ariblk.ttf', 'C:/Windows/Fonts/impact.ttf', 'C:/Windows/Fonts/arialbd.ttf']


def font(size):
    for f in FONTS:
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()


def text_image(text, w, h, italic=0.22):
    """White heavy italic text with a black outline, fitted into w x h."""
    size = h + 6
    while size > 6:
        f = font(size)
        bw = int(f.getlength(text)) + 8
        bh = size + 8
        if bh <= h * 1.6: break
        size -= 1
    img = Image.new('RGBA', (bw + bh, bh + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((4 + bh // 3, 2), text, font=f, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(16, 16, 24, 255))
    img = img.crop(img.getbbox())
    # italic shear
    sh = img.transform((img.width + int(img.height * italic), img.height), Image.AFFINE,
                       (1, italic, -img.height * italic, 0, 1, 0), Image.BICUBIC)
    sh = sh.crop(sh.getbbox())
    # fit: scale to height, condense horizontally if it is too wide
    target_h = h - 2
    s = target_h / sh.height
    nw = int(sh.width * s)
    if nw > w - 2: nw = w - 2
    return sh.resize((max(1, nw), target_h), Image.LANCZOS)


def place(bank, pattern, pal, text):
    img, meta = bank.render(pattern, pal, metadata=True)
    W, H = img.size
    tiles = bank.tiles(pattern)
    cover = np.zeros((H, W), bool)
    owner = np.full((H, W), -1, np.int32); first = owner.copy()
    for t in tiles:
        x, y = t['x'] - meta['origin'][0], t['y'] - meta['origin'][1]
        reg = np.s_[y:y + t['h'], x:x + t['w']]
        cover[reg] = True
        o = owner[reg]; o[(o < 0) & (t['pixels'] != 0)] = t['index']
        f = first[reg]; f[f < 0] = t['index']
    owner = np.where(owner >= 0, owner, first)
    # usable box: rows/cols that are covered
    ys, xs = np.nonzero(cover)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # search size and offset so the letters land inside the tiles that exist (they follow the old letters)
    best = None
    for hs in (1.0, 0.92, 0.85, 0.78, 0.7):
        th = max(8, int((y1 - y0) * hs))
        for ws in (1.0, 0.9, 0.8):
            txt = text_image(text, int((x1 - x0) * ws), th)
            al = np.asarray(txt)[:, :, 3] > 110
            for ox in range(int(x0) - 4, int(x1) - txt.width + 5):
                for oy in range(int(y0), max(int(y0), int(y1) - txt.height) + 1):
                    xs, ys = max(0, ox), max(0, oy)
                    xe, ye = min(W, ox + txt.width), min(H, oy + txt.height)
                    if xe <= xs or ye <= ys: continue
                    sub = al[ys - oy:ye - oy, xs - ox:xe - ox]
                    inside = (sub & cover[ys:ye, xs:xe]).sum()
                    lost = al.sum() - inside
                    score = inside - 6 * lost + 0.5 * al.sum()
                    if best is None or score > best[0]: best = (score, txt, ox, oy)
    _, txt, ox, oy = best
    canvas = Image.new('RGBA', (W + 64, H + 64), (0, 0, 0, 0))
    canvas.alpha_composite(txt, (ox + 32, oy + 32))
    canvas = canvas.crop((32, 32, 32 + W, 32 + H))
    a = np.asarray(canvas).copy()
    a[:, :, 3] = np.where((a[:, :, 3] >= 110) & cover, 255, 0)
    out = np.zeros_like(a)
    for t in tiles:
        m = (owner == t['index']) & (a[:, :, 3] > 0)
        if not m.any(): continue
        p = pal.for_tile(t['attr'], meta['palette_base'])
        opaque = np.flatnonzero(p[:, 3] > 0)
        cols = p[opaque][:, :3].astype(np.int32)
        px = a[m][:, :3].astype(np.int32)
        d = ((px[:, None, :] - cols[None]) ** 2).sum(axis=2)
        out[m] = p[opaque[d.argmin(axis=1)]]
    return Image.fromarray(out, 'RGBA')


def build_names(bank, default, names, cast):
    pmap = {e['id']: e for e in json.loads((MOD / 'build' / 'ef02' / 'pattern_map.json').read_text())}
    pal = Palettes(default.astype('<u2').tobytes(), False)
    for mascot, c in cast.items():
        for kind in ('name', 'win_name'):
            pattern = pmap[c['id']][kind]['pattern']
            img = place(bank, pattern, pal, names[mascot])
            img.save(MOD / 'build' / f'{kind}_{mascot}.png')
            bank.data = bank.replace(pattern, img, pal)
            bank.__init__(bank.data, bank.info)
        print(f'names {mascot:9s} -> {names[mascot]}')
