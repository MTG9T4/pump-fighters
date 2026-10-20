"""Build the PUMP COMBAT override files for SF33RD.AFS.

Reads the extracted originals in sf3mod/build/extracted/ (python sf3mod/tools/afs.py extract ...),
writes replacements to sf3mod/build/overrides/, then repacks game/SF33RD.AFS into sf3mod/build/SF33RD.AFS.
Never touches the original archive.

    python sf3mod/build_mod.py [--only logo,portraits,names,palettes,hud,skins] [--no-repack] [--out DIR]

--out DIR builds into DIR/overrides and DIR/SF33RD.AFS instead of sf3mod/build/ (for test builds; seed
DIR/overrides with a copy of sf3mod/build/overrides first if the parts you skip should still be in it).
"""
import argparse, json, struct, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'sf3mod'
EXT = MOD / 'build' / 'extracted'
OUT = MOD / 'build' / 'overrides'
GEN = MOD / 'assets' / 'gen'
sys.path.insert(0, str(MOD / 'tools'))
from spritebank import SpriteBank, load_palettes, Palettes  # noqa: E402
from runtime_compat import prepare_character_bank  # noqa: E402

# mascot -> SF3 body (character id, ef02 portrait / select name / win name patterns)
CAST = {
    'swordcat': dict(body='Ryu', id=2), 'inu': dict(body='Ken', id=11), 'jeanphil': dict(body='Dudley', id=4),
    'elon': dict(body='Urien', id=13), 'pepe': dict(body='Necro', id=5), 'claudia': dict(body='Chun-Li', id=15),
    'wif': dict(body='Yun', id=3), 'popcat': dict(body='Makoto', id=16), 'pnut': dict(body='Q', id=17),
    'moodeng': dict(body='Hugo', id=6),
    # spare slots
    'giga': dict(body='Alex', id=1), 'goat': dict(body='Akuma', id=14), 'chillguy': dict(body='Sean', id=12),
    'ponke': dict(body='Yang', id=10), 'mew': dict(body='Ibuki', id=7), 'pippin': dict(body='Elena', id=8),
    'slerf': dict(body='Oro', id=9), 'bonk': dict(body='Remy', id=19), 'fwog': dict(body='Twelve', id=18),
    # final boss
    'pill': dict(body='Gill', id=0),
}
NAMES = {'swordcat': 'SWORDCAT', 'inu': 'INU', 'jeanphil': 'JEAN PHIL', 'elon': 'ELON', 'pepe': 'PEPE',
         'claudia': 'CLAUDIA', 'wif': 'WIF', 'popcat': 'POPCAT', 'pnut': 'PNUT', 'moodeng': 'MOO DENG',
         'giga': 'GIGA', 'goat': 'GOAT', 'chillguy': 'CHILL GUY', 'ponke': 'PONKE', 'mew': 'MEW',
         'pippin': 'PIPPIN', 'slerf': 'SLERF', 'bonk': 'BONK', 'fwog': 'FWOG', 'pill': 'PILL'}


def argb1555(rgba):
    r, g, b, a = (int(v) for v in rgba)
    return (0x8000 if a >= 128 else 0) | ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)


def round1555(c):
    return np.array([(c[0] >> 3) << 3 | (c[0] >> 5), (c[1] >> 3) << 3 | (c[1] >> 5), (c[2] >> 3) << 3 | (c[2] >> 5), 255], np.uint8)


def key_black(im, lo=18, hi=60):
    """Black background -> alpha, with a soft edge that we later threshold to 1 bit."""
    a = np.asarray(im.convert('RGB')).astype(np.int32)  # int16 overflows on (lum - lo) * 255
    lum = a.max(axis=2)
    alpha = np.clip((lum - lo) * 255 // max(1, hi - lo), 0, 255).astype(np.uint8)
    rgba = np.dstack([a.astype(np.uint8), alpha])
    return Image.fromarray(rgba, 'RGBA')


def key_background(im, t=3, soft=24):
    """Only the generated backdrop becomes transparent. The portraits are drawn on pure black (#000, values 0-1),
    while hair, dark clothes and shadows sit at 4 and up: flood the backdrop from the image border through
    near-pure-black pixels only (4-connected). Pixels right next to it that are still very dark fade out as
    the anti-aliased edge. (The older erode-and-flood key also ate black hair and dark jackets that touched
    the border: holes in CLAUDIA's hair and ELON's, MEW's and BONK's outfits on the VS screen.)"""
    import cv2
    a = np.asarray(im.convert('RGBA')).copy()
    mx = a[:, :, :3].max(2)
    n, lab = cv2.connectedComponents((mx <= t).astype(np.uint8), connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    fringe = (cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~bg & (mx <= soft)
    alpha = np.where(bg, 0, np.where(fringe, (mx.astype(int) * 255 // soft).clip(0, 255), 255))
    a[:, :, 3] = np.minimum(a[:, :, 3], alpha).astype(np.uint8)
    return Image.fromarray(a, 'RGBA')


def add_outline(rgba, allowed, color=(12, 8, 8)):
    """Draw a 1 px dark contour just outside the opaque figure (Capcom portraits all have one), only where
    a tile can draw it. Keeps the figure from melting into the select-screen backdrop."""
    a = np.asarray(rgba).copy()
    op = a[:, :, 3] > 0
    grow = op.copy()
    grow[1:] |= op[:-1]; grow[:-1] |= op[1:]; grow[:, 1:] |= op[:, :-1]; grow[:, :-1] |= op[:, 1:]
    ring = grow & ~op & allowed
    a[ring] = (*color, 255)
    return Image.fromarray(a, 'RGBA')


# ------------------------------------------------------------------ logo
def build_logo():
    src = key_black(Image.open(GEN / 'logo.png'))
    bbox = src.getchannel('A').point(lambda v: 255 if v > 30 else 0).getbbox()
    src = src.crop(bbox)
    src.thumbnail((380, 188), Image.LANCZOS)
    canvas = Image.new('RGBA', (512, 256), (0, 0, 0, 0))
    canvas.alpha_composite(src, ((384 - src.width) // 2, (192 - src.height) // 2))
    a = np.asarray(canvas).copy()
    a[:, :, 3] = np.where(a[:, :, 3] >= 110, 255, 0)  # ARGB1555: 1-bit alpha
    a[a[:, :, 3] == 0] = 0
    png = MOD / 'build' / 'title_new.png'
    Image.fromarray(a, 'RGBA').save(png)
    run('ppg.py', 'frompng', EXT / 'TitleTM.ppg', '0', png, OUT / 'TitleTM.ppg')
    print('logo ->', OUT / 'TitleTM.ppg')


# ------------------------------------------------------------------ portraits
def frontmost_tile_map(tiles, origin, size):
    """Which tile draws each canvas pixel, mirroring SpriteBank.replace: the front-most tile whose original
    pixel is opaque wins; where every covering tile was transparent, the front-most covering tile wins."""
    owner = np.full((size[1], size[0]), -1, np.int32)
    first = np.full((size[1], size[0]), -1, np.int32)
    for t in tiles:  # front to back
        x, y = t['x'] - origin[0], t['y'] - origin[1]
        reg = np.s_[y:y + t['h'], x:x + t['w']]
        opaque = t['pixels'] != 0
        o = owner[reg]; o[(o < 0) & opaque] = t['index']
        f = first[reg]; f[f < 0] = t['index']
    return np.where(owner >= 0, owner, first)


def fit_art(art, size, coverage, shrink=1.0):
    """Place the keyed art inside the original portrait's tile coverage. Searches scale and offset to keep
    as much of the character visible as possible, weighting the head (top of the art) much more.
    shrink < 1 lowers the whole scale range (used when the pattern's tiles can't cover the art)."""
    W, H = size
    bbox = art.getchannel('A').point(lambda v: 255 if v > 40 else 0).getbbox()
    art = art.crop(bbox)
    cov = coverage.astype(np.float32)
    best = None
    small = art.resize((art.width // 4, art.height // 4))
    for scale in np.linspace(0.8, 1.35, 12) * shrink:
        h = int(H * scale); w = int(art.width * h / art.height)
        a = np.asarray(small.resize((max(1, w // 4), max(1, h // 4))))[:, :, 3] > 100
        if not a.any(): continue
        weight = np.ones(a.shape, np.float32); weight[: max(1, a.shape[0] * 2 // 5)] = 4.0  # the head matters most
        cw, ch = W // 4, H // 4
        c = cov[::4, ::4][:ch, :cw]
        for ox in range(-a.shape[1] // 3, cw - a.shape[1] * 2 // 3 + 1, 2):
            for oy in range(0, max(1, ch // 6) + 1, 2):
                xs, ys = max(0, ox), max(0, oy)
                xe, ye = min(cw, ox + a.shape[1]), min(ch, oy + a.shape[0])
                if xe <= xs or ye <= ys: continue
                sub = a[ys - oy:ye - oy, xs - ox:xe - ox]; wsub = weight[ys - oy:ye - oy, xs - ox:xe - ox]
                inside = (sub * c[ys:ye, xs:xe] * wsub).sum()
                lost = (a * weight).sum() - inside            # art that falls outside tiles or the canvas
                fill = (sub * c[ys:ye, xs:xe]).sum() / max(1, c.sum())  # how much of the tile area we fill
                score = inside - 1.5 * lost + 400 * fill
                if best is None or score > best[0]:
                    best = (score, scale, ox * 4, oy * 4)
    _, scale, ox, oy = best
    h = int(H * scale); w = int(art.width * h / art.height)
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    canvas.alpha_composite(art.resize((w, h), Image.LANCZOS), (0, 0)) if False else None
    big = Image.new('RGBA', (W + 2 * w, H + 2 * h), (0, 0, 0, 0))
    big.alpha_composite(art.resize((w, h), Image.LANCZOS), (w + ox, h + oy))
    return big.crop((w, h, w + W, h + H))


def quantize_rows(desired, owner, tiles, base, default):
    """Give every palette row used by the portrait a fresh 63-colour palette built from its own pixels."""
    rows = {}
    for t in tiles:
        rows.setdefault((base + (t['attr'] & 0x1ff)) & 0x1ff, []).append(t['index'])
    rgba = np.asarray(desired)
    for row, idxs in rows.items():
        sel = np.isin(owner, idxs) & (rgba[:, :, 3] > 0)
        px = rgba[sel][:, :3]
        if len(px) == 0:
            continue
        sample = Image.fromarray(px.reshape(-1, 1, 3).astype(np.uint8), 'RGB')
        q = sample.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = np.array(q.getpalette()[:48 * 3], np.uint8).reshape(-1, 3)
        colors = [round1555(np.append(c, 255)) for c in pal]
        while len(colors) < 63: colors.append(colors[-1])
        default[row - 32] = [0] + [argb1555(c) for c in colors]
    return rows


def snap(desired, owner, tiles, palettes, base):
    """Snap every visible pixel to the exact palette of its front-most tile (as the encoder requires)."""
    out = np.zeros_like(np.asarray(desired))
    src = np.asarray(desired)
    for t in tiles:
        mask = (owner == t['index']) & (src[:, :, 3] > 0)
        if not mask.any(): continue
        pal = palettes.for_tile(t['attr'], base)
        opaque = np.flatnonzero(pal[:, 3] > 0)
        cols = pal[opaque][:, :3].astype(np.int32)
        p = src[mask][:, :3].astype(np.int32)
        d = ((p[:, None, :] - cols[None, :, :]) ** 2).sum(axis=2)
        out[mask] = pal[opaque[d.argmin(axis=1)]]
    return Image.fromarray(out, 'RGBA')


def relayout_tiles(bank, pattern, desired, origin):
    """Re-place every tile of the portrait pattern so the tiles cover our art, not the original fighter's.
    The original layout follows the Capcom art, so wherever our silhouette differs, art pixels had no tile
    and showed as cut edges and blank rectangles in game (cover_holes only moved fully empty tiles to the
    top-left of the gaps and left many). Greedy cover: largest tiles first, each at the position (up to one tile past the canvas)
    that covers the most still-uncovered art (plus the 1 px outline ring), on the 8 px grid the originals
    use; then every tile is re-placed against the others until nothing moves. Tiles with nothing left to
    cover go to an empty spot nearest their old place. Tile count, cells, sizes, palettes and draw order
    stay; only positions change (the canvas may grow or shrink). Returns uncovered art pixels left."""
    P = 32  # tiles may hang up to one tile past the canvas edge
    alpha = np.pad(np.asarray(desired)[:, :, 3] >= 128, P)
    need = alpha.copy()
    need[1:] |= alpha[:-1]; need[:-1] |= alpha[1:]; need[:, 1:] |= alpha[:, :-1]; need[:, :-1] |= alpha[:, 1:]
    H, W = need.shape
    ox0, oy0 = origin[0] - P, origin[1] - P  # sprite coordinates of padded pixel (0, 0)
    tiles = bank.tiles(pattern)
    count = np.zeros((H, W), np.int32)  # how many placed tiles cover each pixel
    place = {}

    def window_sums(mask, w, h):
        c = np.zeros((H + 1, W + 1), np.int32); c[1:, 1:] = mask.cumsum(0).cumsum(1)
        return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]   # [y, x] = sum of mask[y:y+h, x:x+w]

    def best_spot(t, w, h):
        # Capcom's portraits keep every tile on the 8 px sprite grid; off-grid tiles showed 1 px seams in game
        on = np.zeros((H - h + 1, W - w + 1), bool)
        on[(-oy0) % 8::8, (-ox0) % 8::8] = True
        # most uncovered art; among equals, least art that other tiles already cover (less overlap waste)
        score = window_sums(need & (count == 0), w, h) * 64 - window_sums(need & (count > 0), w, h)
        score = np.where(on, score, np.iinfo(np.int32).min)
        if score.max() > 0:
            return np.unravel_index(int(score.argmax()), score.shape)
        empty = np.argwhere(on & (window_sums(need, w, h) == 0))
        if not len(empty):
            empty = np.argwhere(on)  # nowhere fully empty: overlap is harmless, nothing left to cover
        oy, ox = t['y'] - oy0, t['x'] - ox0
        return empty[((empty - (oy, ox)) ** 2).sum(1).argmin()]

    order = sorted(tiles, key=lambda t: (-t['w'] * t['h'], t['index']))
    for sweep in range(4):  # greedy pass, then re-place each tile against the others until nothing improves
        moved = False
        for t in order:
            w, h = t['w'], t['h']
            if t['index'] in place:
                y, x = place[t['index']]; count[y:y + h, x:x + w] -= 1
            y, x = (int(v) for v in best_spot(t, w, h))
            moved |= place.get(t['index']) != (y, x)
            place[t['index']] = (y, x); count[y:y + h, x:x + w] += 1
        if not moved: break
    left = need & (count == 0)
    pos = {i: (x + ox0, y + oy0) for i, (y, x) in place.items()}
    # rewrite the delta-coded list in place: x -= dx, y -= dy per entry
    off = bank.pattern_offsets[pattern]
    data = bytearray(bank.data); px = py = 0
    for i, (dx, dy, attr, code) in enumerate(bank.entries(pattern)):
        nx, ny = pos[i]
        struct.pack_into('<hh', data, off + 2 + i * 8, px - nx, py - ny); px, py = nx, ny
    bank.data = bytes(data); bank.__init__(bank.data, bank.info)
    return int((left & alpha).sum())


def build_portraits(bank, default):
    pmap = {e['id']: e for e in json.loads((MOD / 'build' / 'ef02' / 'pattern_map.json').read_text())}
    jobs = []
    for mascot, c in CAST.items():
        jobs.append((mascot, c, pmap[c['id']]['portrait']['pattern'], ''))
        # Gill's VS art for the first player is a second pattern (eff38.c animation 23): replace it too
        if 'portrait_1p' in pmap[c['id']]:
            jobs.append((mascot, c, pmap[c['id']]['portrait_1p']['pattern'], '_1p'))
    for mascot, c, pattern, tag in jobs:
        pal = Palettes(default.astype('<u2').tobytes(), False)
        img, meta = bank.render(pattern, pal, metadata=True)
        size = img.size
        art = key_background(Image.open(GEN / f'portrait_{mascot}.png'))
        # light denoise: the bank has a fixed byte budget and LZ-p6 compresses flat areas far better
        art = art.filter(ImageFilter.MedianFilter(3))
        tiles = bank.tiles(pattern)
        owner = frontmost_tile_map(tiles, meta['origin'], size)
        layout = bank.data
        for shrink in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
            # the tiles have a fixed total area: if they can't cover the fitted art, fit it a bit smaller
            bank.data = layout; bank.__init__(bank.data, bank.info)
            desired = fit_art(art, size, owner >= 0, shrink)
            uncovered = relayout_tiles(bank, pattern, desired, meta['origin'])
            if not uncovered: break
        img, meta2 = bank.render(pattern, pal, metadata=True)
        if img.size != size or meta2['origin'] != meta['origin']:
            # keep the art where it was placed in sprite space if the canvas bounds moved
            moved_art = Image.new('RGBA', img.size, (0, 0, 0, 0))
            moved_art.paste(desired, (meta['origin'][0] - meta2['origin'][0],
                                      meta['origin'][1] - meta2['origin'][1]), desired)
            desired, size, meta = moved_art, img.size, meta2
        tiles = bank.tiles(pattern)
        owner = frontmost_tile_map(tiles, meta['origin'], size)
        a = np.asarray(desired).copy()
        a[:, :, 3] = np.where((a[:, :, 3] >= 128) & (owner >= 0), 255, 0)
        desired = add_outline(Image.fromarray(a, 'RGBA'), owner >= 0)
        quantize_rows(desired, owner, tiles, meta['palette_base'], default)
        pal = Palettes(default.astype('<u2').tobytes(), False)
        final = snap(desired, owner, tiles, pal, meta['palette_base'])
        final.save(OUT.parent / f'portrait_{mascot}{tag}_final.png')
        bank.data = bank.replace(pattern, final, pal)
        bank.__init__(bank.data, bank.info)
        print(f'portrait {mascot:9s} -> {c["body"]} pattern {pattern} {size}, fit x{shrink}, {uncovered} art px without a tile')


def run(tool, *args):
    subprocess.run([sys.executable, str(MOD / 'tools' / tool), *map(str, args)], check=True)


def collect_skin_sources(root):
    """Require every selected mascot's bank AND palette; never pack original fighters silently."""
    sources = []
    missing = []
    for c in CAST.values():
        # Character ids skip pl15 (the bank numbers after Akuma are shifted by one).
        bank = f'pl{c["id"] + (c["id"] >= 15):02d}'
        for name in (f'{bank}.bin', f'{bank}pl.bin'):
            path = Path(root) / bank / 'out' / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
            else:
                sources.append(path)
    if missing:
        raise ValueError('missing mascot skin outputs (build each skin first): ' + ', '.join(missing))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='logo,portraits')
    ap.add_argument('--no-repack', action='store_true')
    ap.add_argument('--out', type=Path, help='work dir for overrides/ and SF33RD.AFS (default sf3mod/build)')
    a = ap.parse_args()
    global OUT
    work = a.out.resolve() if a.out else MOD / 'build'
    OUT = work / 'overrides'
    parts = set(a.only.split(','))
    unknown = parts - {'logo', 'portraits', 'names', 'palettes', 'hud', 'skins', 'opening'}
    if unknown:
        ap.error('unknown build parts: ' + ', '.join(sorted(unknown)))
    try:
        skin_sources = collect_skin_sources(MOD / 'build' / 'skin') if 'skins' in parts else []
    except ValueError as error:
        ap.error(str(error))
    OUT.mkdir(parents=True, exist_ok=True)
    if 'logo' in parts: build_logo()
    if parts & {'portraits', 'names'}:
        bank = SpriteBank.open(EXT / 'ef02_usa.bin')
        default = np.frombuffer((EXT / 'default.bin').read_bytes(), '<u2').reshape(480, 64).copy()
        if 'portraits' in parts: build_portraits(bank, default)
        if 'names' in parts:
            sys.path.insert(0, str(MOD)); import names as nm
            nm.build_names(bank, default, NAMES, CAST)
        (OUT / 'ef02_usa.bin').write_bytes(bank.data)
        (OUT / 'default.bin').write_bytes(default.astype('<u2').tobytes())
        print('ef02_usa.bin', len(bank.data), 'bytes; default.bin written')
    if 'palettes' in parts:
        sys.path.insert(0, str(MOD)); import recolor
        recolor.build(EXT, OUT)
    if 'hud' in parts:
        sys.path.insert(0, str(MOD)); import hud
        hud.build(EXT, OUT)
    if 'skins' in parts:
        # full-body mascot skins: prebuilt per fighter by `python sf3mod/skin.py plNN --optimal`
        # (sprite bank + its palette; overrides the recolour of the same plNNpl.bin)
        for f in skin_sources:
            (OUT / f.name).write_bytes(prepare_character_bank(f.name, f.read_bytes()))
            print('skin', f.name)
    if 'opening' in parts:
        # opening / attract rebrand; runs after hud (both patch scrscrn.ppg and ef02_usa.bin)
        sys.path.insert(0, str(MOD)); import opening
        opening.build(EXT, OUT)
    if not a.no_repack:
        run('afs.py', 'repack', ROOT / 'game' / 'SF33RD.AFS', OUT, work / 'SF33RD.AFS')
        print('patched archive ->', work / 'SF33RD.AFS')


if __name__ == '__main__':
    main()
