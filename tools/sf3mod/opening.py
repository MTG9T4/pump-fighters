"""PUMP FIGHTERS opening / attract rebrand. Originals in build/extracted are read-only.

Run through build_mod.py --only ...,hud,opening (opening must come after hud: both patch scrscrn.ppg and
ef02_usa.bin, and this part builds on whatever earlier parts already wrote to the overrides dir).

What it replaces (decomp references are relative to mod/3s-psp/src):
* Opening.ppg (AFS 76), drawn by Game/OPENING.c as 256x256 8bpp blocks g_no 602+i. Every pTEX carries a list
  of 16px rectangles, each with its own palette row (ppgWriteQuadUseTrans: ColorRAM 300 + row). Those rows are
  title.bin (AFS 77), loaded to ColorRAM 300 by color_file[] (data 0x12C, apfn 0x4D).
  - sketch close-ups (tex 0-20) -> the mascot in the same composition, re-indexed onto the original ramp
  - Chun-Li (tex 29-32, multi-row colour art) -> CLAUDIA, new rows 75-81 in title.bin, new rectangle lists
  - the "STRIKE" letters (tex 35-46, S/T/R/I/K/E) -> FI/GH/T/E/R/S, new rectangle lists. op_bg0_0003 shows
    exactly six full-screen letter cuts (blocks 0x24..0x2F, one pair each), so the eight letters of FIGHTERS
    go in as FI, GH, T, E, R, S (FIGH-TERS)
* bg20.bin (AFS 74), sprite bank of the opening objects (texgrpdat: base CG 35904 = 0x8C40, to_tex 54492;
  effects F6/48/36 in Game/efff6.c, EFF48.c, eff36.c with _op_char_table):
  - the 19 character sketch panels (patterns 0-5, 8-11, 24-28, 32-35) -> the matching mascots
  - "3rd" zoom frames (16-20) -> "PUMP", the outlined "STRIKE" (58) -> "FIGHTERS"
  - the handshake (36) -> SWORDCAT's gauntlet and INU's wrapped paw (new title.bin rows 21/22)
  - OPBG_Init melts every referenced bg20 cell into display set 9's fixed pages (1024 small / 2176 large
    slots), so the redrawn patterns are re-encoded at cell level: same tile lists, identical cells shared,
    look-alike cells merged (faces/eyes protected) until 128 small / 64 large slots stay free
    (PUMP_BG20_CELLS can also cap the cell count)
  - the outlined letter font (anim 2: A-Z = 132-157) is kept: EFF48 spells the name cards from it, and the
    web build respells those tables (preview/bg20_name_letters.png)
* scrscrn.ppg tex 0, u128-255 v192-239, pPAL row 29: the small "3rd STRIKE" logo SF3_logo() (Game/sc_sub.c)
  draws over the attract fights -> the PUMP FIGHTERS logo (assets/gen/logo.png) in the same 16 colours.
* ef02_usa.bin: stage-intro / history names (_sel_pl_char_table anims 34 and 81, patterns 760-780, 1031-1051)
  and the stage-intro oval portraits (anim 12, patterns 811-831) -> mascot names / heads, in the existing
  shared palette rows (default.bin is not changed).
"""
import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MOD = Path(__file__).resolve().parent
ROOT = MOD.parent
ART = MOD / 'assets' / 'opening'
GEN = MOD / 'assets' / 'gen'
PREVIEW = MOD / 'build' / 'opening' / 'preview'
sys.path.insert(0, str(MOD / 'tools'))
from ppg import PPG  # noqa: E402
from common import decode_color, encode_color, palette_indices  # noqa: E402
from spritebank import SpriteBank, Palettes  # noqa: E402

BG20_INFO = dict(afs=74, base=35904, to_tex=54492, to_chd=0)
# bg20 pattern -> (mascot, title.bin row used for the reference render (first efff6/eff36 colour))
BG20_PANELS = {
    0: ('giga', 10), 1: ('claudia', 12), 2: ('chillguy', 38), 3: ('bonk', 17), 4: ('jeanphil', 39),
    5: ('pippin', 13), 8: ('swordcat', 15), 9: ('goat', 20), 10: ('popcat', 11), 11: ('mew', 27),
    24: ('moodeng', 25), 25: ('pepe', 27), 26: ('pnut', 39), 27: ('fwog', 35), 28: ('elon', 27),
    32: ('wif', 15), 33: ('ponke', 39), 34: ('inu', 14), 35: ('slerf', 25),
}
# Opening.ppg sketch groups: textures shown side by side (each has one rectangle and one row)
PPG_PANELS = {
    'oA': ([0, 1], 'swordcat'), 'oB': ([2, 3], 'giga'), 'oC': ([4, 5], 'bonk'),
    'oD': ([6, 7, 8], 'swordcat'), 'oE': ([9, 10, 11], 'swordcat'), 'oF': ([12, 13, 14], 'inu'),
    'oG': ([15, 16], 'goat'), 'oH': ([17, 18], 'inu'), 'oI': ([19, 20], 'inu'),
}
CLAUDIA_TEX = (29, 30, 31, 32)
CLAUDIA_ROWS = (75, 76, 77, 78, 79, 80, 81)   # title.bin rows only Chun-Li's textures use (80 was spare)
LETTERS = ('FI', 'GH', 'T', 'E', 'R', 'S')     # replaces S T R I K E in tex 35/36 .. 45/46 (6 cuts only)
TITLE_WORD = 'FIGHTERS'                         # replaces the outlined STRIKE (bg20 58)
LETTER_ROW = 43
NAMES = {'swordcat': 'SWORDCAT', 'inu': 'INU', 'jeanphil': 'JEAN PHIL', 'elon': 'ELON', 'pepe': 'PEPE',
         'claudia': 'CLAUDIA', 'wif': 'WIF', 'popcat': 'POPCAT', 'pnut': 'PNUT', 'moodeng': 'MOO DENG',
         'giga': 'GIGA', 'goat': 'GOAT', 'chillguy': 'CHILL GUY', 'ponke': 'PONKE', 'mew': 'MEW',
         'pippin': 'PIPPIN', 'slerf': 'SLERF', 'bonk': 'BONK', 'fwog': 'FWOG'}
# SF3 character id -> mascot (Gill, id 0, stays)
BY_ID = {1: 'giga', 2: 'swordcat', 3: 'wif', 4: 'jeanphil', 5: 'pepe', 6: 'moodeng', 7: 'mew', 8: 'pippin',
         9: 'slerf', 10: 'ponke', 11: 'inu', 12: 'chillguy', 13: 'elon', 14: 'goat', 15: 'claudia',
         16: 'popcat', 17: 'pnut', 18: 'fwog', 19: 'bonk'}
# _sel_pl_char_table anim 34 / 81 (names) and anim 12 (ovals), index = character id; Akuma's Gouki alt too
STAGE_NAMES = {**{760 + i: i for i in range(1, 15)}, **{776 + i - 15: i for i in range(15, 20)}, 775: 14}
HISTORY_NAMES = {**{1031 + i: i for i in range(1, 15)}, **{1047 + i - 15: i for i in range(15, 20)}, 1046: 14}
OVALS = {**{811 + i: i for i in range(1, 15)}, 826: 14, **{812 + i: i for i in range(15, 20)}}
FONTS = ['C:/Windows/Fonts/ariblk.ttf', 'C:/Windows/Fonts/impact.ttf', 'C:/Windows/Fonts/arialbd.ttf']


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        for f in FONTS:
            try: return ImageFont.truetype(f, size)
            except OSError: pass
    return ImageFont.load_default()


def lum(rgba):
    return np.asarray(rgba)[..., :3].astype(np.float32) @ np.array([.299, .587, .114], np.float32)


# ------------------------------------------------------------------ generated art -> panel canvas
def padded_box(w, h):
    """The 3:2 padding prep used for the Codex references (inverse is used to crop the result)."""
    W = max(w, int(h * 1.5)); H = max(h, int(W / 1.5)); W = int(H * 1.5)
    return W, H, (W - w) // 2, (H - h) // 2


def load_generated(path, size):
    g = Image.open(path).convert('L')
    w, h = size; W, H, ox, oy = padded_box(w, h)
    sx, sy = g.width / W, g.height / H
    g = g.crop((round(ox * sx), round(oy * sy), round((ox + w) * sx), round((oy + h) * sy)))
    return np.asarray(g.resize(size, Image.LANCZOS), np.float32)


def tone_map(gray, orig_idx, row, mask, levels=None, hist=0.6):
    """New greyscale art -> palette indices of the original ramp. The art was drawn to look like the
    in-game panel (negative panels were inverted for the reference, so undo that here), then its tones are
    matched to the original panel's tone distribution and snapped to the indices the original uses."""
    L = lum(row)
    used, counts = np.unique(orig_idx[mask], return_counts=True)
    used, counts = used[row[used, 3] > 0], counts[row[used, 3] > 0]
    neg = np.corrcoef(used, L[used])[0, 1] < 0 if len(used) > 2 else False
    src = (255 - gray) if neg else gray
    s = src[mask].astype(np.float64)
    target = np.sort(L[orig_idx[mask]].astype(np.float64))
    order = np.argsort(s, kind='stable')
    matched = np.empty_like(s); matched[order] = target
    lo, hi = np.percentile(target, 1), np.percentile(target, 99)
    linear = lo + (s - s.min()) / max(1e-6, s.max() - s.min()) * (hi - lo)
    want = hist * matched + (1 - hist) * linear
    # one representative index per distinct luminance (the most common one), optionally thinned out
    reps = {}
    for u, c in zip(used, counts):
        k = round(float(L[u]), 1)
        if k not in reps or c > reps[k][1]: reps[k] = (u, c)
    cand = np.array(sorted((v[0] for v in reps.values()), key=lambda u: L[u]))
    if levels and len(cand) > levels:
        cand = cand[np.unique(np.linspace(0, len(cand) - 1, levels).round().astype(int))]
    out = orig_idx.copy()
    out[mask] = cand[np.abs(L[cand][None, :] - want[:, None]).argmin(axis=1)]
    return out


# ------------------------------------------------------------------ Opening.ppg helpers
def tex_rects(doc, i):
    c = doc.textures[i]
    tr = doc.data[c['offset'] + 16:c['payload']]
    out = []
    for k in range(0, len(tr), 3):
        p, ip, cc = tr[k], tr[k + 1], tr[k + 2]
        out.append((p, ip % 16, ip // 16, (cc >> 4) + 1, (cc & 15) + 1))
    return out


def tex_idx(doc, i):
    return np.frombuffer(doc.raw(i), np.uint8).reshape(256, 256).copy()


def set_tex(doc, i, idx, rects=None):
    """Replace one 8bpp pTEX payload (zlib, as the original) and optionally its rectangle list."""
    c = doc.textures[i]
    assert c['pixel'] & 3 == 1 and c['compression'] == 2 and idx.shape == (c['height'], c['width'])
    hdr = bytearray(doc.data[c['offset']:c['offset'] + 16])
    if rects is None:
        trans = doc.data[c['offset'] + 16:c['payload']]
    else:
        trans = b''.join(bytes((p, y * 16 + x, (w - 1) << 4 | (h - 1))) for p, x, y, w, h in rects)
    struct.pack_into('>H', hdr, 14, len(trans) // 3)
    chunk = bytearray(hdr + trans + zlib.compress(idx.astype(np.uint8).tobytes(), 9))
    struct.pack_into('>I', chunk, 4, len(chunk))
    chunk.extend(b'\0' * (-len(chunk) % 4))
    end = c['offset'] + ((c['size'] + 3) & ~3)
    return PPG(doc.data[:c['offset']] + bytes(chunk) + doc.data[end:])


def rects_from(blockrow, x_range):
    """blockrow: 16x16 array of palette row per block (-1 = not drawn) -> merged rectangles."""
    runs = []
    for y in range(16):
        x = x_range[0]
        while x < x_range[1]:
            p = blockrow[y, x]
            if p < 0: x += 1; continue
            x1 = x
            while x1 < x_range[1] and blockrow[y, x1] == p: x1 += 1
            runs.append([p, x, y, x1 - x, 1]); x = x1
    merged = []
    for r in runs:
        for m in merged:
            if m[0] == r[0] and m[1] == r[1] and m[3] == r[3] and m[2] + m[4] == r[2] and m[4] < 16:
                m[4] += 1; break
        else:
            merged.append(r)
    return [tuple(m) for m in merged]


def render_tex(doc, i, title, rects=None):
    idx = tex_idx(doc, i); out = np.zeros((256, 256, 4), np.uint8)
    for p, x, y, w, h in rects or tex_rects(doc, i):
        reg = np.s_[y * 16:(y + h) * 16, x * 16:(x + w) * 16]
        out[reg] = title[p][idx[reg] & 63]
    return out


def group_layout(doc, texs):
    """Each group texture has one rectangle; the panel is those rectangles side by side."""
    lay = []; x = 0
    for t in texs:
        (p, bx, by, bw, bh), = tex_rects(doc, t)
        lay.append((t, p, bx * 16, by * 16, bw * 16, bh * 16, x)); x += bw * 16
    return lay, x, lay[0][5]


# ------------------------------------------------------------------ Opening.ppg parts
def build_ppg_panels(doc, title, report):
    for name, (texs, mascot) in PPG_PANELS.items():
        lay, W, H = group_layout(doc, texs)
        row = lay[0][1]; assert all(l[1] == row for l in lay)
        orig = np.concatenate([tex_idx(doc, t)[y:y + h, x:x + w] for t, p, x, y, w, h, _ in lay], axis=1)
        gray = load_generated(ART / f'{name}_{mascot}.png', (W, H))
        budget = sum(doc.textures[t]['size'] for t in texs)     # keep the file no bigger than the original
        for levels in (None, 24, 16, 12, 10, 8, 6):
            g = gray if levels is None else np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), np.float32)
            new = tone_map(g, orig, title[row], np.ones(orig.shape, bool), levels)
            trial = doc
            for t, p, x, y, w, h, cx in lay:
                idx = tex_idx(trial, t); idx[y:y + h, x:x + w] = new[:, cx:cx + w]
                trial = set_tex(trial, t, idx)
            size = sum(trial.textures[t]['size'] for t in texs)
            if size <= budget: break
        doc = trial
        before = Image.fromarray(title[row][orig]); after = Image.fromarray(title[row][new])
        side_by_side(before, after, PREVIEW / f'{name}_{mascot}.png')
        report.append(dict(part='Opening.ppg', textures=texs, panel=name, mascot=mascot, row=int(row), size=[W, H],
                           levels=levels or 'all', bytes=size, original_bytes=budget))
        print(f'opening {name} tex {texs} -> {mascot} (levels {levels or "all"}, {size:,}/{budget:,} bytes)')
    return doc


def build_claudia(doc, title, report, slack=0):
    """Chun-Li's colour panel (tex 29|30 over 31|32, 384x512) -> CLAUDIA with new rows and rectangles."""
    W, H = 384, 512
    halves = [(29, 0, 0, 64), (30, 192, 0, 0), (31, 0, 256, 64), (32, 192, 256, 0)]  # tex, canvas x, y, tex x
    orig = np.zeros((H, W, 4), np.uint8)
    for t, cx, cy, tx in halves:
        orig[cy:cy + 256, cx:cx + 192] = render_tex(doc, t, title)[:, tx:tx + 192]
    opaque = orig[:, :, 3] > 0
    cols, cnt = np.unique(orig[opaque][:, :3], axis=0, return_counts=True)
    bg = cols[cnt.argmax()]                                     # the flat beige the panel is painted on
    g = Image.open(ART / 'oJ_claudia.png').convert('RGB')       # 1024x1536; reference was 768x1024 at (128,256)
    sx, sy = g.width / 1024, g.height / 1536
    art = np.asarray(g.crop((round(128 * sx), round(256 * sy), round(896 * sx), round(1280 * sy)))
                     .resize((W, H), Image.LANCZOS)).astype(np.int32)
    near_bg = np.abs(art - art[2:12, 2:12].reshape(-1, 3).mean(axis=0)).max(axis=2) < 18
    art[near_bg] = bg
    # blocks that are only background are not drawn (the panel shows the same beige behind them)
    blocks = np.zeros((H // 16, W // 16), bool)
    for by in range(H // 16):
        for bx in range(W // 16):
            blocks[by, bx] = not near_bg[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16].all()
    # cluster drawn blocks by colour into the 7 spare rows, then give each row its own 63 colours
    feats = np.array([art[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16].reshape(-1, 3).mean(axis=0)
                      for by, bx in zip(*np.nonzero(blocks))], np.float32)
    k = len(CLAUDIA_ROWS)
    rng = np.random.default_rng(7)
    cent = feats[rng.choice(len(feats), k, replace=False)]
    for _ in range(30):
        lab = ((feats[:, None] - cent[None]) ** 2).sum(axis=2).argmin(axis=1)
        cent = np.array([feats[lab == j].mean(axis=0) if (lab == j).any() else cent[j] for j in range(k)])
    rowmap = np.full(blocks.shape, -1, np.int32)
    for (by, bx), l in zip(zip(*np.nonzero(blocks)), lab): rowmap[by, bx] = CLAUDIA_ROWS[l]
    # may use what the sketch panels saved, so Opening.ppg as a whole stays within its original size
    budget = sum(doc.textures[t]['size'] for t in CLAUDIA_TEX) + slack
    base_doc, base_title = doc, title.copy()
    for ncol in (63, 48, 40, 32, 24, 16):
        doc, title[:] = base_doc, base_title
        doc = claudia_pass(doc, title, art, rowmap, halves, ncol)
        size = sum(doc.textures[t]['size'] for t in CLAUDIA_TEX)
        if size <= budget: break
    new_rgba = np.zeros((H, W, 4), np.uint8)
    for t, cx, cy, tx in halves:
        new_rgba[cy:cy + 256, cx:cx + 192] = render_tex(doc, t, title)[:, tx:tx + 192]
    show = lambda a: Image.alpha_composite(Image.new('RGBA', (W, H), tuple(int(v) for v in bg) + (255,)), Image.fromarray(a))
    side_by_side(show(orig), show(new_rgba), PREVIEW / 'oJ_claudia.png')
    report.append(dict(part='Opening.ppg', textures=list(CLAUDIA_TEX), panel='oJ', mascot='claudia',
                       title_rows=list(CLAUDIA_ROWS), drawn_blocks=int(blocks.sum()), colours_per_row=ncol,
                       bytes=size, original_bytes=budget))
    print(f'opening oJ tex 29-32 -> claudia (title.bin rows 75-81, {ncol} colours/row, {size:,}/{budget:,} bytes)')
    return doc


def claudia_pass(doc, title, art, rowmap, halves, ncol):
    H, W = art.shape[:2]
    idx = np.zeros((H, W), np.uint8)
    for r in CLAUDIA_ROWS:
        sel = np.kron(rowmap == r, np.ones((16, 16), bool))
        if not sel.any():
            title[r] = 0; continue
        px = art[sel].astype(np.uint8)
        q = Image.fromarray(px.reshape(-1, 1, 3)).quantize(colors=ncol, method=Image.Quantize.MEDIANCUT,
                                                            dither=Image.Dither.NONE)
        pal = np.zeros((64, 4), np.uint8)
        n = len(q.getpalette()) // 3
        pal[1:1 + min(63, n), :3] = np.array(q.getpalette()[:min(63, n) * 3], np.uint8).reshape(-1, 3)
        pal[1:, 3] = 255
        pal = decode_color(encode_color(pal))                     # ARGB1555 round trip
        pal[0] = 0
        title[r] = pal
        idx[sel] = palette_indices(np.column_stack([px, np.full(len(px), 255, np.uint8)]), pal[1:]) + 1
    for t, cx, cy, tx in halves:
        tex = np.zeros((256, 256), np.uint8)
        tex[:, tx:tx + 192] = idx[cy:cy + 256, cx:cx + 192]
        sub = rowmap[cy // 16:cy // 16 + 16, cx // 16:cx // 16 + 12]
        brow = np.full((16, 16), -1, np.int32); brow[:, tx // 16:tx // 16 + 12] = sub
        rects = rects_from(brow, (tx // 16, tx // 16 + 12))
        doc = set_tex(doc, t, tex, rects)
    return doc


def rough_letter(ch, box, size=(416, 256), seed=0):
    """Heavy letter filling box (x0,y0,x1,y1) with the original's streaky dry-brush edges."""
    x0, y0, x1, y1 = box
    f = font('C:/Windows/Fonts/ariblk.ttf', 400)
    im = Image.new('L', (int(f.getlength(ch)) + 200, 600)); ImageDraw.Draw(im).text((50, 20), ch, font=f, fill=255)
    im = im.crop(im.getbbox())
    h = y1 - y0; w = int(im.width * h / im.height * 1.05)
    if w > size[0] - 24:                         # two letters: keep their shape, shrink to the slot width
        s_ = (size[0] - 24) / w; w = size[0] - 24; h2 = int(h * s_); y0 += (h - h2) // 2; h = h2
    w = max(w, int((x1 - x0) * 0.55))
    im = im.resize((w, h), Image.LANCZOS)
    canvas = Image.new('L', size); canvas.paste(im, (min(max(12, (x0 + x1 - w) // 2), size[0] - w - 12), y0))
    m = np.asarray(canvas, np.float32) / 255
    k = 10; c = np.cumsum(np.pad(m, ((0, 0), (k, k))), axis=1)
    blur = (c[:, 2 * k:] - c[:, :-2 * k])[:, :size[0]] / (2 * k)
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(size[::-1]).astype(np.float32)
    nc = np.cumsum(np.pad(n, ((0, 0), (20, 20))), axis=1); n = (nc[:, 40:] - nc[:, :-40])[:, :size[0]] / 6
    edge = (blur > 0.02) & (blur < 0.98)
    mask = np.where(edge, blur + 0.15 * n > 0.5, m > 0.5)
    return mask


def build_letters(doc, title, report):
    for k, ch in enumerate(LETTERS):
        L, R = 35 + 2 * k, 36 + 2 * k
        orig = np.concatenate([tex_idx(doc, L)[:, 48:256], tex_idx(doc, R)[:, 0:208]], axis=1)
        # the letter is the transparent hole in a black (index 48) box
        ys, xs = np.nonzero(orig == 48)
        hull = np.zeros(orig.shape, bool); hull[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        hy, hx = np.nonzero(hull & (orig == 0))
        mask = rough_letter(ch, (hx.min(), hy.min(), hx.max() + 1, hy.max() + 1), seed=k)
        # the hole must sit inside the black box (outside it the scene behind shows): widen the box if needed
        my, mx = np.nonzero(mask)
        x0 = max(0, min(xs.min(), mx.min() - 12)); x1 = min(orig.shape[1], max(xs.max() + 1, mx.max() + 13))
        hull[ys.min():ys.max() + 1, x0:x1] = True
        new = np.where(hull & ~mask, 48, 0).astype(np.uint8)
        for t, cx, tx in ((L, 0, 48), (R, 208, 0)):
            idx = np.zeros((256, 256), np.uint8); idx[:, tx:tx + 208] = new[:, cx:cx + 208]
            brow = np.full((16, 16), -1, np.int32)
            for by in range(16):
                for bx in range(tx // 16, tx // 16 + 13):
                    if idx[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16].any(): brow[by, bx] = LETTER_ROW
            doc = set_tex(doc, t, idx, rects_from(brow, (tx // 16, tx // 16 + 13)))
        show = lambda a: Image.fromarray(np.where(a[..., None] == 48, [0, 0, 0, 255], [235, 235, 235, 255]).astype(np.uint8))
        side_by_side(show(orig), show(new), PREVIEW / f'letter_{k}_{ch}.png')
        report.append(dict(part='Opening.ppg', textures=[L, R], letter_from='STRIKE'[k], letter_to=ch))
    print('opening letters STRIKE -> ' + ' '.join(LETTERS) + ' (tex 35-46)')
    return doc


# ------------------------------------------------------------------ bg20.bin parts
def synthetic(pal, row):
    """A copy of the palettes whose ColorRAM row has one unique colour per index (exact index round trip)."""
    p = Palettes.__new__(Palettes); p.__dict__.update(pal.__dict__); p.rgba = pal.rgba.copy()
    a = p.rgba[row - 32]; a[:, 0] = np.arange(64) * 4; a[:, 1] = 7; a[:, 2] = np.arange(64)
    a[a[:, 3] == 0, :3] = 0
    return p


def bank_idx(bank, pattern, pal, base):
    im, meta = bank.render(pattern, pal, base, True)
    a = np.asarray(im)
    return (a[:, :, 0] // 4).astype(np.uint8), a[:, :, 3] > 0, meta


def bank_put(bank, pattern, idx, alpha, pal, row, base):
    syn = synthetic(pal, row).rgba[row - 32]
    img = syn[idx].copy(); img[~alpha] = 0
    data = bank.replace(pattern, Image.fromarray(img), synthetic(pal, row), base)
    return SpriteBank(data, bank.info)


def zoom_blur(mask, amount, steps=12):
    if amount <= 0: return mask
    im = Image.fromarray((mask * 255).astype(np.uint8)); w, h = im.size; acc = np.zeros(mask.shape, np.float32)
    for s in np.linspace(1, 1 + amount, steps):
        z = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
        acc += np.asarray(z.crop(((z.width - w) // 2, (z.height - h) // 2, (z.width - w) // 2 + w,
                                  (z.height - h) // 2 + h)), np.float32) / 255
    return acc / steps


def text_mask(text, box, size, fontpath, stretch=1.0):
    x0, y0, x1, y1 = box
    f = font(fontpath, 300)
    im = Image.new('L', (int(f.getlength(text)) + 200, 500)); ImageDraw.Draw(im).text((20, 20), text, font=f, fill=255)
    im = im.crop(im.getbbox())
    h = y1 - y0; w = min(int(im.width * h / im.height * stretch), size[0] - 4)
    im = im.resize((w, h), Image.LANCZOS)
    c = Image.new('L', size); c.paste(im, (min(max(2, (x0 + x1 - w) // 2), size[0] - w - 2), y0))
    return np.asarray(c, np.float32) / 255


# The opening objects are drawn through display set 9 (texcash mts_base[9]: 4 pages of 16x16 slots = 1024
# small cells, 34 pages of 32x32 slots = 2176 large cells). OPBG_Init calls mlt_obj_melt2(&mts[9], 0x8C40),
# which decodes every referenced (cell, attr & 3) of bg20 once into those pages, in table order; anything past
# the last slot has no texture and the game dies in flPS2SetTextureRegister. The original bank uses 326 small
# and 2169 large slots. So redrawn patterns may not add cells: tiles are re-encoded by content (identical cells
# are shared, unused cells dropped, codes renumbered), and near-flat tiles are flattened until the bank fits.
MELT_SMALL, MELT_LARGE = 4 * 256, 34 * 64
MELT_SPARE_SMALL, MELT_SPARE_LARGE = 128, 64      # never fill the pages: keep these slots free
# face/eye boxes per redrawn sketch (fractions of the 416x256 panel): merges there cost FACE_GUARD times more
FACE_GUARD = 60.0
FACE_BOX = {0: (.40, 0, .68, .58), 1: (.33, .03, .67, .62), 2: (.18, .03, .78, .78), 3: (.28, 0, .58, .32),
            4: (.43, 0, .82, .62), 5: (.28, .12, .67, .78), 8: (.28, .03, .62, .52), 9: (.33, .03, .77, .77),
            10: (.28, .03, .82, .67), 11: (.33, .03, .77, .62), 24: (.23, 0, .67, .57), 25: (.43, 0, .82, .52),
            26: (.28, .12, .67, .67), 27: (.13, .08, .87, .77), 28: (.28, 0, .72, .67), 32: (.43, 0, .77, .47),
            33: (.28, .08, .67, .62), 34: (.33, 0, .72, .62), 35: (.38, 0, .62, .37)}
BG_SNAP = 0.13          # tones this close (fraction of the panel's tone range) to the background become background


class CellBank:
    """Re-encode whole patterns at cell level, sharing identical cells, without changing any tile list."""

    def __init__(self, bank):
        self.bank = bank
        self.count = bank.count
        self.new = {}                      # pattern -> per tile (key, flat key, cost, attr & 3)
        self.assign = {}
        self.tone = {}
        self.weight = {}                   # pattern -> merge cost factor (hard-edged text/colour art: keep exact)
        self.guard = {}                    # (pattern, tile) -> inside the face/eye box
        self.enc = {}                      # content key -> encoded blob (cache across attempts)

    def cell_key(self, code):
        wh, px, used, w, h = self.bank.cell(code)
        return (wh, px.tobytes())

    def set_pattern(self, p, idx_for_attr, tone=None, guard=None):
        """Per tile: its exact cell content, and a flat alternative (whole cell = the tile's most common
        index) with the cost of using it (how far the tile's pixels are from that value, in tone units)."""
        b = self.bank
        tiles = b.tiles(p)
        ox, oy = min(t['x'] for t in tiles), min(t['y'] for t in tiles)
        keys = []
        W = max(t['x'] + t['w'] for t in tiles) - ox; H = max(t['y'] + t['h'] for t in tiles) - oy
        for i, t in enumerate(tiles):
            if guard:
                gx0, gy0, gx1, gy1 = guard[0] * W, guard[1] * H, guard[2] * W, guard[3] * H
                x, y = t['x'] - ox, t['y'] - oy
                if x < gx1 and x + t['w'] > gx0 and y < gy1 and y + t['h'] > gy0: self.guard[(p, i)] = True
        self.tone[p] = (tone if tone is not None else np.arange(256, dtype=np.float32) * 4).astype(np.float32)
        for t in tiles:
            wh, cell, used, w, h = b.cell(t['code'])
            cell = cell.copy()
            src = idx_for_attr(t['attr'] & 0x1ff)
            reg = src[t['y'] - oy:t['y'] - oy + t['h'], t['x'] - ox:t['x'] - ox + t['w']].copy()
            vals, cnt = np.unique(reg, return_counts=True)
            mode = vals[cnt.argmax()]
            lum_ = tone if tone is not None else np.arange(256, dtype=np.float32) * 4
            cost = float(np.abs(lum_[reg].astype(np.float32) - lum_[mode]).sum())
            if t['attr'] & 0x8000: reg = reg[:, ::-1]
            if t['attr'] & 0x4000: reg = reg[::-1]
            cell[:t['h'], :t['w']] = reg
            flat = np.full_like(cell, mode)
            keys.append(((wh, cell.tobytes()), (wh, flat.tobytes()), cost, t['attr'] & 3))
        self.new[p] = keys

    def choose(self, max_cells, max_small, max_large):
        """Fit the cell and melt-slot limits by merging look-alike cells: within one redrawn pattern, a cell is
        replaced by the most similar other cell of the same size (smallest tone difference over the area its
        tiles show), cheapest merges first, until the distinct cells and the (cell, attr & 3) slots fit."""
        import heapq
        from collections import Counter, defaultdict
        b = self.bank
        cells = Counter(); slots = Counter()
        big = lambda key: (key[0] & 3) == 3
        self.assign = {}
        for q in range(b.count):
            if q in self.new:
                for i, (key, flat, cost, pa) in enumerate(self.new[q]):
                    self.assign[(q, i)] = key; cells[key] += 1; slots[(key, pa)] += 1
            else:
                for dx, dy, attr, code in b.entries(q):
                    key = self.cell_key(code); cells[key] += 1
                    if not attr & 0x1000: slots[(key, attr & 3)] += 1
        def counts():
            ns = sum(1 for (k, a) in slots if not big(k)); return len(cells), ns, len(slots) - ns
        n, ns, nl = counts()
        if n <= max_cells and ns <= max_small and nl <= max_large:
            return dict(merged_cells=0, merge_cost=0)
        # candidates per pattern and cell size
        heap = []; cand = {}; owner = {}
        tiles_of = defaultdict(list)                       # (q, key) -> tile indices
        for q, keys in self.new.items():
            tb = self.bank.tiles(q)
            for i, (key, flat, cost, pa) in enumerate(keys): tiles_of[(q, key)].append(i)
            lut = self.tone[q]
            groups = defaultdict(list)
            for (qq, key), idxs in tiles_of.items():
                if qq == q: groups[key[0]].append(key)       # same cell header: same size AND same w x h
            for wh_, ks in groups.items():
                side = ((wh_ & 3) + 1) * 8
                T = np.stack([lut[np.frombuffer(k[1], np.uint8)] for k in ks]).astype(np.float32)
                M = np.zeros((len(ks), side, side), np.float32)
                for a, k in enumerate(ks):
                    for i in tiles_of[(q, k)]:
                        t = tb[i]; M[a, :t['h'], :t['w']] += 1     # weight = number of tiles showing it
                M = M.reshape(len(ks), -1)
                raw = np.stack([np.frombuffer(k[1], np.uint8) for k in ks])
                for a in range(len(ks)):
                    g = FACE_GUARD if any(self.guard.get((q, i)) for i in tiles_of[(q, ks[a])]) else 1.0
                    d = (np.abs(T - T[a]) * M[a]).sum(axis=1) * self.weight.get(q, 1.0) * g; d[a] = np.inf
                    order = np.argsort(d)[:24]
                    lst = [(float(d[o]), ks[o]) for o in order if np.isfinite(d[o])]
                    # a flat cell (any pattern's: identical content is shared) is usually the least visible merge
                    vis = raw[a][M[a] > 0]
                    if vis.size:
                        vals, cnt = np.unique(vis, return_counts=True)
                        for m in vals[np.argsort(-cnt)[:2]]:
                            fk = (ks[a][0], bytes([int(m)]) * (side * side))
                            if fk != ks[a] and fk in cells:
                                lst.append((float((np.abs(T[a] - lut[m]) * M[a]).sum() * self.weight.get(q, 1.0) * g * 0.8), fk))
                    lst.sort(key=lambda c: c[0])
                    cand[(q, ks[a])] = lst
                    owner[(q, ks[a])] = 0
                    if cand[(q, ks[a])]:
                        c0, kb = cand[(q, ks[a])][0]; heapq.heappush(heap, (c0, id(ks[a]), q, ks[a]))
        alive = defaultdict(lambda: True); target = {}
        merged = 0; worst = 0.0
        def final(q, k):
            while (q, k) in target: k = target[(q, k)]
            return k
        while heap and not (n <= max_cells and ns <= max_small and nl <= max_large):
            c, _, q, ka = heapq.heappop(heap)
            if not alive[(q, ka)]: continue
            lst = cand[(q, ka)]; ptr = owner[(q, ka)]
            if nl > max_large and not big(ka) and n <= max_cells and ns <= max_small: continue
            # skip targets that were merged away
            while ptr < len(lst) and not alive[(q, lst[ptr][1])]: ptr += 1
            owner[(q, ka)] = ptr
            if ptr >= len(lst): continue
            cost, kb = lst[ptr]
            if cost > c + 1e-6:                            # target changed: requeue with the real cost
                heapq.heappush(heap, (cost, id(ka), q, ka)); continue
            for i in tiles_of[(q, ka)]:
                pa = self.new[q][i][3]
                for cnt, k in ((cells, ka), (slots, (ka, pa))):
                    cnt[k] -= 1
                    if not cnt[k]: del cnt[k]
                cells[kb] += 1; slots[(kb, pa)] += 1
                self.assign[(q, i)] = kb
            tiles_of[(q, kb)] += tiles_of.pop((q, ka))
            alive[(q, ka)] = False; target[(q, ka)] = kb
            merged += 1; worst = max(worst, cost)
            n, ns, nl = counts()
        return dict(merged_cells=merged, merge_cost=round(worst, 1))

    def build(self):
        from codecs_sf3 import p6_encode
        from spritebank import morton
        b = self.bank
        order = {}; blobs = []
        def code_for(key, old=None):
            if key not in order:
                if old is not None:
                    pos = b.tex + b.offsets[old]; blob = b.data[pos:pos + b.cell(old)[2]]
                else:
                    if key not in self.enc:
                        wh, raw = key; side = ((wh & 3) + 1) * 8
                        cell = np.frombuffer(raw, np.uint8).reshape(side, side)
                        m = np.empty(cell.size, np.uint8); m[morton(side).reshape(-1)] = cell.reshape(-1)
                        self.enc[key] = bytes([wh]) + p6_encode(m.tobytes())
                    blob = self.enc[key]
                order[key] = len(blobs); blobs.append(blob)
            return order[key]
        result = bytearray(b.data)
        offsets = {}
        for p in range(b.count):
            off = b.pattern_offsets[p]
            if off in offsets and (p in self.new or offsets[off] in self.new):
                raise ValueError(f'pattern {p} shares its tile list with pattern {offsets[off]}')
            offsets.setdefault(off, p)
        melt = set(); small = large = 0
        for p in range(b.count):
            ents = b.entries(p)
            for i, (dx, dy, attr, code) in enumerate(ents):
                if p in self.new:
                    key = self.assign.get((p, i), self.new[p][i][0])
                else:
                    key = self.cell_key(code)
                new = code_for(key, None if p in self.new else code)
                struct.pack_into('<H', result, b.pattern_offsets[p] + 2 + i * 8 + 6, new)
                if not attr & 0x1000 and (new, attr & 3) not in melt:
                    melt.add((new, attr & 3))
                    if (key[0] & 3) == 3: large += 1
                    else: small += 1
        offs = []; payload = bytearray()
        for blob in blobs:
            offs.append(len(blobs) * 4 + len(payload)); payload.extend(blob)
        texture = struct.pack(f'<{len(offs)}I', *offs) + payload
        budget = b.end - b.tex
        stats = dict(cells=len(blobs), melt_small=small, melt_large=large, texture_bytes=len(texture), budget=budget)
        if len(texture) > budget:
            raise ValueError(f'texture bytes {len(texture)} > budget {budget}')
        result[b.tex:b.end] = texture + b'\0' * (budget - len(texture))
        return SpriteBank(bytes(result), b.info), stats


def build_bg20(ext, out, title, report):
    data = (ext / 'bg20.bin').read_bytes()      # always rebuild from the original bank
    info = dict(BG20_INFO, size=len(data))
    bank = SpriteBank(data, info)
    default = bytearray((ext / 'default.bin').read_bytes())
    t = (ext / 'title.bin').read_bytes(); default[(300 - 32) * 128:(300 - 32) * 128 + len(t)] = t
    pal = Palettes(bytes(default))
    orig_cells = bank.texture_count
    # Hard limit: the melt pages (MELT_SMALL / MELT_LARGE slots). The cell count itself is not used by the
    # game code (melt2 renumbers tiles to page slots), but by default it is kept at the original count too.
    # The real limit is melt2's pages; keep spare slots there. The cell count is not used by the game, so it
    # is only capped if PUMP_BG20_CELLS is set.
    cell_cap = int(os.environ.get('PUMP_BG20_CELLS', 65535))
    max_small, max_large = MELT_SMALL - MELT_SPARE_SMALL, MELT_LARGE - MELT_SPARE_LARGE
    # 1. the outlined name-card letters stay (the web build respells the EFF48 tables); draw a reference sheet
    glyph_sheet(bank, pal, title)
    fixed = {}                                    # pattern -> (index image, preview row)
    # 2. "3rd" zoom frames -> PUMP (row: eff36 entry 15 pushes title row 8)
    row = 300 + 8
    sharp_idx, _, _ = bank_idx(bank, 20, synthetic(pal, row), row)
    ink = (sharp_idx > 0) & (sharp_idx < 12)          # index 1 = the letters, 23 = the background
    ys, xs = np.nonzero(ink)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    pump = text_mask('PUMP', box, sharp_idx.shape[::-1], 'C:/Windows/Fonts/impact.ttf')
    for p, amount in ((16, .55), (17, .32), (18, .16), (19, .06), (20, 0)):
        idx, alpha, _ = bank_idx(bank, p, synthetic(pal, row), row)
        fixed[p] = (idx, tone_map(255 * (1 - zoom_blur(pump, amount)), idx, title[8], alpha), 8, f'bg20_{p:03}_PUMP.png')
    # 3. outlined STRIKE -> FIGHTERS (attr 1 + eff36 routine 1 -> title row 3)
    row = 300 + 3; base = row - 1
    idx, alpha, _ = bank_idx(bank, 58, synthetic(pal, row), base)
    H_, W_ = idx.shape
    box = (6, 4, W_ - 6, H_ - 4)                            # the plate, with a small margin all round
    m = text_mask(TITLE_WORD, box, idx.shape[::-1], 'C:/Windows/Fonts/ariblk.ttf')
    fixed[58] = (idx, tone_map(255.0 * m, idx, title[3], alpha, hist=0), 3, f'bg20_058_{TITLE_WORD}.png')
    # 4. the handshake (colour, title rows 21/22) and the 19 character sketches; everything is re-encoded at
    #    cell level and retried with simpler settings until bytes, cells and melt slots all fit.
    base_title = title.copy()
    sketch_src = {}
    for p, (mascot, c) in BG20_PANELS.items():
        idx, alpha, _ = bank_idx(bank, p, synthetic(pal, 300 + c), 300 + c)
        sketch_src[p] = (idx, alpha, load_generated(ART / f'b{p:02}_{mascot}.png', idx.shape[::-1]))
    cb = CellBank(bank)
    for levels, med, ncol in ((None, False, 63), (None, True, 63), (32, True, 56), (24, True, 48), (16, True, 32)):
        title[:] = base_title
        hs_idx, hs_prev = handshake_indices(bank, ext, title, ncol)
        cb.new = {}; cb.assign = {}; cb.tone = {}; cb.guard = {}
        cb.weight = {p: 1e6 for p in (16, 17, 18, 19, 20, 36, 58)}
        cb.weight.update({2: 3.0, 27: 2.0, 25: 1.5})      # clean line art / big flat shapes show merges most
        hs_tone = lum(title[21])
        cb.set_pattern(36, lambda a: hs_idx[a], np.concatenate([hs_tone, np.zeros(192, np.float32)]))
        for p, (idx, new, r, name) in fixed.items():
            cb.set_pattern(p, lambda a, new=new: new, np.concatenate([lum(title[r]), np.zeros(192, np.float32)]))
        done = []
        for p, (mascot, c) in BG20_PANELS.items():
            idx, alpha, gray = sketch_src[p]
            g = np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), np.float32) if med else gray
            new = tone_map(g, idx, title[c], alpha, levels)
            # the original paints its empty paper/background with one index (one shared cell); snap the new
            # art's near-background tones to it so those tiles share cells again
            L = lum(title[c]); vals, cnt = np.unique(idx[alpha], return_counts=True); bgi = vals[cnt.argmax()]
            rng_ = max(1.0, float(np.ptp(L[vals])))
            new = np.where(alpha & (np.abs(L[new] - L[bgi]) <= BG_SNAP * rng_), bgi, new).astype(np.uint8)
            cb.set_pattern(p, lambda a, new=new: new, np.concatenate([L, np.zeros(192, np.float32)]), FACE_BOX.get(p))
            done.append((p, mascot, c, idx))
        fl = cb.choose(cell_cap, max_small, max_large)
        try:
            nb, st = cb.build()
        except ValueError as e:
            print(f'bg20: levels {levels or "all"}, median {med}, {ncol} colours: {e}'); continue
        st.update(fl)
        ok = st['cells'] <= cell_cap and st['melt_small'] <= max_small and st['melt_large'] <= max_large
        print(f'bg20: levels {levels or "all"}, median {med}, {ncol} colours: {st}' + ('' if ok else ' -> does not fit'))
        if ok: break
    else:
        raise ValueError('bg20 panels do not fit the melt pages / bank budget')
    bank = nb
    # previews from the final bank, exactly as encoded
    hp = handshake_pal(ext, title)
    side_by_side(hs_prev, bank.render(36, hp, 300 + 21), PREVIEW / 'bg20_036_handshake.png')
    report.append(dict(part='bg20.bin', patterns='36', change='handshake -> SWORDCAT gauntlet + INU paw',
                       title_rows=[21, 22], colours_per_row=ncol))
    for p, (idx, new, r, name) in fixed.items():
        a, _, _ = bank_idx(bank, p, synthetic(pal, 300 + r), 300 + r if p != 58 else 302)
        side_by_side(Image.fromarray(title[r][idx]), Image.fromarray(title[r][a]), PREVIEW / name)
    report.append(dict(part='bg20.bin', patterns='16-20', change='"3rd" zoom frames -> "PUMP"'))
    report.append(dict(part='bg20.bin', patterns='58', change=f'"STRIKE" -> "{TITLE_WORD}"'))
    for p, mascot, c, idx in done:
        a, _, _ = bank_idx(bank, p, synthetic(pal, 300 + c), 300 + c)
        side_by_side(Image.fromarray(title[c][idx]), Image.fromarray(title[c][a]), PREVIEW / f'bg20_{p:03}_{mascot}.png')
        report.append(dict(part='bg20.bin', patterns=str(p), mascot=mascot, title_row=c, levels=levels or 'all', median=med))
    (out / 'bg20.bin').write_bytes(bank.data)
    print(f'bg20.bin cells {orig_cells} -> {st["cells"]}, melt slots small {st["melt_small"]}/{MELT_SMALL} '
          f'large {st["melt_large"]}/{MELT_LARGE}, texture bytes {st["texture_bytes"]:,}/{st["budget"]:,}')
    report.append(dict(part='bg20.bin', original_cells=orig_cells, settings=dict(levels=levels or 'all', median=med,
                       handshake_colours=ncol), **st, file_size=len(bank.data)))


GLYPH_ROW = 57          # eff48 anim 2: COLOR_COPYn(0x10, (0x12C + col + 0x38) * 4, 4) -> title row 56 + col (col 1 here)


def glyph_sheet(bank, pal, title):
    """Reference sheet of the name-card letters (anim 2 of _op_char_table): frame k -> pattern."""
    frames = list(range(132, 158)) + list(range(125, 132))
    row = 300 + GLYPH_ROW
    cells = []
    for k, p in enumerate(frames, 1):
        idx, alpha, _ = bank_idx(bank, p, synthetic(pal, row), row)
        im = title[GLYPH_ROW][idx].copy(); im[~alpha] = 0
        cells.append((k, p, Image.fromarray(im)))
    s = 3; sheet = Image.new('RGBA', (11 * 40 * s, 3 * 48 * s), (37, 40, 50, 255)); d = ImageDraw.Draw(sheet)
    for n, (k, p, im) in enumerate(cells):
        x, y = (n % 11) * 40 * s, (n // 11) * 48 * s
        sheet.alpha_composite(im.resize((im.width * s, im.height * s), Image.NEAREST), (x, y + 12 * s))
        d.text((x + 2, y + 2), f'{k}:{p}', fill=(255, 255, 0, 255))
    sheet.save(PREVIEW / 'bg20_name_letters.png')


def handshake_pal(ext, title):
    d = bytearray((ext / 'default.bin').read_bytes()); t = encode_color(title.reshape(-1, 4))
    d[(300 - 32) * 128:(300 - 32) * 128 + len(t)] = t
    return Palettes(bytes(d))


def handshake_indices(bank, ext, title, ncol):
    """bg20 pattern 36 (eff36 type 28: title rows 21/22 pushed to ColorRAM 8/9, tile attr 0/1) -> new colour art,
    with new palettes for title rows 21 and 22 (nothing else uses them). Returns per-attr index images."""
    pal = handshake_pal(ext, title); base = 300 + 21
    old, meta = bank.render(36, pal, base, True)
    W, H = old.size
    tiles, cover, owner = tile_maps(bank, 36, meta, (H, W))
    art = Image.open(ART / 'b36_handshake.png').convert('RGB')
    Wp, Hp, ox, oy = padded_box(W, H); sx, sy = art.width / Wp, art.height / Hp
    art = np.asarray(art.crop((round(ox * sx), round(oy * sy), round((ox + W) * sx), round((oy + H) * sy)))
                     .resize((W, H), Image.LANCZOS)).astype(np.uint8).copy()
    # flatten the paper texture to one beige (cleaner, compresses and shares cells far better)
    border = np.concatenate([art[:4].reshape(-1, 3), art[-4:].reshape(-1, 3), art[:, :4].reshape(-1, 3)])
    paper = np.median(border, axis=0)
    art[np.abs(art.astype(np.int32) - paper).max(axis=2) < 16] = paper.astype(np.uint8)
    attr = np.zeros((H, W), np.int32)
    for t in tiles: attr[owner == t['index']] = t['attr'] & 0x1ff
    out = {}
    for r in (0, 1):
        sel = (attr == r) & cover
        px = art[sel]
        q = Image.fromarray(px.reshape(-1, 1, 3)).quantize(colors=ncol, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        p = np.zeros((64, 4), np.uint8)
        p[1:1 + min(63, ncol), :3] = np.array(q.getpalette()[:min(63, ncol) * 3], np.uint8).reshape(-1, 3)
        p[1:, 3] = 255
        p = decode_color(encode_color(p)); p[0] = 0
        title[21 + r] = p
        flat = art.reshape(-1, 3)
        out[r] = (palette_indices(np.column_stack([flat, np.full(len(flat), 255, np.uint8)]), p[1:]) + 1).reshape(H, W).astype(np.uint8)
    return out, old


# ------------------------------------------------------------------ scrscrn demo logo
def build_demo_logo(ext, out, report):
    import hud
    src = out / 'scrscrn.ppg' if (out / 'scrscrn.ppg').exists() else ext / 'scrscrn.ppg'
    doc = PPG(src.read_bytes())
    pal16 = [row for c in doc.palettes if c['colors'] == 16 for row in c['rgba']][29]
    a = hud.indexes(doc, 0)
    region = np.s_[192:240, 128:256]
    before = Image.fromarray(pal16[a[region]])
    logo = Image.open(GEN / 'logo.png').convert('RGB')
    arr = np.asarray(logo).astype(np.int32)
    alpha = np.clip((arr.max(axis=2) - 18) * 255 // 42, 0, 255).astype(np.uint8)
    rgba = Image.fromarray(np.dstack([arr.astype(np.uint8), alpha]))
    rgba = rgba.crop(rgba.getchannel('A').point(lambda v: 255 if v > 30 else 0).getbbox())
    rgba.thumbnail((126, 46), Image.LANCZOS)
    canvas = Image.new('RGBA', (128, 48)); canvas.alpha_composite(rgba, ((128 - rgba.width) // 2, (48 - rgba.height) // 2))
    c = np.asarray(canvas)
    opaque = np.flatnonzero(pal16[:, 3] > 0)
    clear = int(np.flatnonzero(pal16[:, 3] == 0)[0]) if (pal16[:, 3] == 0).any() else 0
    idx = np.full((48, 128), clear, np.uint8)
    m = c[:, :, 3] >= 128
    idx[m] = opaque[palette_indices(c[m], pal16[opaque])]
    a[region] = idx
    doc = hud.put_indexes(doc, 0, a)
    after = Image.fromarray(pal16[idx])
    side_by_side(before, after, PREVIEW / 'demo_logo.png', scale=4)
    (out / 'scrscrn.ppg').write_bytes(doc.data)
    report.append(dict(part='scrscrn.ppg', texture=0, region='x128-255 y192-239', palette_row=29,
                       change='attract-demo 3rd STRIKE logo -> PUMP FIGHTERS (assets/gen/logo.png)'))
    print('scrscrn.ppg demo logo -> logo.png (PUMP FIGHTERS)')


# ------------------------------------------------------------------ ef02 stage names + ovals
def tile_maps(bank, pattern, meta, shape):
    cover = np.zeros(shape, bool)
    owner = np.full(shape, -1, np.int32); first = owner.copy()
    tiles = bank.tiles(pattern)
    for t in tiles:
        x, y = t['x'] - meta['origin'][0], t['y'] - meta['origin'][1]
        reg = np.s_[y:y + t['h'], x:x + t['w']]
        cover[reg] = True
        o = owner[reg]; o[(o < 0) & (t['pixels'] != 0)] = t['index']
        f = first[reg]; f[f < 0] = t['index']
    return tiles, cover, np.where(owner >= 0, owner, first)


def snap_tiles(a, tiles, owner, pal, base):
    out = np.zeros_like(a)
    for t in tiles:
        m = (owner == t['index']) & (a[:, :, 3] > 0)
        if not m.any(): continue
        pr = pal.for_tile(t['attr'], base)
        op = np.flatnonzero(pr[:, 3] > 0)
        out[m] = pr[op[palette_indices(a[m], pr[op])]]
    return out


def place_slanted(bank, pattern, pal, text, original):
    """The stage/history names are rotated (rising to the right). Draw the new name in the same white,
    dark-outlined heavy italic, rotated to the original's slant and fitted into the tiles that exist."""
    img, meta = bank.render(pattern, pal, metadata=True)
    o = np.asarray(original); H, W = o.shape[:2]       # geometry from the original name (re-runs don't drift)
    assert img.size == original.size
    tiles, cover, owner = tile_maps(bank, pattern, meta, (H, W))
    ys, xs = np.nonzero(o[:, :, 3] > 0)
    pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0] if vt[0][0] > 0 else -vt[0]
    ang = -np.degrees(np.arctan2(ax[1], ax[0]))
    along = (pts - c) @ ax; across = (pts - c) @ np.array([-ax[1], ax[0]])
    L = along.max() - along.min(); T = np.percentile(across, 97) - np.percentile(across, 3)
    f = font('C:/Windows/Fonts/ariblk.ttf', 120)
    base = Image.new('RGBA', (2400, 300)); d = ImageDraw.Draw(base)
    d.text((30, 30), text, font=f, fill=(255, 255, 255, 255), stroke_width=9, stroke_fill=(16, 16, 24, 255))
    base = base.crop(base.getbbox())
    sh = .18
    base = base.transform((base.width + int(base.height * sh), base.height), Image.AFFINE,
                          (1, sh, -base.height * sh, 0, 1, 0), Image.BICUBIC)
    best = None
    for hs in (1.3, 1.15, 1.0, .9, .8):
        th = max(6, int(T * hs)); tw = min(int(base.width * th / base.height), int(L * 1.02))
        t = base.resize((max(1, tw), th), Image.LANCZOS).rotate(ang, Image.BICUBIC, expand=True)
        al = np.asarray(t)[:, :, 3] > 110
        for dx in range(-6, 7, 2):
            for dy in range(-6, 7, 2):
                ox = int(round(c[0] - t.width / 2 + dx)); oy = int(round(c[1] - t.height / 2 + dy))
                xs0, ys0 = max(0, ox), max(0, oy); xe, ye = min(W, ox + t.width), min(H, oy + t.height)
                if xe <= xs0 or ye <= ys0: continue
                sub = al[ys0 - oy:ye - oy, xs0 - ox:xe - ox]
                inside = (sub & cover[ys0:ye, xs0:xe]).sum(); lost = al.sum() - inside
                score = inside - 4 * lost
                if best is None or score > best[0]: best = (score, t, ox, oy)
    _, t, ox, oy = best
    canvas = Image.new('RGBA', (W + 400, H + 400)); canvas.alpha_composite(t, (ox + 200, oy + 200))
    a = np.asarray(canvas.crop((200, 200, 200 + W, 200 + H))).copy()
    a[:, :, 3] = np.where((a[:, :, 3] >= 110) & cover, 255, 0)
    return Image.fromarray(snap_tiles(a, tiles, owner, pal, meta['palette_base']))


def build_ef02(ext, out, report):
    import hud
    bankpath = out / 'ef02_usa.bin'
    bank = SpriteBank((bankpath if bankpath.exists() else ext / 'ef02_usa.bin').read_bytes(),
                      SpriteBank.open(ext / 'ef02_usa.bin').info)
    default = (out / 'default.bin') if (out / 'default.bin').exists() else (ext / 'default.bin')
    pal = Palettes(default.read_bytes())
    orig = SpriteBank.open(ext / 'ef02_usa.bin'); opal = Palettes((ext / 'default.bin').read_bytes())
    sheet = []
    for table in (STAGE_NAMES, HISTORY_NAMES):
        for p, cid in table.items():
            img = place_slanted(bank, p, pal, NAMES[BY_ID[cid]], orig.render(p, opal))
            bank = SpriteBank(bank.replace(p, img, pal), bank.info)
            sheet.append((orig.render(p, opal), bank.render(p, pal)))
    report.append(dict(part='ef02_usa.bin', patterns='760-780, 1031-1051', change='stage-intro / history names'))
    for p, cid in OVALS.items():
        mascot = BY_ID[cid]
        old, meta = bank.render(p, pal, metadata=True)
        o = np.asarray(orig.render(p, opal)); mask = o[:, :, 3] > 0     # original oval (re-runs don't drift)
        assert old.size == (o.shape[1], o.shape[0])
        h, w = mask.shape
        # keep the red/white oval frame: pixels near the opaque footprint's outline
        yy, xx = np.indices(mask.shape)
        ys, xs = np.nonzero(mask)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        rx, ry = (xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2
        inner = ((xx - cx) / (rx - 3.5)) ** 2 + ((yy - cy) / (ry - 3.5)) ** 2 <= 1
        face = hud.head(mascot, (int(2 * rx) - 6, int(2 * ry) - 6))
        canvas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        bgc = tuple(int(v) for v in np.median(o[inner & mask][:, :3], axis=0)) if (inner & mask).any() else (255, 255, 255)
        fill = Image.new('RGBA', (w, h), bgc + (255,))
        canvas.alpha_composite(fill)
        canvas.alpha_composite(face, (int(cx - face.width / 2 + .5), int(cy - face.height / 2 + .5)))
        c = np.asarray(canvas).copy()
        a = o.copy(); sel = inner & mask
        a[sel] = c[sel]
        desired = np.zeros_like(a)
        tiles = bank.tiles(p)
        owner = np.full((h, w), -1, np.int32); first = owner.copy()
        for t in tiles:
            x, y = t['x'] - meta['origin'][0], t['y'] - meta['origin'][1]
            reg = np.s_[y:y + t['h'], x:x + t['w']]
            ow = owner[reg]; ow[(ow < 0) & (t['pixels'] != 0)] = t['index']
            f = first[reg]; f[f < 0] = t['index']
        owner = np.where(owner >= 0, owner, first)
        for t in tiles:
            m = (owner == t['index']) & (a[:, :, 3] > 0)
            if not m.any(): continue
            pr = pal.for_tile(t['attr'], meta['palette_base'])
            op = np.flatnonzero(pr[:, 3] > 0)
            desired[m] = pr[op[palette_indices(a[m], pr[op])]]
        final = Image.fromarray(desired)
        bank = SpriteBank(bank.replace(p, final, pal), bank.info)
        sheet.append((orig.render(p, opal), bank.render(p, pal)))
    report.append(dict(part='ef02_usa.bin', patterns='812-825, 826 (Gouki), 827-831',
                       change='stage-intro oval portraits -> mascot heads (existing shared palette rows)'))
    bankpath.write_bytes(bank.data)
    used = bank.texture_count * 4 + sum(bank.cell(c)[2] for c in range(bank.texture_count))
    print(f'ef02 stage names/ovals; ef02 texture bytes {used:,}/{bank.end - bank.tex:,}')
    W = max(a.width for a, _ in sheet) * 2 + 8; H = sum(max(a.height, b.height) + 2 for a, b in sheet)
    s = Image.new('RGBA', (W, H), (37, 40, 50, 255)); y = 0
    for a, b in sheet:
        s.alpha_composite(a, (0, y)); s.alpha_composite(b, (W // 2 + 4, y)); y += max(a.height, b.height) + 2
    s.save(PREVIEW / 'ef02_stage_names_ovals.png')
    report.append(dict(part='ef02_usa.bin', texture_bytes=used, budget=bank.end - bank.tex))


# ------------------------------------------------------------------ previews
def side_by_side(before, after, path, scale=2):
    before, after = before.convert('RGBA'), after.convert('RGBA')
    w, h = before.size
    c = Image.new('RGBA', (w * 2 + 6, h), (37, 40, 50, 255))
    c.alpha_composite(before); c.alpha_composite(after, (w + 6, 0))
    if scale != 1: c = c.resize((c.width * scale, c.height * scale), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True); c.save(path)


def contact(doc, title, path):
    tiles = []
    for i in range(len(doc.textures)):
        bgc = Image.new('RGBA', (256, 256), (37, 40, 50, 255))
        tiles.append(Image.alpha_composite(bgc, Image.fromarray(render_tex(doc, i, title))))
    cols = 10; s = 128
    sheet = Image.new('RGB', (cols * s, ((len(tiles) + cols - 1) // cols) * s))
    for i, t in enumerate(tiles):
        sheet.paste(t.convert('RGB').resize((s, s)), ((i % cols) * s, (i // cols) * s))
    sheet.save(path)


def build(ext, out):
    ext, out = Path(ext), Path(out)
    out.mkdir(parents=True, exist_ok=True); PREVIEW.mkdir(parents=True, exist_ok=True)
    report = []
    title = decode_color((ext / 'title.bin').read_bytes()).reshape(-1, 64, 4).copy()
    source = PPG((ext / 'Opening.ppg').read_bytes())
    contact(source, title, PREVIEW / 'Opening_before.png')
    doc = build_ppg_panels(source, title, report)
    doc = build_claudia(doc, title, report, slack=max(0, len(source.data) - len(doc.data) - 4096))
    doc = build_letters(doc, title, report)
    # every texture not listed above keeps its exact bytes
    touched = {t for v in PPG_PANELS.values() for t in v[0]} | set(CLAUDIA_TEX) | set(range(35, 47))
    for i in range(len(source.textures)):
        if i not in touched: assert doc.raw(i) == source.raw(i) and tex_rects(doc, i) == tex_rects(source, i)
    assert len(doc.data) <= len(source.data), 'Opening.ppg grew past its original size'
    contact(doc, title, PREVIEW / 'Opening_after.png')
    (out / 'Opening.ppg').write_bytes(doc.data)
    report.append(dict(part='Opening.ppg', size=len(doc.data), original_size=len(source.data)))
    print(f'Opening.ppg {len(doc.data):,} bytes (original {len(source.data):,})')
    build_bg20(ext, out, title, report)
    newtitle = encode_color(title.reshape(-1, 4))
    oldtitle = (ext / 'title.bin').read_bytes()
    assert len(newtitle) == len(oldtitle)
    (out / 'title.bin').write_bytes(newtitle)
    changed = [r for r in range(len(title)) if newtitle[r * 128:(r + 1) * 128] != oldtitle[r * 128:(r + 1) * 128]]
    print('title.bin rows rewritten:', changed)
    report.append(dict(part='title.bin', rows_rewritten=changed))
    build_demo_logo(ext, out, report)
    build_ef02(ext, out, report)
    (PREVIEW / 'mapping.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    # dry run into a scratch copy of the current overrides (the real build goes through build_mod.py)
    import shutil
    test = MOD / 'build' / 'overrides_opening_test'
    shutil.rmtree(test, ignore_errors=True)
    shutil.copytree(MOD / 'build' / 'overrides', test)
    build(MOD / 'build' / 'extracted', test)
