"""PUMP COMBAT HUD and select-ring sprites; originals are always read-only.

Run through build_mod.py --only ...,hud. Previews and a machine-readable mapping
are written beside overrides, under hud_preview. No game code changes required.

Source evidence (relative to mod/3sx/src/sf33rd/Source/Game):
* ui/sc_data.c Player_Name_Pos_TBL / Face_Pos_TBL; ui/sc_sub.c player_name,
  player_face, Scrscreen_Init: page 1 / pPAL 0 row 1; page 0 / pPAL 1 row id*13.
  PSP Game/sc_data.c and Game/sc_sub.c use the same cells and 5x1 / 5x3 widths.
* screen/sel_pl.c Setup_Face_Sub creates effect_70 for ids 1..19.
  Setup_FACE_BG marks BG1 Unsubstantial (virtual), not a background texture.
* effect/eff70.c sets family 2, color 0x2090, _sel_pl_char_table, animation 0,
  frame id+1. charset.c uses one-based frames. bin2obj/char_table.c table 0
  therefore selects local pattern 1251+id in ef02_usa.bin (global base 27104).
* stage/bg.c Bg_Texture_Load2 and etcBgGixCnvTable[2] map SelectNew's background
  effects, not these faces. Its transforms use 16px units (PPGFile.c).

Ring palettes are shared with patterns 811+id (Chun-Li onward: 812+id).
Keep every default.bin byte, including the portrait step's palette edits.
"""
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD / 'tools'))
from ppg import PPG
from common import decode_color, encode_color, palette_indices, output_path
from codecs_sf3 import lz77_encode
from spritebank import SpriteBank, Palettes

# Literal copies of the two verified 21-character cell tables (units: 8 pixels).
NAME_CELLS = ((8, 0), (0, 1), (5, 1), (10, 1), (0, 2), (5, 2), (10, 2),
              (0, 3), (5, 3), (10, 3), (0, 4), (5, 4), (10, 4), (0, 5),
              (5, 5), (0, 6), (5, 6), (10, 6), (0, 7), (5, 7), (10, 7))
EXTRA_NAME_CELLS = {14: ((10, 5), (10, 7))}
FACE_CELLS = ((16, 0), (21, 0), (26, 0), (16, 3), (21, 3), (26, 3), (16, 6),
              (21, 6), (26, 6), (16, 9), (21, 9), (26, 9), (16, 12), (21, 12),
              (26, 12), (21, 15), (26, 15), (16, 18), (21, 18), (26, 18), (16, 21))
CAST = ((2, 'swordcat', 'SWORDCAT'), (11, 'inu', 'INU'), (4, 'jeanphil', 'JEAN PHIL'),
        (13, 'elon', 'ELON'), (5, 'pepe', 'PEPE'), (15, 'claudia', 'CLAUDIA'),
        (3, 'wif', 'WIF'), (16, 'popcat', 'POPCAT'), (17, 'pnut', 'PNUT'),
        (6, 'moodeng', 'MOO DENG'), (1, 'giga', 'GIGA'), (14, 'goat', 'GOAT'),
        (12, 'chillguy', 'CHILL GUY'), (10, 'ponke', 'PONKE'), (7, 'mew', 'MEW'),
        (8, 'pippin', 'PIPPIN'), (9, 'slerf', 'SLERF'), (19, 'bonk', 'BONK'),
        (18, 'fwog', 'FWOG'), (0, 'pill', 'PILL'))
# Gill (id 0, the boss) has a HUD name plate and face but no select-ring oval (sel_pl.c: ids 1..19)
NO_RING = {0}
# Art-specific head boxes in normalized source coordinates; exclude hands/weapons.
HEADS = {'swordcat': (.25, 0, .87, .41), 'inu': (.14, 0, .75, .40),
         'jeanphil': (.24, 0, .77, .40), 'elon': (.20, 0, .61, .37),
         'pepe': (.29, 0, .84, .37), 'claudia': (.24, 0, .77, .37),
         'wif': (.30, 0, .76, .38), 'popcat': (.36, 0, .86, .38),
         'pnut': (.29, .07, .98, .50), 'moodeng': (.28, .02, .84, .47),
         'giga': (.39, 0, .72, .285), 'goat': (.23, 0, .90, .35),
         'chillguy': (.16, 0, .83, .38), 'ponke': (.12, 0, .78, .44),
         'mew': (.18, 0, .87, .48), 'pippin': (.22, 0, .85, .40),
         'slerf': (.20, 0, .77, .32), 'bonk': (.29, 0, .75, .38),
         'fwog': (.08, .04, .86, .55), 'pill': (.30, .02, .86, .36)}
# Nearby fists intrude into rectangular head crops. Remove only those corners,
# in normalized portrait coordinates, before keying/resizing the head.
HEAD_EXCLUDES = {'giga': ((.65, .24, .72, .285),),
                 'goat': ((.77, .29, .90, .35),),
                 'chillguy': ((.70, .30, .83, .38),),
                 'mew': ((.72, .455, .87, .48),),
                 'pippin': ((.67, .36, .85, .40),)}
# Front-facing original Ryu/Ken/Chun-Li; the others face inward on the P1 side.
# The game mirrors the entire HUD face for P2 in scfont_sqput_face (0x8000).
HUD_FLIP = {'swordcat', 'elon', 'popcat'}
# All nine newly assigned original HUD faces look right on P1, as does their
# portrait art; none needs an additional flip. P2 is still mirrored by the game.

# Hand-hinted four-pixel caps, five scan lines. No antialiasing or host fonts.
FONT = dict(zip('ACDEFGHIJLMNOPRSTUW', (
    '0110/1001/1111/1001/1001', '0111/1000/1000/1000/0111',
    '1110/1001/1001/1001/1110', '1111/1000/1110/1000/1111',
    '1111/1000/1110/1000/1000', '0111/1000/1011/1001/0111',
    '1001/1001/1111/1001/1001', '111/010/010/010/111',
    '0011/0001/0001/1001/0110', '1000/1000/1000/1000/1111',
    '10001/11011/10101/10001/10001', '1001/1101/1011/1001/1001',
    '0110/1001/1001/1001/0110', '1110/1001/1110/1000/1000',
    '1110/1001/1110/1010/1001', '0111/1000/0110/0001/1110',
    '1111/0110/0110/0110/0110', '1001/1001/1001/1001/0110',
    '1001/1001/1001/1111/0110')))
FONT.update(B='1110/1001/1110/1001/1110',
            K='1001/1010/1100/1010/1001',
            Y='1001/1001/0110/0110/0110')


def indexes(doc, i):
    c = doc.textures[i]
    a = np.frombuffer(doc.raw(i), np.uint8)
    if c['pixel'] & 3 == 0:
        # Low nibble = left pixel, whatever flag 0x20 says (see LOW_NIBBLE_FIRST in tools/ppg.py)
        a = np.stack((a & 15, a >> 4), axis=1).reshape(-1)
    return a.reshape(c['height'], c['width']).copy()


def chunk_replace(doc, chunk, raw):
    """Use PPG's codec/layout metadata, retaining every other chunk byte for byte."""
    packed = (raw if chunk['compression'] == 0 else lz77_encode(raw)
              if chunk['compression'] == 1 else zlib.compress(raw, 9))
    start = chunk['offset']; payload = chunk.get('payload', start + 16)
    result = bytearray(doc.data[start:payload] + packed)
    struct.pack_into('>I', result, 4, len(result))
    result.extend(b'\0' * (-len(result) % 4))
    end = start + ((chunk['size'] + 3) & ~3)
    return PPG(doc.data[:start] + result + doc.data[end:])


def put_indexes(doc, i, a):
    c = doc.textures[i]; flat = a.reshape(-1)
    if c['pixel'] & 3 == 0:
        x, y = flat[::2], flat[1::2]
        flat = y << 4 | x  # low nibble = left pixel (see LOW_NIBBLE_FIRST in tools/ppg.py)
    return chunk_replace(doc, c, flat.tobytes())


def name_plate(text):
    # A one-pixel space and no extra gap after it keep CHILL GUY at 39 pixels.
    glyphs = [np.array([[int(p) for p in row] for row in FONT[c].split('/')], bool)
              if c != ' ' else np.zeros((5, 1), bool) for c in text]
    w = sum(g.shape[1] for g in glyphs) + len(text.replace(' ', '')) - 1
    assert w <= 39, (text, w)
    mask = np.zeros((8, 40), bool); x = (40 - w) // 2
    for c, g in zip(text, glyphs):
        mask[1:6, x:x + g.shape[1]] = g; x += g.shape[1] + (c != ' ')
    # Original plates use flat pale index 7, dark index 2, white 8, shade 4,
    # and a red index 1 baseline. Retain exactly these indexes and pPAL rows.
    a = np.full((8, 40), 7, np.uint8)
    outline = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(mask, (dy, dx), (0, 1))
        if dx == 1: shifted[:, 0] = False
        if dx == -1: shifted[:, -1] = False
        outline |= shifted
    a[outline] = 2; a[mask] = 8
    # Original right-edge grey accent, without thinning the main strokes.
    a[mask & ~np.roll(mask, -1, axis=1) & (np.indices(mask.shape)[0] == 5)] = 4
    a[7] = 1
    return a, w


def head(mascot, size, flip=False):
    src = Image.open(MOD / 'assets/gen' / f'portrait_{mascot}.png').convert('RGBA')
    draw = ImageDraw.Draw(src)
    for rect in HEAD_EXCLUDES.get(mascot, ()):
        draw.rectangle(tuple(round(v * (src.width if i % 2 == 0 else src.height))
                             for i, v in enumerate(rect)), fill=(0, 0, 0, 0))
    box = tuple(round(v * (src.width if i % 2 == 0 else src.height))
                for i, v in enumerate(HEADS[mascot]))
    src = src.crop(box)
    # Flood key only the black background connected to crop edges, retaining
    # dark eyes, mouths and hair enclosed by the illustrated head.
    a = np.array(src); dark = a[:, :, :3].max(axis=2) < 42
    from collections import deque
    seen = np.zeros(dark.shape, bool); q = deque()
    h, w = dark.shape
    for y, x in [(y, x) for y in range(h) for x in (0, w-1)] + [(y, x) for x in range(w) for y in (0, h-1)]:
        if dark[y, x] and not seen[y, x]: seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and dark[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True; q.append((yy, xx))
    a[seen] = 0; src = Image.fromarray(a)
    if flip: src = ImageOps.mirror(src)
    # A wide head crop at tiny resolution; preserve proportions and center it.
    src = ImageOps.contain(src, size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', size)
    canvas.alpha_composite(src, ((size[0]-src.width)//2, (size[1]-src.height)//2))
    src = canvas
    a = np.array(src); a[:, :, 3] = np.where(a[:, :, 3] >= 128, 255, 0)
    a[a[:, :, 3] == 0] = 0
    return Image.fromarray(a)


def quantize15(im):
    a = np.array(im); opaque = a[:, :, 3] != 0
    sample = Image.fromarray(a[opaque, :3].reshape(-1, 1, 3))
    q = sample.quantize(colors=15, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE)
    pal = np.zeros((16, 4), np.uint8)
    pal[1:, :3] = np.array(q.getpalette()[:45], np.uint8).reshape(15, 3)
    pal[1:, 3] = 255
    pal = decode_color(encode_color(pal))
    idx = np.zeros(a.shape[:2], np.uint8)
    idx[opaque] = palette_indices(a[opaque], pal[1:]) + 1
    return idx, pal


def compare(path, old, new, scale=4):
    old, new = old.convert('RGBA'), new.convert('RGBA')
    w, h = old.size; canvas = Image.new('RGBA', (w*2+3, h), '#252832')
    canvas.alpha_composite(old); canvas.alpha_composite(new, (w+3, 0))
    canvas.resize((canvas.width*scale, h*scale), Image.Resampling.NEAREST).save(path)


def ring_patterns():
    source = MOD.parent / 'mod/3sx/src/bin2obj/char_table.c'
    values = [int(v, 0) for v in re.findall(r'0x[\da-fA-F]+|\b\d+\b',
              source.read_text().split('{', 1)[1].split('}', 1)[0])]
    raw = struct.pack(f'<{len(values)}I', *values)
    return {id: struct.unpack_from('<H', raw, values[0] + id*8 + 6)[0] - 27104
            for id, _, _ in CAST if id not in NO_RING}


def build(ext, out):
    ext, out = Path(ext), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    preview = out.parent / 'hud_preview'; preview.mkdir(parents=True, exist_ok=True)
    source = PPG((ext / 'scrscrn.ppg').read_bytes()); doc = source
    names, faces = indexes(doc, 1), indexes(doc, 0)
    name_mask, face_mask = np.zeros_like(names, bool), np.zeros_like(faces, bool)
    face_pal = source.palettes[1]['rgba'].copy(); pal_mask = np.zeros(len(face_pal), bool)
    records = []
    overview = Image.new('RGBA', (83, 36*len(CAST)), '#252832')
    for id, mascot, text in CAST:
        nx, ny = (v*8 for v in NAME_CELLS[id]); fx, fy = (v*8 for v in FACE_CELLS[id])
        plate, width = name_plate(text)
        names[ny:ny+8, nx:nx+40] = plate; name_mask[ny:ny+8, nx:nx+40] = True
        # Akuma has two more plates the US game shows in fights: a second GOUKI (10,5) and AKUMA (10,7)
        for cx, cy in EXTRA_NAME_CELLS.get(id, ()):
            names[cy*8:cy*8+8, cx*8:cx*8+40] = plate; name_mask[cy*8:cy*8+8, cx*8:cx*8+40] = True
        icon = head(mascot, (38, 22), mascot in HUD_FLIP)
        canvas = Image.new('RGBA', (40, 24)); canvas.alpha_composite(icon, (1, 0))
        # Keep the original pale top/side border and red bottom baseline style.
        d = ImageDraw.Draw(canvas); d.line((0, 0, 0, 22), fill=(213, 230, 230, 255))
        d.line((39, 0, 39, 22), fill=(213, 230, 230, 255)); d.line((0, 23, 39, 23), fill=(131, 0, 0, 255))
        idx, pal = quantize15(canvas)
        faces[fy:fy+24, fx:fx+40] = idx; face_mask[fy:fy+24, fx:fx+40] = True
        face_pal[id*13:(id+1)*13] = pal; pal_mask[id*13:(id+1)*13] = True
        compare(preview / f'name_{mascot}.png', Image.fromarray(source.palettes[0]['rgba'][1][indexes(source, 1)[ny:ny+8, nx:nx+40]]), Image.fromarray(source.palettes[0]['rgba'][1][plate]), 6)
        for color in range(13):
            oldpal = source.palettes[1]['rgba'][id*13+color]
            compare(preview / f'face_{mascot}_{color:02}.png', Image.fromarray(oldpal[indexes(source, 0)[fy:fy+24, fx:fx+40]]), Image.fromarray(pal[idx]))
        y = len(records)*36
        overview.alpha_composite(Image.fromarray(source.palettes[0]['rgba'][1][indexes(source, 1)[ny:ny+8, nx:nx+40]]), (0, y))
        overview.alpha_composite(Image.fromarray(source.palettes[0]['rgba'][1][plate]), (43, y))
        overview.alpha_composite(Image.fromarray(source.palettes[1]['rgba'][id*13][indexes(source, 0)[fy:fy+24, fx:fx+40]]), (0, y+8))
        overview.alpha_composite(Image.fromarray(pal[idx]), (43, y+8))
        records.append(dict(id=id, mascot=mascot, text=text, head_box=HEADS[mascot], hud_flip=mascot in HUD_FLIP,
                            name_xy=[nx, ny], name_size=[40, 8], ink_width=width,
                            face_xy=[fx, fy], face_size=[40, 24], palette_rows=list(range(id*13, (id+1)*13))))
    overview.resize((overview.width*6, overview.height*6), Image.Resampling.NEAREST).save(preview / 'hud_all.png')
    doc = put_indexes(doc, 0, faces); doc = put_indexes(doc, 1, names)
    c = doc.palettes[1]
    doc = chunk_replace(doc, c, encode_color(face_pal, c['format'], bool(c['mode'] & 4)))
    assert np.array_equal(indexes(doc, 0)[~face_mask], indexes(source, 0)[~face_mask])
    assert np.array_equal(indexes(doc, 1)[~name_mask], indexes(source, 1)[~name_mask])
    assert np.array_equal(doc.palettes[1]['rgba'][~pal_mask], source.palettes[1]['rgba'][~pal_mask])
    for i in range(2, len(source.textures)): assert doc.raw(i) == source.raw(i)
    for i in (0, 2, 3): assert np.array_equal(doc.palettes[i]['rgba'], source.palettes[i]['rgba'])
    # A composite atlas uses each face's own pPAL row, unlike a single-row export.
    for i, oldidx, newidx in ((0, indexes(source, 0), faces), (1, indexes(source, 1), names)):
        old = source.palettes[0]['rgba'][1][oldidx].copy(); new = old.copy()
        if i == 1: new = source.palettes[0]['rgba'][1][newidx]
        else:
            for id in range(len(FACE_CELLS)):
                x, y = (v*8 for v in FACE_CELLS[id]); r = np.s_[y:y+24, x:x+40]
                row = (0 if id == 20 else id)*13
                old[r] = source.palettes[1]['rgba'][row][oldidx[r]]
                new[r] = face_pal[row][newidx[r]]
        compare(preview / f'scrscrn_texture_{i}.png', Image.fromarray(old), Image.fromarray(new), 2)

    bankpath = out / 'ef02_usa.bin'
    before = bankpath.read_bytes() if bankpath.exists() else (ext / bankpath.name).read_bytes()
    bank = SpriteBank(before, SpriteBank.open(ext / bankpath.name).info)
    texture_budget = bank.end - bank.tex
    incoming_texture_bytes = bank.texture_count*4 + sum(bank.cell(c)[2] for c in range(bank.texture_count))
    original_bank = SpriteBank.open(ext / bankpath.name)
    defaultpath = out / 'default.bin'
    default = defaultpath.read_bytes() if defaultpath.exists() else (ext / 'default.bin').read_bytes()
    palettes = Palettes(default); original_palettes = Palettes((ext / 'default.bin').read_bytes())
    mapping = ring_patterns(); changed = set(mapping.values())
    ring_sheet = Image.new('RGBA', (48*2+3, 32*len(CAST)), '#252832')
    for n, record in enumerate(records):
        id, mascot = record['id'], record['mascot']
        if id in NO_RING: continue
        pattern = mapping[id]
        assert pattern == 1251 + id
        old, meta = bank.render(pattern, palettes, metadata=True)
        assert old.size == (48, 32)
        # Original opaque oval is the exact footprint; retain its black rim.
        # Leave room inside the oval's rim for the horns, including their tips.
        ring_head = head(mascot, (40, 22)) if mascot in {'goat', 'pippin'} else head(mascot, old.size)
        ring_canvas = Image.new('RGBA', old.size)
        ring_canvas.alpha_composite(ring_head, ((48-ring_head.width)//2, (32-ring_head.height)//2 + (1 if mascot in {'goat', 'pippin'} else 0)))
        a = np.array(ring_canvas); original = np.array(old)
        mask = original[:, :, 3] != 0
        a[~mask] = 0; a[mask & (a[:, :, 3] == 0)] = (0, 0, 0, 255)
        yy, xx = np.indices(mask.shape)
        rim = ((xx-23.5)/22)**2 + ((yy-17)/13)**2 > .88
        a[rim & mask] = (0, 0, 0, 255)
        desired = np.zeros_like(a)
        for t in bank.tiles(pattern):
            x, y = t['x']-meta['origin'][0], t['y']-meta['origin'][1]
            reg = np.s_[y:y+t['h'], x:x+t['w']]
            p = palettes.for_tile(t['attr']); desired[reg] = p[palette_indices(a[reg], p)]
        final = Image.fromarray(desired)
        bank = SpriteBank(bank.replace(pattern, final, palettes), bank.info)
        old = original_bank.render(pattern, original_palettes)
        compare(preview / f'ring_{mascot}.png', old, final)
        ring_sheet.alpha_composite(old, (0, n*32)); ring_sheet.alpha_composite(final, (51, n*32))
        rows = sorted({144+(t['attr'] & 511) for t in bank.tiles(pattern)})
        users = [p for p in range(original_bank.count) if any(144+(a & 511) in rows for _, _, a, _ in original_bank.entries(p))]
        record.update(ring_pattern=pattern, ring_global_cg=27104+pattern, ring_palette_rows=rows,
                      ring_palette_shared_patterns=users, ring_tiles=meta['tiles'])
        print(f'hud {mascot:9s}: name ({record["name_xy"]}), face ({record["face_xy"]}), ring {pattern}, rows {rows}')
    ring_sheet.resize((ring_sheet.width*4, ring_sheet.height*4), Image.Resampling.NEAREST).save(preview / 'ring_all.png')
    # Verify all unaffected patterns against the incoming portraits/names override.
    prior = SpriteBank(before, bank.info)
    for p in range(bank.count):
        if p in changed: continue
        assert prior.entries(p) == bank.entries(p), f'changed placement: {p}'
        for _, _, _, code in prior.entries(p):
            assert np.array_equal(prior.cell(code)[1], bank.cell(code)[1]), f'changed cell in {p}'
    if defaultpath.exists(): assert defaultpath.read_bytes() == default
    output_path(out / 'scrscrn.ppg', ext / 'scrscrn.ppg').write_bytes(doc.data)
    output_path(bankpath, ext / bankpath.name).write_bytes(bank.data)
    texture_bytes = bank.texture_count*4 + sum(bank.cell(c)[2] for c in range(bank.texture_count))
    report = dict(status='PASS', runtime_tested=False, original_left_new_right=True,
                  ef02_texture_budget=texture_budget, ef02_incoming_texture_bytes=incoming_texture_bytes,
                  ef02_texture_bytes=texture_bytes, ef02_texture_free_bytes=texture_budget-texture_bytes,
                  default_bin_changed=False, default_sha256=hashlib.sha256(default).hexdigest(),
                  hud_unrelated_pixels_and_palettes_preserved=True,
                  ring_unrelated_patterns_preserved=True, characters=records,
                  limits=['Ring uses existing shared 64-color palettes; mascot colors are approximated.',
                          'Static decoded previews; gameplay/emulator validation not performed.'])
    (preview / 'mapping.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'ef02 textures: {texture_bytes:,}/{texture_budget:,} bytes; {texture_budget-texture_bytes:,} free '
          f'(before rings: {incoming_texture_bytes:,})')
    print('HUD previews ->', preview)


if __name__ == '__main__':
    build(MOD / 'build/extracted', MOD / 'build/overrides')
