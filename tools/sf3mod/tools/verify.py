"""Reproducible codec and real-archive checks; all artifacts stay in sf3mod/build/."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw
from afs import AFS, align
from common import ROOT, ORIGINAL, decode_color, encode_color, output_path
from codecs_sf3 import lz77_decode, lz77_encode, p6_decode, p6_encode
from ppg import PPG
from spritebank import SpriteBank, Palettes, load_palettes, BANKS, CHARACTERS
from pal import export, import_col, preview
from runtime_compat import prepare_character_bank


class Codecs(unittest.TestCase):
    def test_skin_build_requires_every_bank_and_palette(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'sf3mod'))
        from build_mod import collect_skin_sources, CAST
        root = Path('skin-fixture')
        files = set()
        for character in CAST.values():
            bank = f'pl{character["id"] + (character["id"] >= 15):02d}'
            files.update(root / bank / 'out' / name for name in (f'{bank}.bin', f'{bank}pl.bin'))
        with patch.object(Path, 'is_file', lambda path: path in files), \
             patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=1)):
            self.assertEqual(len(collect_skin_sources(root)), 2 * len(CAST))
            files.remove(root / 'pl20/out/pl20.bin')
            with self.assertRaisesRegex(ValueError, 'pl20.bin'):
                collect_skin_sources(root)

    def test_akuma_loader_compatibility(self):
        from unittest.mock import patch
        info = dict(afs=0, base=0, to_tex=1800, to_chd=1900, size=2200, count=167)
        data = bytearray(2200)
        struct.pack_into('<167I', data, 0, *([668] * 167))
        struct.pack_into('<HhhHH', data, 668, 1, 0, 0, 0, 0)
        blob = b'\x24' + p6_encode(bytes([1]) * 64)
        struct.pack_into('<I', data, 1800, 4)
        data[1804:1804 + len(blob)] = blob
        with patch.dict(BANKS, {'pl14.bin': info}):
            self.assertEqual(prepare_character_bank('pl14.bin', bytes(data)), bytes(data))
            struct.pack_into('<I', data, 166 * 4, len(data))
            tiles = [(4, 85, 0, 0), (-8, 0, 0, 0)]
            data += struct.pack('<H', 2) + b''.join(struct.pack('<hhHH', *t) for t in tiles)
            fixed = prepare_character_bank('pl14.bin', bytes(data))
            bank = SpriteBank(fixed, info)
            before = bank.entries(166)
            # Exactly the deletion / merge used by PSP and the console-balance web port.
            after = [(before[0][0] + before[1][0], before[0][1] + before[1][1], *before[1][2:]), *before[2:]]
            self.assertEqual(after, tiles)
            self.assertEqual(prepare_character_bank('pl14.bin', fixed), fixed)
            self.assertEqual(fixed[1900:2200], data[1900:2200])
            palette = bytearray(3584); struct.pack_into('<H', palette, 2, 0xffff)
            pal = Palettes(bytes(palette), True)
            wanted = SpriteBank(bytes(data), info).render(166, pal)
            self.assertEqual(bank.render(166, pal).tobytes(), wanted.tobytes())

    def test_lz77_opcodes(self):
        vectors = [(b'\x81\x03abc', b'abc'), (b'\x82\x00\x03abc', b'abc'),
                   (b'\x83A\x04', b'AAAA'), (b'\x84A\x00\x04', b'AAAA'),
                   (b'\x85\xfa\x04\x03', bytes([250, 254, 2])),
                   (b'\x86\xfa\x04\x00\x03', bytes([250, 254, 2])),
                   (b'\x81\x01A\x00\x14', b'AAAAA'),
                   (b'\x81\x01A\xc0\x01\x04', b'AAAAA'),
                   (b'\x81\x01A\xc0\x01\x84\x01', b'ABCDE'),
                   (b'\x81\x00' + b'A' * 256, b'A' * 256),
                   (b'\x83A\x00', b'A' * 256),
                   (b'\x84A\0\0', b'A' * 65536)]
        for stream, expected in vectors:
            self.assertEqual(lz77_decode(stream, len(expected)), expected)
        raw = bytes(range(256)) * 300
        self.assertEqual(lz77_decode(lz77_encode(raw), len(raw)), raw)
        for stream in (b'\x80', b'\x00\x12', b'\xc0'):
            with self.assertRaises(ValueError): lz77_decode(stream, 1)

    def test_p6(self):
        rng = np.random.default_rng(381)
        for side in (8, 16, 32):
            samples = [bytes(side * side), bytes([63]) * (side * side),
                       rng.integers(0, 64, side * side, dtype=np.uint8).tobytes(),
                       bytes(range(16)) * (side * side // 16)]
            for raw in samples:
                packed = p6_encode(raw)
                decoded, consumed = p6_decode(packed, len(raw))
                self.assertEqual(decoded, raw); self.assertEqual(consumed, len(packed))
        with self.assertRaises(ValueError): p6_decode(b'\x40', 2)
        with self.assertRaises(ValueError): p6_encode(b'\x40')

    def test_color_words(self):
        for fmt in (0x1555, 0x565, 0x4444):
            for little in (True, False):
                raw = np.arange(65536, dtype=np.uint16).astype('<u2' if little else '>u2').tobytes()
                self.assertEqual(encode_color(decode_color(raw, fmt, little), fmt, little), raw)

    def test_ppg_matrix(self):
        for mode in range(4):
            for comp in range(3):
                for little in (True, False):
                    for low_first in (True, False) if mode == 0 else (True,):
                        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
                        rgba[:, :, 3] = 255; rgba[:8, :, 0] = 255; rgba[8:, :, 1] = 255
                        pal = np.zeros((16 if mode == 0 else 256, 4), dtype=np.uint8)
                        pal[:, 3] = 255; pal[0, 0] = 255; pal[1, 1] = 255
                        idx = np.zeros((16, 16), dtype=np.uint8); idx[8:] = 1
                        # Left/right asymmetric, so a swapped nibble order cannot pass
                        idx[:, 0::4] ^= 1; rgba[idx == 0, :2] = (255, 0); rgba[idx == 1, :2] = (0, 255)
                        if mode == 0:
                            # Hardware order: low nibble = left pixel, with or without flag 0x20
                            a, b = idx.reshape(-1)[::2], idx.reshape(-1)[1::2]
                            raw = (a | b << 4).tobytes()
                        elif mode == 1: raw = idx.tobytes()
                        else: raw = encode_color(rgba, 0x1555 if mode == 2 else 0x8888, little)
                        packed = raw if comp == 0 else lz77_encode(raw) if comp == 1 else __import__('zlib').compress(raw)
                        pixel = mode | (4 if little else 0) | (0 if low_first else 0x20)
                        tex = struct.pack('>4sI4BHH', b'pTEX', 19 + len(packed), 1, 1, comp, pixel,
                                          (0x1555 if mode == 2 else 0x8888) if mode >= 2 else 0, 1) + b'xyz' + packed
                        tex += b'\0' * (-len(tex) % 4)
                        if mode < 2:
                            colors = encode_color(pal, little=little)
                            tex += struct.pack('>4sIHBBHH', b'pPAL', 16 + len(colors), 0, 0,
                                               (0 if mode == 0 else 2) | (4 if little else 0), 0x1555, 1) + colors
                        # Unknown/compressed auxiliary chunks and trailing bytes survive replacement.
                        tex += struct.pack('>4sIHBBI', b'pCMP', 17, 0, 0, 0, 1) + b'Q\0\0\0pENDtail'
                        doc = PPG(tex)
                        self.assertTrue(np.array_equal(doc.pixels(0), rgba), (mode, comp, little, low_first))
                        self.assertEqual(doc.replace(0, Image.fromarray(rgba)), tex)
                        changed = np.flip(rgba, 0).copy()
                        result = doc.replace(0, Image.fromarray(changed))
                        self.assertTrue(np.array_equal(PPG(result).pixels(0), changed))
                        self.assertEqual(result[-24:], tex[-24:])

    def test_original_guard(self):
        with self.assertRaises(ValueError): output_path(ORIGINAL)

    def test_sprite_shared_cells_and_budget(self):
        palette = bytearray(61440)
        struct.pack_into('<H', palette, 112 * 128 + 2, 0xffff)
        pal = Palettes(bytes(palette))
        self.assertTrue(np.all(pal.rgba[479, 1:] == [0, 0, 0, 255]))
        transforms = struct.pack('<2I', 8, 18) + struct.pack('<HhhHH', 1, 0, 0, 0, 0) * 2
        texture = struct.pack('<I', 4) + b'\x24' + p6_encode(bytes([1]) * 64)
        info = dict(afs=0, base=0, to_tex=28, to_chd=0)
        for padding in (0, 64):
            bank = SpriteBank(transforms + texture + bytes(padding), info)
            original = bank.render(0, pal)
            self.assertEqual(bank.replace(0, original, pal), bank.data)
            blank = Image.new('RGBA', (8, 8))
            if padding:
                changed = SpriteBank(bank.replace(0, blank, pal), info)
                self.assertEqual(changed.render(0, pal).getbbox(), None)
                self.assertEqual(changed.render(1, pal).tobytes(), original.tobytes())
                self.assertNotEqual(changed.entries(0)[0][3], changed.entries(1)[0][3])
            else:
                with self.assertRaisesRegex(ValueError, 'budget'): bank.replace(0, blank, pal)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pattern_map(source_root, out, bank, palettes):
    """Resolve source animation tables using the interpreter's 8-byte records."""
    source = (source_root / 'mod/3sx/src/bin2obj/char_table.c').read_text()
    values = [int(x, 0) for x in re.findall(r'0x[\da-fA-F]+|\b\d+\b', source.split('{', 1)[1].split('}', 1)[0])]
    raw = struct.pack(f'<{len(values)}I', *values); records = []
    for i, name in enumerate(CHARACTERS):
        record = dict(id=i, character=name)
        for table, key in ((2, 'portrait'), (1, 'name'), (9, 'win_name')):
            index = 23 if i == 14 and key == 'name' else 20 if i == 14 and key == 'win_name' else i
            cg = struct.unpack_from('<H', raw, values[table] + index * 8 + 6)[0]
            pattern = cg - bank.info['base']
            im, meta = bank.render(pattern, palettes, metadata=True)
            filename = f'{i:02}_{name.lower().replace("-", "_")}_{key}_{pattern:04}.png'
            im.save(out / filename)
            record[key] = dict(pattern=pattern, global_cg=cg, file=filename, width=im.width, height=im.height,
                               tiles=len(meta['tiles']), palette_base=meta['palette_base'],
                               palette_rows=sorted({meta['palette_base'] + (t['attr'] & 0x1ff) for t in meta['tiles']}))
        records.append(record)
    records[0]['portrait_1p'] = dict(pattern=1179, global_cg=28283, file='alternate_1179.png',
                                    note='eff38.c substitutes animation index 23 for first-player Gill')
    (out / 'pattern_map.json').write_text(json.dumps(records, indent=2) + '\n')
    # Also dump alternate Gouki / Gill faces and the Japanese name variants.
    for pattern in (18, 869, 1179, 1199, 1200, 1201):
        bank.render(pattern, palettes).save(out / f'alternate_{pattern:04}.png')
    sheet = Image.new('RGB', (2000, 1160), '#222530'); draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        portrait = Image.open(out / record['portrait']['file'])
        name = Image.open(out / record['name']['file'])
        portrait.thumbnail((390, 225))
        x, y = i % 5 * 400, i // 5 * 290
        sheet.paste(portrait, (x + (400 - portrait.width) // 2, y), portrait)
        sheet.paste(name, (x + (400 - name.width) // 2, y + 229), name)
        draw.text((x + 8, y + 267), f"{record['character']}: portrait {record['portrait']['pattern']}, name {record['name']['pattern']}", fill='white')
    sheet.save(out / 'contact_sheet.png')
    return records


def real_checks(path, out):
    out.mkdir(parents=True, exist_ok=True)
    print('Hashing original and extracting real AFS...', flush=True)
    initial = sha(path); archive = AFS(path); extracted = out / 'extracted'
    # Independently cross-check the saved PC layouts against the PSP table and real entries.
    psp = (ROOT / 'mod/3s-psp/src/Game/texgroup.c').read_text()
    layouts = [tuple(map(int, m.groups())) for m in re.finditer(
        r'\{\s*(\d+),\s*(\d+),\s*\d+,\s*\d+,\s*(\d+),\s*(\d+),\s*(\d+)\s*\}', psp)]
    for name, info in BANKS.items():
        assert archive.entries[info['afs']]['name'] == name
        assert (info['base'], info['afs'], info['size'], info['to_tex'], info['to_chd']) in layouts
        assert archive.entries[info['afs']]['size'] == info['size']
    archive.extract(extracted)
    empty = out / 'empty_overrides'; empty.mkdir(exist_ok=True)
    if any(empty.iterdir()): raise ValueError('empty_overrides must be empty')
    archive.repack(empty, out / 'empty_repack.afs')
    assert sha(out / 'empty_repack.afs') == initial
    assert all(e['offset'] % 2048 == 0 for e in archive.entries)
    print('Empty AFS repack is byte-identical. Checking every real PPG...', flush=True)
    ppg_count = texture_count = cmp_count = 0; formats = {}; no_palette = []
    default = decode_color(archive.read('default.bin'))
    for entry in archive.entries:
        if not entry['name'].lower().endswith('.ppg'): continue
        doc = PPG(archive.read(entry)); doc.external_palette = default
        for i, c in enumerate(doc.textures):
            pixels = doc.pixels(i)
            assert np.array_equal(PPG(doc.replace(i, Image.fromarray(pixels))).raw(i), doc.raw(i))
            texture_count += 1
        for c in doc.chunks:
            if c['magic'] == 'pCMP':
                from ppg import decompress
                decompress(doc.data[c['offset'] + 16:c['offset'] + c['size']], c['compression'], c['expanded'])
                cmp_count += 1
        if not doc.palettes and any(c['pixel'] & 3 < 2 for c in doc.textures): no_palette.append(entry['name'])
        if entry['name'] in ('TitleTM.ppg', 'scrscrn.ppg', 'SelectNew.ppg'):
            doc.export(out / 'ppg' / Path(entry['name']).stem)
            formats[entry['name']] = dict(size=entry['size'], textures=doc.textures,
                                          palettes=[{k: v for k, v in c.items() if k != 'rgba'} for c in doc.palettes])
        ppg_count += 1
    print(f'{ppg_count} PPG files / {texture_count} textures passed. Checking changed AFS repack...', flush=True)
    title = PPG(archive.read('TitleTM.ppg'))
    rgba = title.pixels(0).copy(); rgba[:] = (0, 0, 0, 0)
    changed = title.replace(0, Image.fromarray(rgba))
    override = out / 'overrides'; override.mkdir(exist_ok=True)
    (override / 'TitleTM.ppg').write_bytes(changed)
    archive.repack(override, out / 'modified_repack.afs')
    repacked = AFS(out / 'modified_repack.afs')
    for old, new in zip(archive.entries, repacked.entries):
        assert old['name'] == new['name']
        assert repacked.read(new) == (changed if old['name'] == 'TitleTM.ppg' else archive.read(old))
        assert new['offset'] % 2048 == 0
        assert struct.unpack_from('<I', repacked.names, new['index'] * 48 + 44)[0] == new['size']
        assert repacked.names[new['index'] * 48:new['index'] * 48 + 44] == archive.names[old['index'] * 48:old['index'] * 48 + 44]
    print('AFS payloads and updated name table passed. Rendering ef02 and all character previews...', flush=True)
    efout = out / 'ef02'; efout.mkdir(exist_ok=True)
    bank = SpriteBank.open(extracted / 'ef02_usa.bin'); palettes = load_palettes(extracted / 'ef02_usa.bin')
    mapping = pattern_map(ROOT, efout, bank, palettes)
    for code in range(bank.texture_count): bank.cell(code)
    original = bank.render(1180, palettes)
    assert bank.replace(1180, original, palettes) == bank.data
    blank = Image.new('RGBA', original.size)
    result = bank.replace(1180, blank, palettes)
    (efout / 'replaced_alex.bin').write_bytes(result)
    edited = SpriteBank(result, bank.info)
    assert np.asarray(edited.render(1180, palettes))[:, :, 3].max() == 0
    # Every other pattern retains its placements AND decompressed cell indexes.
    for pattern in range(bank.count):
        if pattern == 1180: continue
        assert bank.entries(pattern) == edited.entries(pattern)
        for _, _, _, code in bank.entries(pattern):
            assert np.array_equal(bank.cell(code)[1], edited.cell(code)[1])
    palette_sizes = {}; standing = {}; paldir = out / 'palettes'; paldir.mkdir(exist_ok=True)
    cell_count = bank.texture_count
    for name, info in BANKS.items():
        if not name.startswith('pl'): continue
        src = extracted / name; palette = src.with_name(src.stem + 'pl.bin')
        export(palette, paldir / f'{palette.stem}.png')
        import_col(paldir / f'{palette.stem}.json', palette, paldir / f'{palette.stem}.bin')
        assert palette.read_bytes() == (paldir / f'{palette.stem}.bin').read_bytes()
        preview(src, palette, 0, paldir / f'{src.stem}_preview.png')
        character = SpriteBank.open(src)
        standing[name] = character.standing_patterns()
        for code in range(character.texture_count): character.cell(code)
        cell_count += character.texture_count
        palette_sizes[palette.name] = palette.stat().st_size
    assert sha(path) == initial, 'ORIGINAL ARCHIVE CHANGED'
    report = dict(status='PASS', original_sha256=initial, archive_bytes=Path(path).stat().st_size,
                  afs_entries=len(archive.entries), empty_repack_byte_identical=True, modified_repack_other_payloads_identical=True,
                  ppg_files=ppg_count, ppg_textures=texture_count, pCMP_chunks=cmp_count, external_palette_ppgs=no_palette,
                  formats=formats, ef02=dict(size=len(bank.data), patterns=bank.count, cells=bank.texture_count,
                                           changed_pattern=1180, other_patterns_unchanged=True),
                  decoded_sprite_cells=cell_count, palette_sizes=palette_sizes, standing_patterns=standing,
                  portrait_name_map=mapping, emulator_tested=False)
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {cell_count} sprite cells, 20 palettes/previews, original SHA256 {initial}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', action='store_true', help='also extract/check real archive and create assets (about 2 GB scratch)')
    p.add_argument('--afs', type=Path, default=ORIGINAL)
    p.add_argument('--out', type=Path, default=ROOT / 'sf3mod/build')
    a = p.parse_args()
    if not a.out.resolve().is_relative_to((ROOT / 'sf3mod/build').resolve()):
        p.error('--out must be inside sf3mod/build/')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Codecs)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for script in ('afs.py', 'ppg.py', 'spritebank.py', 'pal.py'):
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)), '--help'],
                       check=True, stdout=subprocess.DEVNULL, env=env)
    if a.real: real_checks(a.afs, a.out)


if __name__ == '__main__': main()
