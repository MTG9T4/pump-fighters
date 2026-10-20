"""Render, dump and replace SFIII LZ-p6 sprite patterns (bank-local numbers)."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
from common import ROOT, decode_color, palette_indices, output_path
from codecs_sf3 import p6_decode, p6_encode

BANKS = json.loads(Path(__file__).with_name('banks.json').read_text())
CHARACTERS = ['Gill', 'Alex', 'Ryu', 'Yun', 'Dudley', 'Necro', 'Hugo', 'Ibuki', 'Elena', 'Oro',
              'Yang', 'Ken', 'Sean', 'Urien', 'Akuma', 'Chun-Li', 'Makoto', 'Q', 'Twelve', 'Remy']
# sel_data.c Victory_Color_Data; eff76.c Setup_Color_76 adds it to 0x90.
VICTORY_COLORS = (0, 53, 0, 0, 53, 49, 0, 51, 51, 53, 0, 53, 0, 0, 0, 51, 51, 0, 49, 0)
WIN_BASES = {855 + i + (i >= 15): 0x90 + offset for i, offset in enumerate(VICTORY_COLORS)}
WIN_BASES[870] = WIN_BASES[869]  # English Akuma / Japanese Gouki


@lru_cache(maxsize=3)
def morton(side):
    y, x = np.indices((side, side)); index = np.zeros_like(x)
    for bit in range(5):
        index |= ((x >> bit) & 1) << (bit * 2 + 1)
        index |= ((y >> bit) & 1) << (bit * 2)
    return index


class Palettes:
    def __init__(self, data, character=False, row=0, bank=0):
        self.rgba = decode_color(data).reshape(-1, 64, 4)
        self.character = character; self.row = row; self.bank = bank
        if character and (len(data) not in (3584, 7168) or not 0 <= row < 28 or not 0 <= bank < len(data) // 3584):
            raise ValueError('palette must have 1 or 2 sets of 28x64 colours; row 0..27, half must exist')
        if not character and len(data) != 61440:
            raise ValueError('default.bin must contain 480 x 64 ARGB1555 colours')
        if not character:
            # color3rd.c init_trans_color_ram overrides this row after loading default.bin.
            self.rgba[479] = 0
            self.rgba[479, 1:, 3] = 255

    def for_tile(self, attr, base=0x90):
        if not self.character:
            row = (base + (attr & 0x1ff)) & 0x1ff
            if not 32 <= row < 512:
                raise ValueError(f'ColorRAM row {row} not in default.bin')
            return self.rgba[row - 32]
        # color3rd.c: selected colour at RAM 0 and 8, fixed rows16..21 at 1..6/9..14.
        part = attr & 0x1ff
        if part not in range(7) and part not in range(8, 15):
            raise ValueError(f'character part palette {part} is not mapped by COL loader')
        local = part & 7
        row = self.row if local == 0 else 15 + local
        half = (self.bank ^ (part >> 3)) if getattr(self, 'gill', False) else self.bank
        return self.rgba[half * 28 + row]


def load_palettes(bankpath, source=None, row=0, half=0):
    path = Path(bankpath)
    character = path.name.lower().startswith('pl')
    if source is None:
        source = path.with_name(path.stem + 'pl.bin' if character else 'default.bin')
        if not source.is_file():
            # Use the original archive read-only if the companion was not extracted.
            from afs import AFS
            data = AFS(ROOT / 'game' / 'SF33RD.AFS').read(source.name)
        else:
            data = source.read_bytes()
    else:
        data = Path(source).read_bytes()
    pal = Palettes(data, character, row, half)
    pal.gill = path.name.lower() == 'pl00.bin'
    return pal


class SpriteBank:
    def __init__(self, data, info):
        self.data = data; self.info = info; self.tex = info['to_tex']
        if info['afs'] == 1454 and len(data) > 1128236:
            raise ValueError('ef02_usa.bin exceeds the conservative 1,128,236-byte size budget')
        self.end = info['to_chd'] or len(data)
        # Pattern count: stored per bank (skins may move pattern 0's list to the end of the file, so the first
        # table entry no longer marks where the table ends); falls back to the first entry for unknown banks.
        self.count = info.get('count') or struct.unpack_from('<I', data)[0] // 4
        if self.count <= 0 or self.count * 4 > self.tex or self.end > len(data):
            raise ValueError('invalid bank pattern table/texture boundary')
        self.pattern_offsets = struct.unpack_from(f'<{self.count}I', data)
        self.texture_count = struct.unpack_from('<I', data, self.tex)[0] // 4
        if self.texture_count <= 0 or self.tex + self.texture_count * 4 >= self.end:
            raise ValueError('invalid bank texture table')
        self.offsets = list(struct.unpack_from(f'<{self.texture_count}I', data, self.tex))
        # cells may also sit after the character data (PUMP COMBAT skins grow the bank past to_chd)
        if any(o < self.texture_count * 4 or self.tex + o >= len(data) for o in self.offsets):
            raise ValueError('texture offset outside bank')
        self._cells = {}

    @classmethod
    def open(cls, path, kind=None):
        name = kind or Path(path).name.lower()
        if name not in BANKS:
            raise ValueError('unknown bank filename; use --kind with an original bank name')
        return cls(Path(path).read_bytes(), BANKS[name])

    def entries(self, pattern):
        if not 0 <= pattern < self.count:
            raise ValueError(f'pattern must be 0..{self.count - 1}')
        pos = self.pattern_offsets[pattern]
        # PUMP COMBAT skins may move a pattern's tile list to the end of the file (past the character data)
        tail = pos >= self.end
        if not (self.count * 4 <= pos < self.tex - 1 or tail and pos < len(self.data) - 1):
            raise ValueError('pattern offset outside transform data')
        count = struct.unpack_from('<H', self.data, pos)[0]
        if pos + 2 + count * 8 > (len(self.data) if tail else self.tex):
            raise ValueError('pattern tiles cross texture table')
        return [struct.unpack_from('<hhHH', self.data, pos + 2 + i * 8) for i in range(count)]

    def cell(self, code):
        if code not in self._cells:
            if not 0 <= code < self.texture_count:
                raise ValueError('tile code outside texture table')
            pos = self.tex + self.offsets[code]
            wh = self.data[pos]; side = ((wh & 3) + 1) * 8
            w, h = (wh & 0xe0) >> 2, (wh & 0x1c) * 2
            if side not in (8, 16, 32) or not (0 < w <= side and 0 < h <= side):
                raise ValueError(f'invalid TEX header 0x{wh:02x}')
            raw, used = p6_decode(self.data[pos + 1:self.end if pos < self.end else len(self.data)], side * side)
            pixels = np.frombuffer(raw, dtype=np.uint8)[morton(side)]
            self._cells[code] = (wh, pixels, used + 1, w, h)
        return self._cells[code]

    def tiles(self, pattern):
        x = y = 0; tiles = []
        for i, (dx, dy, attr, code) in enumerate(self.entries(pattern)):
            x -= dx; y -= dy
            wh, pixels, _, w, h = self.cell(code)
            pixels = pixels[:h, :w]
            if attr & 0x8000: pixels = pixels[:, ::-1]
            if attr & 0x4000: pixels = pixels[::-1]
            tiles.append(dict(index=i, x=x, y=y, w=w, h=h, attr=attr, code=code, pixels=pixels))
        return tiles

    def palette_base(self, pattern, palettes):
        return 0 if palettes.character else WIN_BASES.get(pattern, 0x90) if self.info['afs'] == 1454 else 0x90

    def render(self, pattern, palettes, base=None, metadata=False):
        if base is None: base = self.palette_base(pattern, palettes)
        tiles = self.tiles(pattern)
        if not tiles:
            im = Image.new('RGBA', (1, 1))
            return (im, dict(pattern=pattern, origin=[0, 0], tiles=[])) if metadata else im
        left = min(t['x'] for t in tiles); top = min(t['y'] for t in tiles)
        right = max(t['x'] + t['w'] for t in tiles); bottom = max(t['y'] + t['h'] for t in tiles)
        if right - left > 8192 or bottom - top > 8192:
            raise ValueError('unreasonable pattern bounds')
        canvas = Image.new('RGBA', (right - left, bottom - top))
        # Earlier chips are in front (also documented by arcade_cg.c).
        for t in reversed(tiles):
            rgba = palettes.for_tile(t['attr'], base)[t['pixels']]
            canvas.alpha_composite(Image.fromarray(rgba), (t['x'] - left, t['y'] - top))
        meta = dict(pattern=pattern, global_cg=self.info['base'] + pattern, origin=[left, top],
                    palette_base=base,
                    width=canvas.width, height=canvas.height,
                    tiles=[{k: v for k, v in t.items() if k != 'pixels'} for t in tiles])
        return (canvas, meta) if metadata else canvas

    def replace(self, pattern, image, palettes, base=None):
        if base is None: base = self.palette_base(pattern, palettes)
        original, meta = self.render(pattern, palettes, base, True)
        if image.size != original.size:
            raise ValueError(f'PNG must use the exported {original.width}x{original.height} canvas')
        desired = np.asarray(image.convert('RGBA')).copy()
        # Transparent RGB is not drawn, and cannot be recovered from compositing.
        desired[desired[:, :, 3] == 0] = 0
        before = np.asarray(original)
        changed = np.any(desired != before, axis=2)
        if not np.any(changed):
            return self.data
        tiles = self.tiles(pattern); claimed = np.zeros(changed.shape, dtype=bool); replacements = {}
        for t in tiles:
            x = t['x'] - meta['origin'][0]; y = t['y'] - meta['origin'][1]
            region = np.s_[y:y + t['h'], x:x + t['w']]
            pal = palettes.for_tile(t['attr'], base)
            opaque = pal[t['pixels']][:, :, 3] != 0
            visible = ~claimed[region] & (opaque | (before[region][:, :, 3] == 0))
            mask = changed[region] & visible
            if not np.any(mask):
                claimed[region] |= opaque
                continue
            cell = self.cell(t['code'])[1].copy()
            local = t['pixels'].copy()
            # Only changed visible pixels are quantized; hidden pixels stay intact.
            target = pal[local].copy(); target[mask] = desired[region][mask]
            # Match transparent pixels to any transparent palette index.
            transparent = target[:, :, 3] == 0
            target[transparent] = pal[np.flatnonzero(pal[:, 3] == 0)[0]] if np.any(pal[:, 3] == 0) else target[transparent]
            indexes = palette_indices(target, pal, exact=True)
            local[mask] = indexes[mask]
            claimed[region] |= pal[local][:, :, 3] != 0
            if t['attr'] & 0x4000: local = local[::-1]
            if t['attr'] & 0x8000: local = local[:, ::-1]
            cell[:t['h'], :t['w']] = local
            raw = np.empty(cell.size, dtype=np.uint8); raw[morton(cell.shape[0]).reshape(-1)] = cell.reshape(-1)
            blob = bytes([self.cell(t['code'])[0]]) + p6_encode(raw.tobytes())
            replacements[t['index']] = blob
        # Fixed to_tex/to_chd offsets remain valid. Rebuild cells inside the original size budget.
        users = {}
        for p in range(self.count):
            for i, (_, _, _, code) in enumerate(self.entries(p)):
                users.setdefault(code, []).append((p, i))
        blobs = []
        for code in range(self.texture_count):
            pos = self.tex + self.offsets[code]; used = self.cell(code)[2]
            blobs.append(self.data[pos:pos + used])
        result = bytearray(self.data)
        # Clone shared cells, so replacing this pattern never changes other patterns.
        assigned = {}
        for i, blob in replacements.items():
            code = tiles[i]['code']
            if blob == blobs[code]: continue
            key = (code, blob)
            if key in assigned:
                newcode = assigned[key]
            elif users[code] == [(pattern, i)]:
                blobs[code] = blob; newcode = code; assigned[key] = code
            else:
                newcode = len(blobs); blobs.append(blob); assigned[key] = newcode
            if newcode > 65535:
                raise ValueError('replacement exceeds 16-bit cell code limit')
            struct.pack_into('<H', result, self.pattern_offsets[pattern] + 2 + i * 8 + 6, newcode)
        offsets = []; payload = bytearray()
        for blob in blobs:
            offsets.append(len(blobs) * 4 + len(payload)); payload.extend(blob)
        texture = struct.pack(f'<{len(offsets)}I', *offsets) + payload
        budget = self.end - self.tex
        if len(texture) > budget:
            raise ValueError(f'replacement needs {len(texture)} texture bytes; fixed budget is {budget} '
                             '(simplify the image; shared cells require clones)')
        result[self.tex:self.end] = texture + b'\0' * (budget - len(texture))
        check = SpriteBank(bytes(result), self.info)
        if not np.array_equal(np.asarray(check.render(pattern, palettes, base)), desired):
            raise ValueError('PNG changes pixels outside tiles or has an occlusion conflict; no output written')
        # The transform list itself can be shared by multiple pattern numbers.
        aliases = [p for p in range(self.count) if p != pattern and self.pattern_offsets[p] == self.pattern_offsets[pattern]]
        if aliases:
            raise ValueError(f'pattern transform list aliases {aliases}; cannot change only this pattern')
        return bytes(result)

    def standing_patterns(self, count=3):
        chd = self.info['to_chd']
        if not chd:
            raise ValueError('standing animation exists only in character banks')
        # charid.c maps WORK char_table koc values to these CharInitData members.
        members = (0, 1, 3, 4, 5, 6, 2, 7, 8, 9)

        def resolve(koc, index, frame):
            if not 0 <= koc < 10 or index < 0 or frame < 0:
                raise ValueError('invalid standing animation script target')
            table = chd + struct.unpack_from('<I', self.data, chd + members[koc] * 4)[0]
            anim = table + struct.unpack_from('<I', self.data, table + index * 4)[0]
            stride = struct.unpack_from('<I', self.data, anim - 8)[0] * 4
            if stride < 8 or stride > 128:
                raise ValueError('invalid standing animation record stride')
            return anim + frame * stride

        state = (0, 0, 0); stack = []; seen = set(); patterns = []
        for _ in range(256):
            if state in seen: break
            seen.add(state)
            koc, index, frame = state; pos = resolve(*state)
            command, target_koc, target_index, target_frame = struct.unpack_from('<Hhhh', self.data, pos)
            state = (koc, index, frame + 1)
            if command < 0x100:
                if command in (3, 4, 5):  # jmp/jpss/jsr
                    if command == 5: stack.append(state)
                    state = (target_koc, target_index, max(1, target_frame) - 1)
                elif command == 6:
                    if not stack: break
                    state = stack.pop()
                elif command == 2:  # end -> frame in the same animation
                    state = (koc, index, max(1, target_frame) - 1)
                elif command not in (0, 7, 8, 9, 12, 14):
                    # Enough for all 20 original idle sequences; don't guess unknown flow.
                    break
                continue
            cg = struct.unpack_from('<H', self.data, pos + 6)[0]
            local = cg - self.info['base']
            if 0 <= local < self.count and local not in patterns:
                patterns.append(local)
                if len(patterns) == count: break
        if not patterns:
            raise ValueError('no drawable frames in standing animation')
        return patterns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('render', help='compose one local pattern with a palette source')
    r.add_argument('bank'); r.add_argument('palette_source', help='default.bin/plNNpl.bin, or auto')
    r.add_argument('pattern', type=int); r.add_argument('out_png')
    d = sub.add_parser('dump', help='render a pattern range; auto companion palette by default')
    d.add_argument('bank'); d.add_argument('--patterns', required=True, help='inclusive A-B (or one number)')
    d.add_argument('outdir'); d.add_argument('--palette-source')
    rp = sub.add_parser('replace', help='replace one pattern; PNG must already use its tile palettes')
    rp.add_argument('bank'); rp.add_argument('pattern', type=int); rp.add_argument('png'); rp.add_argument('out_bin')
    rp.add_argument('--palette-source')
    for parser in (r, d, rp):
        parser.add_argument('--kind', choices=list(BANKS), help='original bank type for a renamed bank')
        parser.add_argument('--row', type=int, default=0, help='character palette row 0..27 (selectable 0..12)')
        parser.add_argument('--half', type=int, choices=(0, 1), default=0)
        parser.add_argument('--base', type=lambda x: int(x, 0), default=None, help='ColorRAM base (ef02 default 0x90)')
    a = p.parse_args()
    try:
        bank = SpriteBank.open(a.bank, a.kind)
        kindpath = Path(a.bank).with_name(a.kind) if a.kind else Path(a.bank)
        source = getattr(a, 'palette_source', None)
        pal = load_palettes(kindpath, None if source == 'auto' else source, a.row, a.half)
        base = a.base
        if a.cmd == 'render':
            im, meta = bank.render(a.pattern, pal, base, True)
            im.save(output_path(a.out_png, a.bank, source) if source and source != 'auto' else output_path(a.out_png, a.bank))
            output_path(str(a.out_png) + '.json', a.bank).write_text(json.dumps(meta, indent=2) + '\n')
        elif a.cmd == 'dump':
            numbers = [int(x) for x in a.patterns.split('-')]
            if len(numbers) not in (1, 2) or numbers[-1] < numbers[0]: raise ValueError('use an inclusive A-B range')
            manifest = []
            for pattern in range(numbers[0], numbers[-1] + 1):
                im, meta = bank.render(pattern, pal, base, True)
                im.save(output_path(Path(a.outdir) / f'pattern_{pattern:04}.png', a.bank))
                manifest.append(meta)
            output_path(Path(a.outdir) / 'patterns.json', a.bank).write_text(json.dumps(manifest, indent=2) + '\n')
        else:
            result = bank.replace(a.pattern, Image.open(a.png), pal, base)
            output_path(a.out_bin, a.bank, a.png).write_bytes(result)
    except (ValueError, OSError, IndexError, struct.error) as e:
        p.exit(1, f'error: {e}\n')


if __name__ == '__main__':
    main()
