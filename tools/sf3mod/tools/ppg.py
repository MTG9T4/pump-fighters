"""Convert PPG texture chunks to PNG and replace textures in their original format."""
import argparse
import json
from pathlib import Path
import struct
import zlib
import numpy as np
from PIL import Image
from common import decode_color, encode_color, palette_indices, output_path
from codecs_sf3 import lz77_decode, lz77_encode


# LOW_NIBBLE_FIRST
# 4bpp: the low nibble is always the left (even) pixel. That is how the PS2 GS (PSMT4), the PSP GU
# (T4) and 3SX's renderer read the bytes; they upload pTEX data as is. Pixel flag 0x20 only selects
# the SDK's software accessor order (prilay.c) and is not used when textures are drawn. The original
# scrscrn.ppg textures carry 0x20 and are stored low-first.


def decompress(src, mode, size):
    if size > 256 * 1024 * 1024:
        raise ValueError('unreasonable decompressed chunk size')
    if mode == 0:
        raw = src[:size]
    elif mode == 1:
        raw = lz77_decode(src, size)
    elif mode == 2:
        dec = zlib.decompressobj()
        raw = dec.decompress(src, size + 1)
        if not dec.eof:
            raise ValueError('incomplete/oversized zlib stream')
    else:
        raise ValueError(f'unsupported compression {mode}')
    if len(raw) != size:
        raise ValueError(f'decompressed size {len(raw)} != {size}')
    return raw


class PPG:
    def __init__(self, data):
        self.data = data; self.chunks = []; self.textures = []; self.palettes = []
        pos = 0
        while pos + 4 <= len(data):
            magic = data[pos:pos + 4]
            if magic == b'pEND':
                self.end = pos
                break
            if pos + 8 > len(data):
                raise ValueError('truncated PPG chunk header')
            size = struct.unpack_from('>I', data, pos + 4)[0]
            if size < 16 or pos + size > len(data):
                raise ValueError(f'invalid PPG chunk at 0x{pos:x}')
            c = dict(magic=magic.decode('ascii'), offset=pos, size=size)
            self.chunks.append(c)
            if magic == b'pTEX':
                w, h, comp, pixel, fmt, trans = struct.unpack_from('>4BHH', data, pos + 8)
                c.update(width=w * 16, height=h * 16, compression=comp & 3, pixel=pixel,
                         format=fmt, transforms=trans, payload=pos + 16 + trans * 3)
                if not w or not h or c['payload'] > pos + size:
                    raise ValueError('invalid pTEX dimensions/transform table')
                self.textures.append(c)
            elif magic == b'pPAL':
                comp, mode, fmt, count = struct.unpack_from('>BBHH', data, pos + 10)
                colors = (16, 64, 256, 1)[mode & 3]
                raw = decompress(data[pos + 16:pos + size], comp & 3,
                                 count * colors * (4 if fmt == 0x8888 else 2))
                c.update(compression=comp & 3, mode=mode, format=fmt, count=count, colors=colors,
                         rgba=decode_color(raw, fmt, bool(mode & 4)).reshape(count, colors, 4))
                self.palettes.append(c)
            elif magic == b'pCMP':
                c.update(compression=data[pos + 10] & 3, expanded=struct.unpack_from('>I', data, pos + 12)[0])
            pos += (size + 3) & ~3
        else:
            raise ValueError('PPG missing pEND')

    def palette(self, texture_index, palette=0):
        # PPG stores no texture->palette binding: draw calls select cix externally.
        mode = self.textures[texture_index]['pixel'] & 3
        colors = 16 if mode == 0 else 256
        pals = [row for c in self.palettes if c['colors'] == colors for row in c['rgba']]
        if not pals and mode == 1:
            pals = [row for c in self.palettes if c['colors'] == 64 for row in c['rgba']]
        if not pals and hasattr(self, 'external_palette'):
            pals = list(self.external_palette.reshape(-1, colors, 4))
        if not 0 <= palette < len(pals):
            raise ValueError('paletted texture requires a valid pPAL row (--palette)')
        return pals[palette][:16 if mode == 0 else 256]

    def raw(self, index):
        c = self.textures[index]; mode = c['pixel'] & 3
        size = c['width'] * c['height'] * (4, 8, 16, 32)[mode] // 8
        return decompress(self.data[c['payload']:c['offset'] + c['size']], c['compression'], size)

    def pixels(self, index, palette=0):
        c = self.textures[index]; mode = c['pixel'] & 3
        raw = self.raw(index)
        if mode <= 1:
            idx = np.frombuffer(raw, dtype=np.uint8)
            if mode == 0:
                lo, hi = idx & 15, idx >> 4
                idx = np.stack((lo, hi), axis=1).reshape(-1)  # see LOW_NIBBLE_FIRST
            pal = self.palette(index, palette)
            if np.any(idx >= len(pal)):
                raise ValueError('texture index outside chosen palette')
            rgba = pal[idx]
        else:
            rgba = decode_color(raw, c['format'], bool(c['pixel'] & 4))
        return rgba.reshape(c['height'], c['width'], 4)

    def replace(self, index, image, palette=0):
        c = self.textures[index]
        if image.size != (c['width'], c['height']):
            raise ValueError(f'PNG must be {c["width"]}x{c["height"]}')
        rgba = np.asarray(image.convert('RGBA')); mode = c['pixel'] & 3
        # Retain duplicate indexes, unused bits and compression on an exact no-op.
        if np.array_equal(rgba, self.pixels(index, palette)):
            return self.data
        if mode < 2:
            idx = palette_indices(rgba, self.palette(index, palette)).reshape(-1)
            if mode == 0:
                a, b = idx[::2], idx[1::2]
                raw = (b << 4 | a).tobytes()  # see LOW_NIBBLE_FIRST
            else:
                raw = idx.tobytes()
        else:
            raw = encode_color(rgba, c['format'], bool(c['pixel'] & 4))
        packed = (raw if c['compression'] == 0 else lz77_encode(raw) if c['compression'] == 1
                  else zlib.compress(raw, 9))
        chunk = bytearray(self.data[c['offset']:c['payload']] + packed)
        struct.pack_into('>I', chunk, 4, len(chunk))
        chunk.extend(b'\0' * (-len(chunk) % 4))
        end = c['offset'] + ((c['size'] + 3) & ~3)
        return self.data[:c['offset']] + chunk + self.data[end:]

    def export(self, outdir, palette=0):
        outdir = Path(outdir)
        manifest = []
        for i, c in enumerate(self.textures):
            Image.fromarray(self.pixels(i, palette)).save(output_path(outdir / f'tex_{i:03}.png'))
            manifest.append(dict(index=i, **c, palette=palette if c['pixel'] & 3 < 2 else None))
        output_path(outdir / 'textures.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    ex = sub.add_parser('topng', help='export every pTEX and a textures.json manifest')
    ex.add_argument('ppg'); ex.add_argument('outdir'); ex.add_argument('--palette', type=int, default=0, help='flattened pPAL row (default 0)')
    ex.add_argument('--palette-source', help='external PPG or raw LE ARGB1555 palettes (default: sibling default.bin when no pPAL)')
    im = sub.add_parser('frompng', help='replace a zero-based pTEX; retain dimensions, format and compression')
    im.add_argument('orig_ppg'); im.add_argument('tex_index', type=int); im.add_argument('png'); im.add_argument('out_ppg')
    im.add_argument('--palette', type=int, default=0)
    im.add_argument('--palette-source')
    a = p.parse_args()
    try:
        source = a.ppg if a.cmd == 'topng' else a.orig_ppg
        doc = PPG(Path(source).read_bytes())
        if a.palette_source or not doc.palettes:
            companion = Path(a.palette_source) if a.palette_source else Path(source).with_name('default.bin')
            if companion.is_file():
                data = companion.read_bytes()
                if data.startswith((b'pTEX', b'pPAL')):
                    doc.palettes = PPG(data).palettes
                else:
                    doc.external_palette = decode_color(data)
        if a.cmd == 'topng':
            doc.export(a.outdir, a.palette)
        else:
            if not 0 <= a.tex_index < len(doc.textures):
                raise ValueError('texture index out of range')
            result = doc.replace(a.tex_index, Image.open(a.png), a.palette)
            output_path(a.out_ppg, source, a.png).write_bytes(result)
    except (ValueError, OSError, IndexError, struct.error, zlib.error) as e:
        p.exit(1, f'error: {e}\n')


if __name__ == '__main__':
    main()
