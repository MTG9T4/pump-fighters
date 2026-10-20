"""Export/import COL character palettes and preview actual standing animation frames."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from common import decode_color, encode_color, output_path
from spritebank import SpriteBank, load_palettes


def read_col(path):
    data = Path(path).read_bytes()
    if len(data) not in (3584, 7168):
        raise ValueError('expected 3584 bytes (28x64), or 7168 bytes for Gill (2x28x64)')
    return data


def export(source, target):
    data = read_col(source)
    halves = len(data) // 3584
    colors = decode_color(data).reshape(halves, 28, 64, 4)
    size = 12; gutter = 114
    image = Image.new('RGB', (gutter + 64 * size, halves * 28 * size), '#20232b')
    draw = ImageDraw.Draw(image); rows = []
    for half in range(halves):
        for row in range(28):
            y = (half * 28 + row) * size
            draw.text((3, y), f'{half}:{row:02} ' + ('select' if row <= 12 else 'aux'), fill='white')
            rgba = colors[half, row]
            for i, color in enumerate(rgba):
                x = gutter + i * size
                bg = 160 if (i + row) % 2 else 90
                rgb = tuple((int(c) * int(color[3]) + bg * (255 - int(color[3]))) // 255 for c in color[:3])
                draw.rectangle((x, y, x + size - 1, y + size - 1), fill=rgb)
            rows.append(dict(half=half, row=row, selectable=row < 13,
                             colors=['#' + bytes(c).hex() for c in rgba]))
    image.save(output_path(target, source))
    doc = dict(schema='pumpcombat-col-v1', format='ARGB1555_LE', source=Path(source).name,
               note='colors are #RRGGBBAA; all stored rows preserved; selectable rows 0..12', rows=rows)
    output_path(Path(target).with_suffix('.json'), source).write_text(json.dumps(doc, indent=2) + '\n')


def import_col(source, original, target):
    doc = json.loads(Path(source).read_text())
    if doc.get('schema') != 'pumpcombat-col-v1' or doc.get('format') != 'ARGB1555_LE':
        raise ValueError('unknown palette JSON schema/format')
    data = bytearray(read_col(original)); seen = set()
    for entry in doc['rows']:
        half, row = entry['half'], entry['row']
        if type(half) is not int or type(row) is not int or not 0 <= half < len(data) // 3584 or not 0 <= row < 28:
            raise ValueError('invalid half/row')
        if (half, row) in seen: raise ValueError('duplicate palette row')
        seen.add((half, row)); colors = entry['colors']
        if len(colors) != 64 or any(not isinstance(c, str) or len(c) != 9 or c[0] != '#' for c in colors):
            raise ValueError('each row needs 64 #RRGGBBAA colours')
        rgba = np.array([list(bytes.fromhex(c[1:])) for c in colors], dtype=np.uint8)
        offset = (half * 28 + row) * 128
        data[offset:offset + 128] = encode_color(rgba)
    output_path(target, source, original).write_bytes(data)


def preview(bankpath, palettepath, row, target, kind=None, half=0):
    if not 0 <= row < 28: raise ValueError('palette row must be 0..27')
    bank = SpriteBank.open(bankpath, kind)
    pal = load_palettes(Path(bankpath).with_name(kind) if kind else bankpath, palettepath, row, half)
    patterns = bank.standing_patterns()
    images = [bank.render(n, pal, 0) for n in patterns]
    w = max(im.width for im in images) + 24; h = max(im.height for im in images) + 48
    canvas = Image.new('RGBA', (w * len(images), h), '#252832'); draw = ImageDraw.Draw(canvas)
    for i, (im, n) in enumerate(zip(images, patterns)):
        canvas.alpha_composite(im, (i * w + (w - im.width) // 2, h - 12 - im.height))
        draw.text((i * w + 8, 8), f'row {row} / pattern {n}', fill='white')
    canvas.resize((canvas.width * 2, canvas.height * 2), Image.Resampling.NEAREST).save(output_path(target, bankpath, palettepath))
    output_path(str(target) + '.json', bankpath, palettepath).write_text(json.dumps(
        dict(row=row, half=half, patterns=patterns, animation='nmca[0], first distinct drawable frames'), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    ex = sub.add_parser('export', help='all stored rows (28, or Gill 56) as labelled swatches plus editable JSON')
    ex.add_argument('palette'); ex.add_argument('out_png')
    im = sub.add_parser('import', help='apply JSON rows to original COL, preserving unlisted rows')
    im.add_argument('json'); im.add_argument('orig_bin'); im.add_argument('out_bin')
    pr = sub.add_parser('preview', help='render three actual standing animation frames side by side')
    pr.add_argument('bank'); pr.add_argument('palette'); pr.add_argument('row', type=int); pr.add_argument('out_png')
    pr.add_argument('--kind', help='original bank name when input was renamed'); pr.add_argument('--half', type=int, choices=(0, 1), default=0)
    a = p.parse_args()
    try:
        if a.cmd == 'export': export(a.palette, a.out_png)
        elif a.cmd == 'import': import_col(a.json, a.orig_bin, a.out_bin)
        else: preview(a.bank, a.palette, a.row, a.out_png, a.kind, a.half)
    except (ValueError, KeyError, OSError, IndexError) as e:
        p.exit(1, f'error: {e}\n')


if __name__ == '__main__':
    main()
