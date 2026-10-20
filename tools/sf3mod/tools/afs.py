"""Read/extract/repack AFS archives without touching the source archive."""
import argparse
import json
from pathlib import Path
import shutil
import struct
from common import output_path

SECTOR = 2048


def align(n, alignment=SECTOR):
    return (n + alignment - 1) // alignment * alignment


class AFS:
    def __init__(self, path):
        self.path = Path(path)
        total = self.path.stat().st_size
        with self.path.open('rb') as f:
            magic, count = struct.unpack('<4sI', f.read(8))
            if magic != b'AFS\0' or not 0 < count <= (total - 16) // 8:
                raise ValueError('invalid AFS header')
            self.entries = [dict(index=i, offset=o, size=s) for i, (o, s) in
                            enumerate(struct.iter_unpack('<II', f.read(count * 8)))]
            if any(e['offset'] + e['size'] > total for e in self.entries):
                raise ValueError('AFS entry outside archive')
            first = min(e['offset'] for e in self.entries if e['offset'])
            end = max(e['offset'] + e['size'] for e in self.entries)
            self.pointer_positions = []
            for pos in dict.fromkeys([8 + count * 8, first - 8]):
                f.seek(pos)
                o, s = struct.unpack('<II', f.read(8))
                if o >= end and s >= count * 48 and o + s <= total:
                    self.pointer_positions.append(pos)
                    self.name_offset, self.name_size = o, s
            if not self.pointer_positions:
                raise ValueError('AFS has no valid 48-byte name table')
            f.seek(self.name_offset)
            self.names = f.read(self.name_size)
            for e in self.entries:
                name = self.names[e['index'] * 48:e['index'] * 48 + 32].split(b'\0')[0].decode('ascii')
                if not name or name in ('.', '..') or any(c in name for c in '/\\:'):
                    raise ValueError(f'unsafe/empty entry name at {e["index"]}: {name!r}')
                e['name'] = name
            names = [e['name'].casefold() for e in self.entries]
            if len(set(names)) != len(names):
                raise ValueError('duplicate AFS names cannot be extracted/replaced unambiguously')

    def read(self, entry):
        if isinstance(entry, str):
            entry = next(e for e in self.entries if e['name'].casefold() == entry.casefold())
        elif isinstance(entry, int):
            entry = self.entries[entry]
        with self.path.open('rb') as f:
            f.seek(entry['offset'])
            return f.read(entry['size'])

    def extract(self, outdir):
        outdir = Path(outdir)
        for e in self.entries:
            output_path(outdir / e['name'], self.path).write_bytes(self.read(e))
        output_path(outdir / 'index.json', self.path).write_text(json.dumps(self.entries, indent=2) + '\n')

    def repack(self, overrides, target):
        target = output_path(target, self.path)
        overrides = Path(overrides)
        if not overrides.is_dir():
            raise ValueError('overrides directory does not exist')
        changes = {e['index']: overrides / e['name'] for e in self.entries
                   if (overrides / e['name']).is_file() and (overrides / e['name']).read_bytes() != self.read(e)}
        if not changes:
            shutil.copyfile(self.path, target)
            return
        # Keep same-size archives and every padding byte identical when possible.
        if all(p.stat().st_size == self.entries[i]['size'] for i, p in changes.items()):
            shutil.copyfile(self.path, target)
            with target.open('r+b') as f:
                for i, p in changes.items():
                    f.seek(self.entries[i]['offset'])
                    f.write(p.read_bytes())
            return
        first = min(e['offset'] for e in self.entries if e['offset'])
        with self.path.open('rb') as src, target.open('w+b') as dst:
            dst.write(src.read(first))
            offsets = []
            names = bytearray(self.names)
            for e in self.entries:
                if e['offset'] == 0 and e['index'] not in changes:
                    offsets.append((0, e['size']))
                    continue
                dst.write(b'\0' * (align(dst.tell()) - dst.tell()))
                data = changes[e['index']].read_bytes() if e['index'] in changes else self.read(e)
                offsets.append((dst.tell(), len(data)))
                dst.write(data)
                struct.pack_into('<I', names, e['index'] * 48 + 44, len(data))
            dst.write(b'\0' * (align(dst.tell()) - dst.tell()))
            name_offset = dst.tell()
            dst.write(names)
            # Preserve any trailer after the original attribute table.
            src.seek(self.name_offset + self.name_size)
            shutil.copyfileobj(src, dst)
            dst.write(b'\0' * (align(dst.tell()) - dst.tell()))
            dst.seek(8)
            for o, s in offsets:
                dst.write(struct.pack('<II', o, s))
            for pos in self.pointer_positions:
                dst.seek(pos)
                dst.write(struct.pack('<II', name_offset, len(names)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    ls = sub.add_parser('list', help='list index, offset, size and entry name')
    ls.add_argument('afs')
    ex = sub.add_parser('extract', help='extract named files plus index.json')
    ex.add_argument('afs'); ex.add_argument('outdir')
    rp = sub.add_parser('repack', help='replace matching filenames; empty overrides produce an exact copy')
    rp.add_argument('afs'); rp.add_argument('overrides_dir'); rp.add_argument('out_afs')
    a = p.parse_args()
    try:
        afs = AFS(a.afs)
        if a.cmd == 'list':
            for e in afs.entries:
                print(f'{e["index"]:4} {e["offset"]:10} {e["size"]:9} {e["name"]}')
        elif a.cmd == 'extract':
            afs.extract(a.outdir)
        else:
            afs.repack(a.overrides_dir, a.out_afs)
    except (ValueError, OSError, StopIteration, struct.error) as e:
        p.exit(1, f'error: {e}\n')


if __name__ == '__main__':
    main()
