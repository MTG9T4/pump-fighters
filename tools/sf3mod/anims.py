"""Every animation of one SF3 fighter as an ordered list of bank-local patterns.

The character data at to_chd starts with offsets to the CharInitData tables (charid.c: nmca, dmca, caca, cuca,
atca, saca, btca, exca, cbca, yuca in WORK char_table order 0..9). Each table is a list of offsets to animation
scripts; a script is a run of records cgd_type words long (stride at script-8). A record whose u16 code is
>= 0x100 is a drawn frame (byte 1 = frames held, u16 at +6 = global CG number); codes below 0x100 are
commands (charset.c decode_chcmd: 2 end, 3 jmp, 4 jpss, 5 jsr, 6 ret, 61 stop ...).

We walk each script straight through (branches are not taken: conditional jumps just fall through, which keeps
every drawn record in file order) and stop at end / jmp / ret / stop. That is all the skinning needs: which
patterns follow which, so poses can be kept consistent from one frame of an animation to the next.

    python sf3mod/anims.py pl18          # writes build/skin/pl18/anims.json and prints a summary
anims.json: {"anims": [{"koc": k, "index": i, "frames": [[pattern, hold], ...], "loop": bool}, ...]}
"""
import json, struct, sys
from pathlib import Path

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD / 'tools'))
from spritebank import SpriteBank  # noqa

MEMBERS = (0, 1, 3, 4, 5, 6, 2, 7, 8, 9)  # WORK char_table[koc] -> CharInitData member (charid.c)
KOC_NAMES = ('nmca', 'dmca', 'caca', 'cuca', 'atca', 'saca', 'btca', 'exca', 'cbca', 'yuca')
STOP = {2, 3, 6, 61}


def parse(bank):
    d = bank.data; chd = bank.info['to_chd']; base = bank.info['base']; n = len(d)
    tables = {}
    for koc in range(10):
        table = chd + struct.unpack_from('<I', d, chd + MEMBERS[koc] * 4)[0]
        offs = []; first = None
        while chd <= table < n:
            i = len(offs)
            if table + i * 4 + 4 > n or (first is not None and i * 4 >= first) or i > 2000: break
            off = struct.unpack_from('<I', d, table + i * 4)[0]
            if off < 8 or table + off + 8 > n: break
            first = off - 8 if first is None else min(first, off - 8)
            offs.append(table + off)
        tables[koc] = offs

    def stride_of(anim):
        st = struct.unpack_from('<h', d, anim - 8)[0] * 4  # cgd_type is an s16 (charset.c setupCharTableData)
        return st if 8 <= st <= 128 else None

    def walk(koc, ix, pat, depth, out):
        """Append (pattern, hold) for the records of one script from 1-based record pat; returns the exit."""
        if koc not in tables or not 0 <= ix < len(tables[koc]): return None
        anim = tables[koc][ix]; st = stride_of(anim)
        if st is None: return None
        pos = anim + max(0, pat - 1) * st
        for _ in range(400):
            if pos + 8 > n: return None
            code, tk, ti, tp = struct.unpack_from('<Hhhh', d, pos)
            if code >= 0x100:
                local = struct.unpack_from('<H', d, pos + 6)[0] - base
                if 0 <= local < bank.count: out.append([local, code >> 8])
                if code >> 8 == 0xfa: return ('hold',)
            elif code == 1: return ('roa',)
            elif code == 2: return ('end', koc, ix, tp)
            elif code in (3, 4): return ('jmp', tk, ti, tp)
            elif code == 5:
                if depth < 3: walk(tk, ti, tp, depth + 1, out)
            elif code == 6: return ('ret',)
            elif code >= 125: return None
            pos += st
        return None

    anims = []
    for koc, offs in tables.items():
        for i in range(len(offs)):
            frames = []
            ex = walk(koc, i, 1, 0, frames)
            loop = bool(ex) and ex[0] in ('end', 'jmp') and ex[1] == koc and ex[2] == i
            nxt = list(ex[1:]) if ex and ex[0] == 'jmp' and not loop else None
            anims.append(dict(koc=koc, name=KOC_NAMES[koc], index=i, frames=frames, loop=loop, next=nxt))
    return anims


def load(bank_name, bank=None, rebuild=False):
    """Cached animations for one bank (build/skin/<bank>/anims.json)."""
    path = MOD / 'build' / 'skin' / bank_name / 'anims.json'
    if path.exists() and not rebuild:
        return json.loads(path.read_text())['anims']
    bank = bank or SpriteBank.open(MOD / 'build' / 'extracted' / f'{bank_name}.bin')
    anims = parse(bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(anims=anims)))
    return anims


def neighbours(anims):
    """{pattern: {other pattern: count}} for patterns that follow each other directly in some animation
    (consecutive repeats of one pattern are collapsed; looping animations also link last -> first)."""
    nb = {}
    def link(a, b):
        if a == b: return
        nb.setdefault(a, {}); nb[a][b] = nb[a].get(b, 0) + 1
        nb.setdefault(b, {}); nb[b][a] = nb[b].get(a, 0) + 1
    for an in anims:
        seq = [p for p, _ in an['frames']]
        for a, b in zip(seq, seq[1:]): link(a, b)
        if an['loop'] and len(seq) > 2: link(seq[-1], seq[0])
    return nb


if __name__ == '__main__':
    for name in sys.argv[1:]:
        an = load(name, rebuild=True)
        bank = SpriteBank.open(MOD / 'build' / 'extracted' / f'{name}.bin')
        used = {p for a in an for p, _ in a['frames']}
        print(f'{name}: {len(an)} animations, {sum(1 for a in an if a["frames"])} with frames, '
              f'{len(used)}/{bank.count} patterns referenced')
        by = {}
        for a in an: by.setdefault(a['name'], []).append(len(a['frames']))
        print('  ', {k: (len(v), sum(v)) for k, v in by.items()})
