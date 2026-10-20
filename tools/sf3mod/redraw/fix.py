"""Redraw flagged cells on a new fix sheet, and whole sheets again, then cut and stack them for a look.

  python sf3mod/redraw/fix.py <bank> <fixname> [--note "extra prompt note"] [sNNN:c,c ...] [sNNN ...]

`s010:9` means cell 9 of sheet s010 (cells 0-11, row by row); a bare `s056` redraws that whole sheet. The cells go onto
sheet <fixname> (added to plan.json after the main sheets, so its frames win at publish). A note is appended to the
fighter's cast note for this and later runs.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'sf3mod' / 'build' / 'redraw'
args = sys.argv[1:]
note = None
if '--note' in args:
    i = args.index('--note'); note = args[i + 1]; del args[i:i + 2]
bank, fixname, *specs = args
plan = json.loads((OUT / bank / 'plan.json').read_text())
cells, whole = [], []
for sp in specs:
    if ':' in sp:
        sh, cs = sp.split(':')
        cells += [plan[sh][int(c)] for c in cs.split(',') if int(c) < len(plan[sh])]
    else:
        whole.append(sp)
if note:
    cp = ROOT / 'sf3mod' / 'redraw' / 'cast.json'; cast = json.loads(cp.read_text())
    old = cast[bank].get('note', '')
    if note not in old: cast[bank]['note'] = (old + ' ' + note).strip()
    cp.write_text(json.dumps(cast, indent=1))
run = [sys.executable, str(ROOT / 'sf3mod' / 'redraw' / 'run_plan.py'), bank, '--jobs', '4']
if cells:
    for k in range(0, len(cells), 12):
        n = fixname if k == 0 else f'{fixname}_{k // 12}'
        plan[n] = cells[k:k + 12]
    (OUT / bank / 'plan.json').write_text(json.dumps(plan))
names = [n for n in plan if n == fixname or n.startswith(fixname + '_')] + whole
subprocess.run(run + ['--only', ','.join(names), '--redo'], cwd=ROOT)
from PIL import Image
for n in names:
    sd = OUT / bank / 'sheets' / n
    if not (sd / 'codex.png').exists(): print(n, 'no codex sheet'); continue
    a = Image.open(sd / 'src.png').convert('RGB'); b = Image.open(sd / 'codex.png').convert('RGB').resize(a.size)
    o = Image.new('RGB', (a.width, a.height * 2 + 10)); o.paste(a, (0, 0)); o.paste(b, (0, a.height + 10))
    o.thumbnail((1500, 1500)); o.save(sd / 'pair.png'); print(sd / 'pair.png')
