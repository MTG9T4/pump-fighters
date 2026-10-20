"""Build every fighter's skin as soon as its pose read exists, a few at a time.

    python sf3mod/skin_all.py [pl11 pl04 ...] [--jobs 2]
Each build runs `python sf3mod/skin.py <bank> --optimal` and logs to build/skin/<bank>/build.log.
The checker, audit and encoder reuse one prepared pose set and the same rendered pixels.
"""
import subprocess, sys, time
from pathlib import Path

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD))
from skin import SKINS

args = [a for a in sys.argv[1:] if not a.startswith('--')]
jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 2
if not 1 <= jobs <= 2:
    raise SystemExit('Use --jobs 1 or 2 (larger builds exhaust memory).')
args = [a for a in args if not a.isdigit()]
todo = args or list(SKINS)
running = {}
failed = []
missing = [b for b in todo if not (MOD / 'build' / 'skin' / b / 'pose.json').exists()]
if missing: raise SystemExit(f'Missing pose data: {missing}')
while todo or running:
    for b, p in list(running.items()):
        if p.poll() is not None:
            print(b, 'done' if p.returncode == 0 else f'FAILED ({p.returncode})', flush=True)
            if p.returncode: failed.append(b)
            del running[b]
    for b in list(todo):
        if len(running) >= jobs: break
        if (MOD / 'build' / 'skin' / b / 'pose.json').exists():
            log = open(MOD / 'build' / 'skin' / b / 'build.log', 'w')
            running[b] = subprocess.Popen([sys.executable, str(MOD / 'skin.py'), b, '--optimal'],
                                          stdout=log, stderr=subprocess.STDOUT, cwd=MOD.parent)
            todo.remove(b); print(b, 'started', flush=True)
    time.sleep(5)
if failed: raise SystemExit(f'Skin builds failed: {failed}')
print('all skins built', flush=True)
