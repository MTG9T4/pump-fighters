"""Assemble the phone/PPSSPP folder: PSP/GAME/pumpcombat/{EBOOT.PBP, resources/SF33RD.AFS (patched)}."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The 3s-psp EBOOT.PBP comes from its own release (github.com/demmis98/3s-psp); put it at tools/psp/EBOOT.PBP.
EBOOT = ROOT / 'psp' / 'EBOOT.PBP'
AFS = ROOT / 'sf3mod' / 'build' / 'SF33RD.AFS'
OUT = ROOT / 'sf3mod' / 'build' / 'phone' / 'PSP' / 'GAME' / 'pumpcombat'

if not AFS.is_file():
    raise SystemExit('build the mod first: python sf3mod/build_mod.py')
if not EBOOT.is_file():
    raise SystemExit('put the 3s-psp EBOOT.PBP at tools/psp/EBOOT.PBP first (from github.com/demmis98/3s-psp releases)')
(OUT / 'resources').mkdir(parents=True, exist_ok=True)
shutil.copy2(EBOOT, OUT / 'EBOOT.PBP')
shutil.copy2(AFS, OUT / 'resources' / 'SF33RD.AFS')
print('phone folder ->', OUT)
