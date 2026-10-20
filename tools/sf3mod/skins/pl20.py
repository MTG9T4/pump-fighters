"""BONK on Remy (pl20).

Remy is slim (long legs, narrow hips, small boots); the shared BONK proportions made a bulky dog with huge boots
whose soles hung well below Remy's. Thinner limbs, smaller boots and fists, and the boot's ankle pivot moved down
to where the shaft meets the foot so the sole lands on Remy's sole line.
"""
import numpy as np
from pathlib import Path
from PIL import Image

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('skins_groupD_lib', str(Path(__file__).resolve().parent / 'pl18.py'))
D = _ilu.module_from_spec(_spec); _spec.loader.exec_module(D)  # group D shared helpers (connect, IK, pieces)

FACE = 1
# Remy's 32x32 portrait icons: BONK's head
ICONS = set([0, 48, 63, 64, 79, 128, 144, 207, 319, 350, 351, 657, 814, 815, 828, 1070, 1071]) | set(range(281, 288)) | \
    set(range(328, 331)) | set(range(381, 384)) | set(range(573, 576)) | set(range(940, 944)) | set(range(1098, 1104))
# Remy burning away (399, 835-845; 397-398 / 842-843 the core already borrows): BONK standing, in the burn's colours.
# 846-847 are the last ash piles (no body shape) and stay effects.
BURN = {399} | set(range(835, 846))
BORROW = {p: 8 for p in ICONS}
BORROW.update({p: 834 for p in BURN})
# Effects the completion pass drew as BONK: the kick crescents (451-452, 459, 607, 1044-1048) and arcs (944-956)
EFFECT = [(451, 452), (459, 459), (607, 607), (944, 956), (1044, 1048)]

CONFIG = dict(limb_w=0.72, foot=0.68, fist=0.62, head=0.66)

PIVOTS = {'foot': ((0.35, 0.5), (0.8, 0.85))}


def sane_ground(p, J, bank_name):
    """Drop the core's floor line when it sits well above the bottom of the original drawing: frames with an
    unusual anchor (e.g. pl18 1004, origin (64, 10)) would otherwise have their feet yanked up onto it."""
    g = J.get('ground')
    if g is None: return J
    f = Path(__file__).resolve().parent.parent / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
    if not f.exists(): return J
    a = np.asarray(Image.open(f).convert('RGBA'))[..., 3] > 0
    ys = np.nonzero(a.any(1))[0]
    if len(ys) and g < ys.max() - 6: J['ground'] = None
    return J


def fix_pose(p, J, rec, ctx):
    if p in ICONS: return D.head_piece('pl20', p, FACE)
    J = D.connect(sane_ground(p, J, 'pl20'))
    if p in BURN: J = D.borrowed_look(D.fit_to_drawing(J, 'pl20', p), ctx, p)
    return J

# loose hand pieces drawn beside Remy during kicks/throws: mascot arm parts, not heads
PIECES = {"arm": [(455, 463), (602, 606), (1040, 1043)]}
