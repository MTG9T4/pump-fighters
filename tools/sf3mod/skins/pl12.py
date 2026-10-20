"""pl12 Sean -> chillguy. Uses the group-A silhouette fit shared from pl02.py (see its docstring)."""
import importlib.util, sys
from pathlib import Path


def _group_a():
    name = 'skins_group_a'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('pl02.py'))
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return sys.modules[name]


A = _group_a()

CONFIG = dict(scale=1.05, head=0.57, head_sink=0.08, limb_w=0.9, fist=0.45, foot=0.50)
PIVOTS = {'pelvis': ((0.5, 0.3), (0.5, 0.85)), 'foot': ((0.3, 0.55), (0.8, 0.85))}
FIX = dict(face=-1, face_right=set(), overrides={}, leg_w=0.5, pivots=PIVOTS,
           # small glove / face placeholder drawings repeated under many patterns: mascot forearm + fist, never the
           # original's glove (the core takes them for effects)
           arm_pieces=A.same_as('pl12', [169]), head_pieces=A.same_as('pl12', [0]))
BORROW = {p: 1 for p in FIX['arm_pieces'] | FIX['head_pieces']}
# Props keep their drawing: the ball (736)
EFFECT = [(736, 736)]


def fix_pose(p, J, rec, ctx):
    return A.fix(p, J, rec, ctx, FIX)
