"""pl03 Yun -> wif. Uses the group-A silhouette fit shared from pl02.py (see its docstring)."""
import importlib.util, sys
from pathlib import Path


def _group_a():
    name = 'skins_group_a'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('pl02.py'))
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return sys.modules[name]


A = _group_a()

CONFIG = dict(scale=0.95, head=0.65, head_sink=0.12, limb_w=0.9, fist=0.42, foot=0.49)
PIVOTS = {'pelvis': ((0.5, 0.42), (0.5, 0.85)), 'tail': ((0.35, 0.55), (0.5, 0.9)), 'foot': ((0.28, 0.56), (0.8, 0.85))}
FIX = dict(face=-1, face_right=set(), overrides={}, leg_w=0.42, pivots=PIVOTS)
# Props keep their drawing: Yun's skateboard on its own (911-924)
EFFECT = [(911, 924)]


def fix_pose(p, J, rec, ctx):
    return A.fix(p, J, rec, ctx, FIX)
