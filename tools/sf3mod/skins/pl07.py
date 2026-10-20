"""pl07 Ibuki -> mew. Uses the group-A silhouette fit shared from pl02.py (see its docstring)."""
import importlib.util, sys
from pathlib import Path


def _group_a():
    name = 'skins_group_a'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('pl02.py'))
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return sys.modules[name]


A = _group_a()

CONFIG = dict(scale=0.78, head=0.62, head_sink=0.14, limb_w=0.8, fist=0.56, foot=0.45)
PIVOTS = {'pelvis': ((0.5, 0.3), (0.5, 0.85)), 'tail': ((0.55, 0.92), (0.5, 0.9)), 'foot': ((0.25, 0.52), (0.8, 0.85))}
FIX = dict(face=-1, face_right=set(), overrides={}, leg_w=0.5, pivots=PIVOTS)
# Props keep their drawing: Don-chan the tanuki and the yellow bundles of the win pose (1685-1730), the smoke burst
# (1775-1776)
EFFECT = [(1685, 1730), (1775, 1776)]


def fix_pose(p, J, rec, ctx):
    return A.fix(p, J, rec, ctx, FIX)

# Ibuki's ponytail is its own sprite object (1201-1229, 1441-1560): MEW has no ponytail, so it is hidden.
HIDE = set(range(1201, 1230)) | set(range(1441, 1561))
