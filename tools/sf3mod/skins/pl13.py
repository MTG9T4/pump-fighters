"""ELON on Urien (pl13). Parody caricature: the art stays as it is; only fit fixes here.

* Feet: the shared sneaker was ~2x Urien's foot and hung ~10 px under the floor; size set to the original's.
* Effect frames were being given a puppet: Aegis Reflector sparkles / mirror (1089-1108), hit sparks (992-1007),
  blood (1056-1058), the suit he tears off in his intro (1217-1231), Gill fading in / out of his win pose
  (1248-1260), the fire column (1301-1331). They keep their own drawing now.
* The shock x-rays (556-557) take the pose of the shock frame (555).
"""
from pathlib import Path
import importlib.util

_spec = importlib.util.spec_from_file_location('_groupB_pl16', Path(__file__).with_name('pl16.py'))
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

CONFIG = dict(foot=0.55)
PIVOTS = {'foot': ((0.3, 0.13), (0.8, 0.85))}

# Body-shaped frames in effect colours: the white flash crouch (1341-1342), the morph into a pose (1436-1439).
KEEP = {1341, 1342} | set(range(1436, 1440))
# The fire-pillar super (1302-1331): flames over his dark silhouette, an effect drawing, not a body pose.
NOT_BODY = set(range(1302, 1332))
# Effects the completion pass drew as ELON: the red burst (995) and the fire pillar's top with no body in it (1329-1330)
EFFECT = [(995, 995), (1329, 1330)]


# Shock x-rays take the shock pose they flicker with.
BORROW = {556: 555, 557: 555}


def fix_pose(p, J, rec, ctx):
    return H.connect(_fix_pose(p, J, rec, ctx))


def _fix_pose(p, J, rec, ctx):
    if p in BORROW: return H.borrow(ctx, BORROW[p], p, J['face']) or J
    if p in NOT_BODY or H.not_body(ctx, p, KEEP): return None
    return J
