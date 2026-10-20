"""JEAN PHIL on Dudley (pl04).

* Dudley is a heavyweight; the slim suited puppet covered ~75% of his silhouette. The whole puppet is drawn 15%
  wider (scale; bones still follow the skeleton), with fuller limbs; head and shoes scaled back so they stay the
  original's size, fists a little bigger (boxer).
* Feet: the shared shoe was ~2x Dudley's and hung ~10 px under the floor; ankles are put into Dudley's shoes
  (their dark red-grey leather is only used on the shoes), knees re-solved when the leg no longer fits.
* His effect frames (Rolling Thunder / Corkscrew spirals 989-1074, sparks, thrown jacket 697, rose 1087-1089) were
  being given a puppet; they keep their own drawing now.
* The electrocuted x-ray frames (66-67) take the pose of the shock frame it flickers with (65).
* Frames drawn facing right (turn 22-24, the turned jab / lunge 529-538) face right.
"""
from pathlib import Path
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location('_groupB_pl16', Path(__file__).with_name('pl16.py'))
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

CONFIG = dict(scale=1.15, head=0.55, foot=0.52, limb_w=1.0, fist=0.6)
PIVOTS = {'foot': ((0.3, 0.05), (0.8, 0.85))}

# Body-shaped frames in effect colours: the bronze-statue win pose (1113-1122).
KEEP = set(range(1113, 1123))
# Effects in his own colours: the thrown jacket (695-700, 729), sparkles (893-896), the car and its wheel (1086-1099).
NOT_BODY = set(range(695, 701)) | {729} | set(range(892, 897)) | set(range(1086, 1100))
# Props and effects keep their drawing (the completion pass drew them as GIGA): the thrown jacket (690-700, 868-869),
# the car, its wheel, smoke and sparkles (701-729, 732, 1086-1112), sparkle lines (790-793, 821-828), the tornado
# (981-1061). The butler (817) is another character.
EFFECT = [(690, 700), (701, 729), (732, 732), (790, 793), (821, 828), (868, 869), (981, 1061), (1086, 1112)]
OTHER = [(815, 817)]  # the butler (815-816 in his lift cage)
TURNED = {22, 23, 24} | set(range(529, 539))


# Electrocuted x-rays (66, 67; 67 read as a body lying down) are drawn in the shock pose of 65.
BORROW = {66: 65, 67: 65, 673: 1, 674: 1}


# Loose glove + face pieces of a throw (673-674): drawn as the mascot's arm over the piece.
ARM_PIECES = {673, 674}


def arm_piece(p, J, ctx):
    import skin
    a = H.frame_rgba(ctx, p)
    m = a[..., 3] > 0
    org = np.asarray(ctx['pose'][str(p)]['origin'], float)
    Jp = skin._piece_J('arm', m, J['face'], anchor=-org)
    Jp['_src'] = (ctx['bank_name'], p)
    return Jp


def fix_pose(p, J, rec, ctx):
    return H.connect(_fix_pose(p, J, rec, ctx))


def _fix_pose(p, J, rec, ctx):
    if p in ARM_PIECES: return arm_piece(p, J, ctx)
    if p in BORROW:
        J = H.borrow(ctx, BORROW[p], p, J['face']) or J
    elif p in NOT_BODY or H.not_body(ctx, p, KEEP): return None
    if p in TURNED: J['face'] = 1  # drawn facing right (the core's turned-run test now agrees; never flip twice)
    l1, l2 = H.leg_lengths(ctx)
    return H.shoe_legs(J, ctx, H.shoe_blobs(ctx, p, J), l1, l2)
