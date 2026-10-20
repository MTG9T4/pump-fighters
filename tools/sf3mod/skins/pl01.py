"""GIGA on Alex (pl01).

* Feet: the shared combat boot was ~1.8x Alex's and hung ~10 px under the floor. Boot size set to the original's,
  ankle pivot at the top of the boot's foot; ankles are put into Alex's boots (their brown leather is only used on
  the boots), knees re-solved when the leg no longer fits.
* Head: the pose model often read Alex's green glove as his face (guards, crouches: 138, 373, 50, 100), so the
  puppet's head sat on his fist. His head is found from his hair / headband instead (the front of the hair blob,
  plus the idle frames' head offset) whenever the read head is more than 9 px from it, in upright frames; the
  shoulders follow when the misread had pulled them away too.
* Effect frames (Flash Chop / Air Stampede swooshes 542-544 and 800-802, thrown shirt 903-916, sleeve 928) were being given a
  puppet; they keep their own drawing now.
"""
from pathlib import Path
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location('_groupB_pl16', Path(__file__).with_name('pl16.py'))
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

CONFIG = dict(foot=0.63)
PIVOTS = {'foot': ((0.3, 0.34), (0.8, 0.85))}

# Body-shaped frames in effect colours: the intro's flesh-coloured morph into Alex (990-999).
KEEP = set(range(990, 1000))
# Swooshes and a torn sleeve in his own colours.
NOT_BODY = set(range(541, 545)) | {928}
# Not Alex: chop swooshes (797-805, 864-868) and the grey-blue cloth / debris pieces (886-912) keep their drawing
EFFECT = [(797, 805), (864, 868), (886, 912)]


def fix_head(p, J, ctx):
    pal = H.top_palette(ctx)
    h = H.hair_front(ctx, p, J['face'], pal, min_area=25)
    if h is None: return J
    est = h + H.head_offset(ctx, pal, J['face'])
    tor = np.asarray(J['hip'], float) - np.asarray(J['sho'], float)
    tl = float(np.linalg.norm(tor)) or 30.0
    # upright frames only (torso within ~40 degrees of vertical, hair above the shoulders): the idle offsets
    # don't hold for flips, rolls and knockdowns
    if tor[1] < 0.75 * tl or est[1] > np.asarray(J['sho'], float)[1] - 2: return J
    # a read head that touches his hair / headband is trusted
    if H.near_palette(ctx, p, J['head'], pal, 8): return J
    if np.linalg.norm(est - np.asarray(J['head'], float)) > 9 and np.linalg.norm(est - np.asarray(J['sho'], float)) < 1.6 * tl:
        J['head'] = est
        # the same misread usually dragged the shoulders onto the glove too: a neck much longer than the idle one
        # brings the top of the torso back under the head (arms follow at the shoulder, elbows half way)
        v0 = neck(ctx)
        if np.linalg.norm(np.asarray(J['sho'], float) - est) > 1.35 * np.linalg.norm(v0):
            d = est + v0 - np.asarray(J['sho'], float)
            J['sho'] = np.asarray(J['sho'], float) + d
            for s in 'lr':
                J[s]['sho'] = np.asarray(J[s]['sho'], float) + d
                J[s]['elb'] = np.asarray(J[s]['elb'], float) + 0.5 * d
    return J


def neck(ctx):
    """Median shoulder-centre minus head over the idle frames."""
    key = ('neck', ctx['bank_name'])
    if key not in H._cache:
        core = ctx.get('core', {})
        v = [np.asarray(core[q]['sho'], float) - np.asarray(core[q]['head'], float)
             for q in ctx['bank'].standing_patterns(8) if q in core and core[q].get('head') is not None]
        H._cache[key] = np.median(v, 0) if v else np.array([0.0, 12.0])
    return H._cache[key]


# Crouch frames read as a heap with an arm in the air (drawn in the crouch idle pose); the jump frame 145 read
# as one arm hanging to the floor (drawn like 146); the shock x-rays (824-825) take the shock pose (823).
BORROW = {50: 60, 51: 60, 145: 146, 824: 823, 825: 823,
          # jump tucks read with the head down at the fist (the body tipped forward): the upright tuck of 101
          85: 101, 86: 101, 87: 101, 98: 101, 99: 101, 100: 101}


def fix_pose(p, J, rec, ctx):
    return H.connect(_fix_pose(p, J, rec, ctx))


def _fix_pose(p, J, rec, ctx):
    if p in BORROW:
        J = H.borrow(ctx, BORROW[p], p, J['face']) or J
    elif p in NOT_BODY or H.not_body(ctx, p, KEEP): return None
    J = fix_head(p, J, ctx)
    l1, l2 = H.leg_lengths(ctx)
    return H.shoe_legs(J, ctx, H.shoe_blobs(ctx, p, J), l1, l2)
