"""FWOG on Twelve (pl19).

Twelve is a shapeshifter; nothing of him may show, so every form gets the frog:
  * body forms (stance, walk, normals, glides, ceiling hang, blob morph, wing form) take the frog puppet
  * dissolve / teleport frames (body breaking into stripes and particles: 1021-1034, 1047-1059, and any posed frame
    with almost no solid interior) are a frog pose drawn as an outline (sparser: dithered too), so the frog fades
  * the small portrait icons (32x24) and the hanging cup form (896-901): FWOG's head
  * the loose tentacle 717: a frog arm; the coil 770 and the wing frames 1106 / 1109: the neighbouring frog pose
Stretch arms (672-703, 904-955): the arm bones follow the stretched limb (paste_bone stretches them).
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('skins_groupD_lib', str(Path(__file__).resolve().parent / 'pl18.py'))
D = _ilu.module_from_spec(_spec); _spec.loader.exec_module(D)  # group D shared helpers (connect, IK, pieces)

MOD = Path(__file__).resolve().parent.parent

FACE = 1
ICONS = set([0, 47, 48, 55, 109, 110, 111, 126, 127, 191, 276, 277, 278, 287, 349, 350, 351, 490, 547, 548, 559, 606, 607,
             618, 644, 670, 671, 847, 894, 895]) | set(range(15, 22)) | set(range(28, 32)) | set(range(35, 38)) | \
    set(range(60, 64)) | set(range(203, 208)) | set(range(315, 322)) | set(range(407, 416)) | set(range(442, 448)) | \
    set(range(1035, 1040)) | set(range(896, 902))
ARMS = {717}
DISSOLVE = set(range(1021, 1035)) | set(range(1047, 1060))
BORROW = {p: 1 for p in ICONS | ARMS}
BORROW.update({p: 1020 for p in range(1021, 1035)})
BORROW.update({p: 1046 for p in range(1047, 1060)})
BORROW.update({770: 771, 1106: 1105, 1109: 1108})
# Effects the completion pass drew as FWOG heads / arms: needles (257-258, 271-272, 956-992), spin rings (758-761)
EFFECT = [(257, 258), (271, 272), (758, 761), (956, 992)]


# Ankle pivot low on the webbed foot (default sits at its top edge and sank the toes into the floor)
PIVOTS = {'foot': ((0.28, 0.4), (0.8, 0.85))}

_solid = {}


def solid(p):
    if p not in _solid:
        f = MOD / 'build' / 'skin' / 'pl19' / 'frames' / f'{p:04d}.png'
        r = 1.0
        if f.exists():
            a = (np.asarray(Image.open(f).convert('RGBA'))[..., 3] > 0).astype(np.uint8)
            n = a.sum()
            if n >= 20: r = cv2.erode(a, np.ones((3, 3), np.uint8)).sum() / n
        _solid[p] = r
    return _solid[p]


STRETCH = set(range(672, 704)) | set(range(904, 956))
_arm = {}


def arm_len(ctx):
    """Idle arm length (shoulder -> elbow -> wrist) from the core's idle poses."""
    if 'a' not in _arm:
        ls = []
        for q in ctx['bank'].standing_patterns(8):
            Jq = ctx['core'].get(q)
            if Jq is None: continue
            for sd in 'lr':
                L = Jq[sd]
                if L['sho'] is None or L['elb'] is None or L['wri'] is None: continue
                ls.append(np.linalg.norm(L['elb'] - L['sho']) + np.linalg.norm(L['wri'] - L['elb']))
        _arm['a'] = float(np.median(ls)) if ls else 30.0
    return _arm['a']


def stretch_arms(p, J, ctx):
    """Twelve's stretching limbs: the frog's arm reaches all the way to the stretched hand (paste_bone stretches the
    sleeve along it). Tips = the drawing's points farthest from the shoulders, beyond 1.4 idle arm lengths."""
    m = D.alpha('pl19', p)
    if m is None: return J
    A = arm_len(ctx)
    sho = np.asarray(J['sho'], float)
    ys, xs = np.nonzero(m); pts = np.stack([xs, ys], 1).astype(float)
    d = np.linalg.norm(pts - sho, axis=1)
    for sd in 'lr':  # a stretched hand, never a foot
        d[np.linalg.norm(pts - np.asarray(J[sd]['ank'], float), axis=1) < 0.7 * A] = 0
    if d.max() < 1.4 * A: return J
    tips = [pts[d.argmax()]]
    u0 = (tips[0] - sho) / (np.linalg.norm(tips[0] - sho) or 1)
    dirs = (pts - sho) / np.maximum(d, 1e-6)[:, None]
    other = (dirs @ u0 < 0.5) & (d > 1.4 * A)
    if other.any():
        i = np.flatnonzero(other)[d[other].argmax()]; tips.append(pts[i])
    used = set()
    for tip in tips:
        side = min((sd for sd in 'lr' if sd not in used),
                   key=lambda sd: np.linalg.norm(np.asarray(J[sd]['wri'], float) - tip))
        used.add(side)
        L = J[side]
        root = np.asarray(L['sho'], float)
        mid = root + 0.5 * (tip - root)
        k = np.argmin(np.linalg.norm(pts - mid, axis=1))
        elb = pts[k] if np.linalg.norm(pts[k] - mid) < 25 else mid
        L['elb'], L['wri'] = elb, tip - (tip - elb) / (np.linalg.norm(tip - elb) or 1) * 3
    return J


def fix_pose(p, J, rec, ctx):
    if p in ICONS: return D.head_piece('pl19', p, FACE)
    if p in ARMS: return D.arm_piece('pl19', p, FACE, thick=7)
    if p in DISSOLVE and rec['size'][0] < 30:
        return None  # the teleport down to a thin vertical beam (1031-1034, 1055-1059): the core draws it as a frog arm
    J = D.connect(sane_ground(p, J, 'pl19'))
    sol = solid(p)
    if p in (770, 1106, 1109): J['_borrowed'] = BORROW[p]
    if p in STRETCH:
        # thin looping limbs fool the core's after-image test (filled area >> drawn area): keep the frog's colours
        J = stretch_arms(p, J, ctx); J['_treat'] = None
    elif p in DISSOLVE or sol < 0.45:
        # fading frog: every other pixel (a checkerboard stays one 8-connected body)
        J['_treat'] = dict(tone=False, dither=('check', 0), outline=False)
        J['_src'] = ('pl19', p)
        J['ground'] = None
        J['_borrowed'] = J.get('_borrowed', 'fade')  # out of the core's pose library
    return J


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

# Twelve's stretched-arm tendrils: frog arm parts, not heads
PIECES = {"arm": [(704, 726)]}
