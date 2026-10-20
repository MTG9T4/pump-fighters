"""PIPPIN on Elena (pl08).

* Feet: the shared hoof hung ~10 px under the floor; size set to Elena's bare foot, ankle pivot at the fetlock.
* Legs: Elena is barefoot and long-legged, and the pose model often put both ankles on one foot or lost the front
  foot (walk 13-60, stance 0-23). Her gold anklets are easy to see: each ankle goes onto its anklet (wrist bands
  of the same gold are told apart by the read wrists and by sitting above the hips), knees kept when they still
  fit, else re-solved.
* Effect frames (music notes 861-870, kick trails 1500-1521, the food icons of her healing super) were being given
  a puppet; they keep their own drawing now.
* The electrocuted x-ray frames (1537-1538) take the pose of the shock frame (1536).
* Her win dance (939-1010) is drawn facing right: the puppet faces right there too.
"""
from pathlib import Path
import importlib.util
import numpy as np
import cv2

_spec = importlib.util.spec_from_file_location('_groupB_pl16', Path(__file__).with_name('pl16.py'))
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

CONFIG = dict(foot=0.72)
PIVOTS = {'foot': ((0.3, 0.43), (0.8, 0.85))}

# Body-shaped frames in effect colours: the super's silhouette afterimages.
# Shock x-rays take the pose of the frame they flicker with.
KEEP = {606, 607}
# Props keep their drawing: the food thought-bubble icons (1514-1519)
EFFECT = [(1514, 1519)]
# Dotted-outline after-images of her lunge (567-570) and crouch dash (571-584) found no pose of their own: they
# take the lunge (559) / the super's crouching silhouette (606); the core draws them as an outline like the original.
BORROW = {1537: 1536, 1538: 1536, **{p: 559 for p in range(567, 571)}, **{p: 606 for p in range(571, 585)}}
TURNED = set(range(939, 1011))
# The win dance stands on both feet, legs side by side (one hides the other): a leg the reader put up on the
# hands (990-1010 read as a kick) stands next to the other one.
STANDING_TOGETHER = set(range(960, 1011))


def legs_together(J):
    la, ra = np.asarray(J['l']['ank'], float), np.asarray(J['r']['ank'], float)
    low, high = ('l', 'r') if la[1] >= ra[1] else ('r', 'l')
    if abs(la[1] - ra[1]) < 8: return J
    d = np.asarray(J[high]['hip'], float) - np.asarray(J[low]['hip'], float)
    J[high]['kne'] = np.asarray(J[low]['kne'], float) + d * 0.5
    J[high]['ank'] = np.asarray(J[low]['ank'], float) + np.array([-2.0 * J['face'], 0.0])
    return J


def gold_mask(a):
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    m = (al > 0) & (r >= 140) & (b <= 10) & (g >= 90)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def anklets(ctx, p, J):
    a = H.frame_rgba(ctx, p)
    if a is None: return []
    l1, l2 = H.leg_lengths(ctx)
    hip = np.asarray(J['hip'], float); sho = np.asarray(J['sho'], float)
    down = (hip - sho) / (np.linalg.norm(hip - sho) or 1)
    rec = ctx['pose'][str(p)]
    kp = np.array(rec.get('kp', [[0, 0]] * 17), float); sc = np.array(rec.get('score', [0] * 17), float)
    anks = [kp[i] for i in (15, 16) if sc[i] >= 0.4]
    wris = [kp[i] for i in (9, 10) if sc[i] >= 0.4] + [np.asarray(J[s]['wri'], float) for s in 'lr']
    out = []
    for b in H.blobs(gold_mask(a), 4)[:6]:
        c = b['c']
        d = np.linalg.norm(c - hip)
        if d > 1.9 * (l1 + l2) or d < 0.8 * l1: continue
        near_ank = any(np.linalg.norm(c - k) < 12 for k in anks)
        if any(np.linalg.norm(c - w) < 8 for w in wris) and not near_ank: continue
        if np.dot(c - hip, down) < 0.5 * l1 and not near_ank: continue
        # two rings of one anklet can come out as two blobs: merge with a chosen one closer than 7 px
        if any(np.linalg.norm(c - o['c']) < 7 for o in out): continue
        out.append(b)
        if len(out) == 2: break
    return out


def fix_pose(p, J, rec, ctx):
    return H.connect(_fix_pose(p, J, rec, ctx))


def _fix_pose(p, J, rec, ctx):
    if p in BORROW:
        J = H.borrow(ctx, BORROW[p], p, J['face']) or J
    elif H.not_body(ctx, p, KEEP): return None
    if p in TURNED: J['face'] = 1  # drawn facing right (the core's turned-run test now agrees; never flip twice)
    if p in STANDING_TOGETHER: J = legs_together(J)
    l1, l2 = H.leg_lengths(ctx)
    found = anklets(ctx, p, J)
    cands = [(b['c'], b['c']) for b in found]
    for s, (ank, _) in H.assign(J, cands).items():
        L = J[s]
        L['ank'] = np.asarray(ank, float)
        kne = np.asarray(L['kne'], float); h = np.asarray(L['hip'], float)
        sl = float(np.linalg.norm(kne - L['ank'])); tl = float(np.linalg.norm(kne - h))
        if not (0.35 * l2 <= sl <= 1.8 * l2) or tl > 1.8 * l1:
            d = L['ank'] - h; n = np.array([-d[1], d[0]])
            off = float(np.dot(kne - h, n))
            bend = n * np.sign(off) if abs(off) > 1e-3 else np.array([J['face'] * 1.0, -0.3])
            L['kne'] = H.ik_knee(h, L['ank'], l1, l2, bend)
    return J

# Loose pieces of the original fighter drawn as separate objects (busts, arms, blurs): redrawn as the mascot.
REDRAW_EXTRA = [613]
