"""PNUT on Q (pl18).

Q wears a long trench coat, so the pose model rarely sees his legs: in crouches, sits and knockdowns it reads the
coat's hem as legs and the puppet kicks where Q squats (537). What we can see are his shoes (neutral dark grey),
so the legs are rebuilt from them: ankles on the shoe blobs, knees by two-bone IK bending forward.

This file also holds the leg helpers shared by the other group-D hooks (pl06 / pl19 / pl20 load it with lib()).
"""
from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
MOD = HERE.parent

# Smaller widths, head and feet than the shared default: Q is tall and thin and PNUT was much wider than him.
CONFIG = dict(head=0.62, head_sink=0.12, limb_w=0.66, fist=0.62, foot=0.72, scale=0.92, tail=0.66)

# Ankle joint sits low on the cowboy boot (default pivot is the top of the shaft, which sank the sole ~14 px
# under the floor).
# Tail hung from its middle, a little smaller: hung from its tip it reached ~20 px under the floor in sits (13-19,
# 520-559); now its top half tucks behind the back.
PIVOTS = {'foot': ((0.3, 0.6), (0.8, 0.85)), 'tail': ((0.5, 0.42), (0.3, 0.9))}


# ---------------------------------------------------------------- shared helpers (group D)

_cache = {}


def frame_rgba(ctx, p):
    """The original frame as an int RGBA array (sprite px), cached."""
    key = (ctx['bank_name'], p)
    if key not in _cache:
        f = MOD / 'build' / 'skin' / ctx['bank_name'] / 'frames' / f'{p:04d}.png'
        _cache[key] = np.asarray(Image.open(f).convert('RGBA')).astype(int) if f.exists() else None
    return _cache[key]


def blobs(mask, min_area):
    """Connected blobs of a bool mask: list of dict(area, c=centroid, box=(x0,y0,x1,y1), pts), largest first."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area: continue
        ys, xs = np.nonzero(lab == i)
        out.append(dict(area=int(a), c=np.array(cen[i]), box=(x, y, x + w - 1, y + h - 1), xs=xs, ys=ys))
    out.sort(key=lambda b: -b['area'])
    return out


def ik_knee(hip, ank, l1, l2, bend):
    """Knee for a two-bone leg hip->ank with bone lengths l1, l2, on the side of the hip->ank line that `bend`
    (a direction) points to. Out of reach -> knee on the straight line."""
    d = ank - hip; L = float(np.linalg.norm(d))
    if L < 1e-3: return hip + bend / (np.linalg.norm(bend) or 1) * l1
    u = d / L
    if L >= l1 + l2 - 1e-3:
        return hip + u * L * l1 / (l1 + l2)
    L = max(L, abs(l1 - l2) + 1e-3)
    a = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    n = np.array([-u[1], u[0]])
    if np.dot(n, bend) < 0: n = -n
    return hip + u * a + n * h


def leg_lengths(ctx):
    """Typical (thigh, shin) of the original from confidently read frames (median of hip-knee / knee-ankle)."""
    key = ('legs', ctx['bank_name'])
    if key in _cache: return _cache[key]
    th, sh = [], []
    for rec in ctx['pose'].values():
        if 'kp' not in rec: continue
        kp = np.array(rec['kp']); sc = np.array(rec['score'])
        for h, k, a in ((11, 13, 15), (12, 14, 16)):
            if min(sc[h], sc[k], sc[a]) < 0.6: continue
            th.append(np.linalg.norm(kp[k] - kp[h])); sh.append(np.linalg.norm(kp[a] - kp[k]))
    _cache[key] = (float(np.percentile(th, 60)), float(np.percentile(sh, 60)))
    return _cache[key]


def standing_lengths(ctx, place_ankles, slack=1.0):
    """(thigh, shin) whose sum is the idle stance's hip->ankle reach after this fighter's ankle placement, so idle
    legs come out straight and only real crouches bend. Thigh/shin ratio from leg_lengths()."""
    key = ('stand', ctx['bank_name'])
    if key in _cache: return _cache[key]
    import skin
    th, sh = leg_lengths(ctx)
    reach = []
    for p in ctx['bank'].standing_patterns(8):
        rec = ctx['pose'][str(p)]
        J = skin.copy_J(ctx['core'][p]) if 'core' in ctx and p in ctx['core'] else skin.pose_frame(rec)
        if J is None: continue
        J = place_ankles(p, skin.fill_limbs(J, ctx['G']), rec)
        reach += [np.linalg.norm(np.asarray(J[s]['ank'], float) - np.asarray(J[s]['hip'], float)) for s in 'lr']
    T = float(np.median(reach)) * slack if reach else th + sh
    _cache[key] = (T * th / (th + sh), T * sh / (th + sh))
    return _cache[key]


def solve_knees(J, ctx, rec, keep_score=0.55, lengths=None):
    """Re-derive both knees by IK. The current knee (core-cleaned) only chooses which way the leg bends, and only
    when the detector was confident about knees in this frame; otherwise the knee bends forward (toward the facing
    side) and slightly up."""
    l1, l2 = lengths or leg_lengths(ctx)
    sc = np.array(rec.get('score', [0.0] * 17))
    fwd = np.array([J['face'] * 1.0, -0.35])
    for side, ki in (('l', 13), ('r', 14)):
        L = J[side]
        hip, ank = np.asarray(L['hip'], float), np.asarray(L['ank'], float)
        bend = fwd
        if sc[ki] >= keep_score and L.get('kne') is not None:
            d = ank - hip; n = np.array([-d[1], d[0]])
            off = np.dot(np.asarray(L['kne'], float) - hip, n) / (np.linalg.norm(n) or 1)
            if abs(off) > 2: bend = n * np.sign(off)
        reach = np.linalg.norm(ank - hip)
        k = reach / (l1 + l2) if reach > l1 + l2 else 1.0  # kicks: stretch the bones rather than leave a gap
        L['kne'] = ik_knee(hip, ank, l1 * k, l2 * k, bend)
    return J


def assign_feet(J, targets, keep=None):
    """Move ankles onto target points (list of np arrays). Two targets: matched to l/r by least total distance.
    One target: goes to the nearer ankle, the other ankle stays (or joins it when `keep` says it is unreliable)."""
    la, ra = np.asarray(J['l']['ank'], float), np.asarray(J['r']['ank'], float)
    if len(targets) >= 2:
        a, b = targets[:2]
        if np.linalg.norm(la - a) + np.linalg.norm(ra - b) <= np.linalg.norm(la - b) + np.linalg.norm(ra - a):
            J['l']['ank'], J['r']['ank'] = a, b
        else:
            J['l']['ank'], J['r']['ank'] = b, a
    elif len(targets) == 1:
        t = targets[0]
        near = 'l' if np.linalg.norm(la - t) <= np.linalg.norm(ra - t) else 'r'
        other = 'r' if near == 'l' else 'l'
        J[near]['ank'] = t
        if keep is not None and not keep.get(other, True):
            J[other]['ank'] = t + np.array([J['face'] * 3.0, -1.0])
    return J


def foot_point(b, hip, band=10, lift=4):
    """Ankle target inside a foot blob: the blob end farthest from the hip (a shoe merged with a trouser leg
    keeps only its far end), centre of that band, lifted toward the hip."""
    pts = np.stack([b['xs'], b['ys']], 1).astype(float)
    d = np.linalg.norm(pts - hip, axis=1)
    c = pts[d >= d.max() - band].mean(0)
    u = hip - c; n = np.linalg.norm(u) or 1
    return c + u / n * lift


def floor_feet(J, mask, ctx, ok, spread=(-3, 8)):
    """Feet the original hides (coat, effects): an ankle outside the silhouette goes to the floor line under the
    hip (back foot a little behind, front foot a little ahead)."""
    ys = np.nonzero(mask.any(1))[0]
    if not len(ys): return J
    l1, l2 = leg_lengths(ctx)
    floor = ys.max() - 0.12 * (l1 + l2)
    hip = np.asarray(J['hip'], float)
    if floor - hip[1] < 4: return J  # hip at/below the floor line: not a stance, leave it
    for side, dx in zip(('far', 'near'), spread):
        s = ('r' if J['face'] > 0 else 'l') if side == 'far' else ('l' if J['face'] > 0 else 'r')
        if ok.get(s, True): continue
        J[s]['ank'] = np.array([hip[0] + J['face'] * dx, floor])
    return J


def inside(mask, pt, pad=3):
    if mask is None: return True
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = mask.shape
    x0, x1, y0, y1 = max(0, x - pad), min(w, x + pad + 1), max(0, y - pad), min(h, y + pad + 1)
    return x0 < x1 and y0 < y1 and mask[y0:y1, x0:x1].any()


def connect(J, up=0.08, down=0.3, hip_up=0.15, hip_down=0.15):
    """Keep limb roots on the body along the spine: each shoulder no more than `up` torso lengths above the
    shoulder centre (or `down` below it), each hip within hip_up / hip_down of the hip centre. The core's compose()
    pulls roots in across the body; this handles the along-spine case (a stretched arm whose shoulder is read high
    above the torso, pl19 937-939; a hip read far down a kicking leg)."""
    sho, hip = np.asarray(J['sho'], float), np.asarray(J['hip'], float)
    t = hip - sho; tl = float(np.linalg.norm(t))
    if tl < 2: return J
    ey = t / tl
    for side in 'lr':
        L = J[side]
        for key, c, lo, hi in (('sho', sho, -up, down), ('hip', hip, -hip_up, hip_down)):
            if L.get(key) is None: continue
            q = np.asarray(L[key], float); v = (q - c) @ ey
            vv = float(np.clip(v, lo * tl, hi * tl))
            if vv != v: L[key] = q + ey * (vv - v)
    return J


# ---------------------------------------------------------------- loose pieces (group D shared)

FAR = np.array([-4000.0, -4000.0])  # parts parked here are off the canvas and never drawn


def alpha(bank_name, p):
    f = MOD / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
    return (np.asarray(Image.open(f).convert('RGBA'))[..., 3] > 0) if f.exists() else None


def main_blob(m, grow=2):
    """The largest connected blob of a mask (after closing small gaps), as an erase mask."""
    g = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=grow)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    if n < 2: return m
    i = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
    return (lab == i) & m


def head_piece(bank_name, p, face):
    """A head-only drawing (portrait icon, a head poking out of a cocoon) -> the mascot head over it."""
    import skin
    m = alpha(bank_name, p)
    return skin._piece_J('head', main_blob(m), face)


def arm_piece(bank_name, p, face, anchor=None, thick=None):
    import skin
    m = alpha(bank_name, p)
    J = skin._piece_J('arm', main_blob(m), face, anchor=anchor)
    if thick is not None: J['_piece']['thick'] = min(J['_piece']['thick'], thick)
    return J


def move(J, d):
    """Translate every joint of J by d (sprite px)."""
    d = np.asarray(d, float)
    for k, v in list(J.items()):
        if isinstance(v, dict): J[k] = {kk: (None if vv is None else np.asarray(vv, float) + d) for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.shape == (2,): J[k] = v + d
    if J.get('ground') is not None: J['ground'] = J['ground'] + d[1]
    return J


def fit_to_drawing(J, bank_name, p):
    """Stand a borrowed pose on the drawing: hips over its centre column, lowest ankle on its bottom row."""
    m = alpha(bank_name, p)
    ys, xs = np.nonzero(m)
    low = max(np.asarray(J[s]['ank'], float)[1] for s in 'lr')
    return move(J, (xs.mean() - np.asarray(J['hip'], float)[0], ys.max() - 4 - low))


def leg_only(face, hip, kne, ank, near=None):
    """A J that draws one leg (thigh, shin, foot) and nothing else: the body, the other leg and both arms are
    parked off the canvas. For loose leg drawings (pl18 1259-1267)."""
    near = near or ('r' if face > 0 else 'l')
    far = 'l' if near == 'r' else 'r'
    off = {k: FAR.copy() for k in ('sho', 'elb', 'wri', 'hip', 'kne', 'ank')}
    legd = dict(sho=FAR.copy(), elb=FAR.copy(), wri=FAR.copy(), hip=np.asarray(hip, float),
                kne=np.asarray(kne, float), ank=np.asarray(ank, float))
    # '_treat': None keeps the mascot's own colours (the core would re-tone a frame unlike the idle drawing)
    return {'sho': FAR.copy(), 'hip': FAR.copy(), 'head': FAR.copy(), 'face': face, near: legd, far: off,
            'ground': None, 'plant': False, 'near': near, '_treat': None, '_borrowed': 'leg piece'}


def leg_from_drawing(m, ank):
    """hip = the far end of the drawing from the ankle, knee = the drawing's point farthest off the hip-ankle line."""
    ys, xs = np.nonzero(m); pts = np.stack([xs, ys], 1).astype(float)
    d = np.linalg.norm(pts - ank, axis=1)
    hip = pts[d >= d.max() - 6].mean(0)
    v = ank - hip; L = np.linalg.norm(v) or 1.0; n = np.array([-v[1], v[0]]) / L
    t = (pts - hip) @ v / L**2
    mid = (t > 0.25) & (t < 0.75)
    off = (pts - hip) @ n
    if mid.any():
        k = np.argmax(np.abs(off) * mid)
        kne = hip + v * t[k] + n * off[k] * 0.6
    else:
        kne = (hip + ank) / 2
    return hip, kne


def borrowed_look(J, ctx, p):
    """Keep the original frame's look (burn tone, after-image outline, dither) on a mascot pose."""
    import skin
    m = alpha(ctx['bank_name'], p)
    tr = skin._treatment(ctx['bank_name'], ctx['bank'], p, m) if m is not None else None
    J['_treat'] = tr or dict(tone=True, dither=None, outline=False)
    J['_src'] = (ctx['bank_name'], p)
    J['_borrowed'] = J.get('_borrowed', 'hook')  # out of the core's pose library
    return J


def lib():
    return __import__('sys').modules[__name__]


# ---------------------------------------------------------------- Q / PNUT

FACE = -1          # every Q frame is drawn facing left
TURNED = set()     # patterns where Q really faces right (none found)


def shoe_mask(a):
    """Q's shoes: neutral dark greys (82,82,82 / 65,65,65), grown into the bluish shading right next to them."""
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    strict = (al > 0) & (np.abs(r - b) <= 10) & (np.abs(r - g) <= 10) & (r >= 50) & (r <= 90)
    strict = cv2.morphologyEx(strict.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    broad = (al > 0) & (np.abs(r - g) <= 10) & (b - r >= -10) & (b - r <= 18) & (r >= 50) & (r <= 120)
    return broad & (cv2.dilate(strict, np.ones((3, 3), np.uint8), iterations=2) > 0)


def shoes(ctx, p, J, rec):
    """Ankle targets from Q's shoe blobs (far end of each blob from the hip). A blob above the knees only counts
    when the pose model also put an ankle there (kicks); otherwise it is a glove / face shadow."""
    a = frame_rgba(ctx, p)
    if a is None: return []
    l1, l2 = leg_lengths(ctx)
    hip = np.asarray(J['hip'], float)
    sc = np.array(rec.get('score', [0.0] * 17))
    anks = [np.asarray(J[s]['ank'], float) for s, i in (('l', 15), ('r', 16)) if sc[i] >= 0.5]
    m = cv2.morphologyEx(shoe_mask(a).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    out = []; first = None
    for b in blobs(m, 10)[:5]:
        if first is not None and b['area'] < max(20, 0.3 * first): break
        if np.linalg.norm(b['c'] - hip) > 1.7 * (l1 + l2): continue
        if np.linalg.norm(b['c'] - np.asarray(J['head'], float)) < 16: continue  # Q's grey face
        if b['c'][1] < hip[1] + 0.35 * l1 and not any(np.linalg.norm(b['c'] - k) < 14 for k in anks): continue
        out.append(foot_point(b, hip)); first = first or b['area']
        if len(out) == 2: break
    return out


# Not Q's body: other characters running away (1130-1177, the core lists them as other characters) and the super's
# energy frames (1308-1425, the core borrows poses for the body ones). The hook leaves them to the core.
NOT_BODY = set(range(1130, 1178)) | set(range(1308, 1426))
# Q's portrait / head icons (48x48 and smaller, unused dummies with their hex number): PNUT's head over them
ICONS = set([0, 783, 821, 1034, 1071, 142, 143, 176, 177, 213]) | set(range(21, 32)) | set(range(248, 256)) | \
    set(range(896, 901)) | set(range(980, 986)) | set(range(1224, 1229)) | set(range(1243, 1246))
# Q's loose leg (1259-1267): one PNUT leg along it
LEGS = set(range(1259, 1268))
# 781-782: Q's arms + hat as loose pieces of the 779-780 pose: PNUT in 780's pose
BORROW = {p: 1 for p in ICONS | LEGS}
BORROW.update({781: 780, 782: 780})
# 1412-1418: Q crumbling away at the end of the super: PNUT in the last standing pose (1411), dithered in the ash colours
FADE = set(range(1412, 1419))
BORROW.update({p: 1411 for p in FADE})


def place_ankles(p, J, rec, ctx):
    J['face'] = -FACE if p in TURNED else FACE
    a = frame_rgba(ctx, p)
    if a is None: return J
    mask = a[..., 3] > 0
    sc = np.array(rec.get('score', [0.0] * 17))
    l1, l2 = leg_lengths(ctx)
    floor = np.nonzero(mask.any(1))[0].max()
    crouch = floor - J['hip'][1] < 0.7 * (l1 + l2)   # hips low over the floor: squat / sit / kneel
    targets = shoes(ctx, p, J, rec)
    ok = {s: (not crouch) and sc[15 if s == 'l' else 16] >= 0.6 and inside(mask, J[s]['ank'], 2) for s in 'lr'}
    J = assign_feet(J, targets, ok)
    if not targets:
        J = floor_feet(J, mask, ctx, ok)
    return J


def fix_pose(p, J, rec, ctx):
    if p in ICONS: return head_piece('pl18', p, FACE)
    if p in LEGS:
        m = alpha('pl18', p); a = frame_rgba(ctx, p)
        sh = blobs(shoe_mask(a), 6)
        ys, xs = np.nonzero(m); c = np.array([xs.mean(), ys.mean()])
        ank = foot_point(sh[0], c) if sh else np.array([xs.mean(), ys.max() - 6.0])
        hip, kne = leg_from_drawing(m, ank)
        return leg_only(FACE, hip, kne, ank)
    if p in (781, 782): J['_borrowed'] = 780
    if p in FADE:
        J = fit_to_drawing(J, 'pl18', p)
        J['_treat'] = dict(tone=True, dither=('check', 0), outline=False); J['_src'] = ('pl18', p); J['ground'] = None
        J['_borrowed'] = 1411  # keeps it out of the core's pose library (its _treat dict isn't a joint list)
        return J
    if p in NOT_BODY: return None
    J = sane_ground(p, J, 'pl18')
    J = place_ankles(p, J, rec, ctx)
    lengths = standing_lengths(ctx, lambda q, Jq, r: place_ankles(q, Jq, r, ctx), slack=1.02)
    return connect(solve_knees(J, ctx, rec, lengths=lengths))


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
