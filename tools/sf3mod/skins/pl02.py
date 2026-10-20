"""pl02 Ryu -> swordcat.

This file also holds the helpers shared by the group-A fighters (pl02 Ryu, pl11 Ken, pl12 Sean, pl03 Yun, pl07 Ibuki);
their hook files load it as module 'skins_group_a' (see pl11.py). Per frame, on top of the core's cleaned skeleton:

  * facing: SF3 draws every fighter facing left, so face = -1 unless a frame is listed in FIX['face_right'];
  * torso cap: a torso read longer than 1.25x the fighter's typical one (tucked jumps, rolls: the hips read down at
    the knees) has its pelvis pulled up to a normal length;
  * planted feet: on a grounded frame the flat sole runs on the silhouette's bottom edge become ankle targets;
  * fit: a capsule model (torso, head, arm/leg capsules, hands, feet) is laid over the ORIGINAL frame's alpha mask and
    elbows, wrists, knees, ankles, hips and head move (coordinate descent, plus leg templates for tucks and a reach
    search for punches / kicks) to maximise overlap + chamfer with the drawing, held near the core's joints (by
    detector confidence) and the fighter's bone lengths; a limb that gains little by moving keeps the core's pose;
  * temporal: a limb end that jumps away from where both agreeing animation neighbours put it is replaced by their
    average (if that lies on the drawing) - stops one-frame arm/leg pops in idles and walks;
  * per-frame overrides: FIX['overrides'] = {pattern: {'l.kne': (x, y), 'head': (x, y), ...}}.
Fitting costs ~0.25 s a frame (about 3 min per bank).
"""
import json, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

MOD = Path(__file__).resolve().parent.parent


def same_as(bank_name, refs):
    """Every pattern whose original frame is pixel-identical to one of `refs` (SF3 banks repeat small placeholder
    drawings, e.g. a glove, under hundreds of pattern numbers; the animation walk can't prove which are unused)."""
    import hashlib
    d = MOD / 'build' / 'skin' / bank_name / 'frames'
    h = lambda f: hashlib.md5(f.read_bytes()).hexdigest()
    want = {h(d / f'{r:04d}.png') for r in refs if (d / f'{r:04d}.png').exists()}
    return {int(f.stem) for f in d.glob('*.png') if f.stem.isdigit() and h(f) in want}

# swordcat: Ryu is broad (gi, big shoulders, wide trousers); the knight puppet needs a wider torso and thicker limbs,
# a smaller head (the cat head part is big for its torso) and smaller sabatons (they read huge next to Ryu's feet).
CONFIG = dict(scale=1.0, head=0.60, head_sink=0.08, limb_w=0.95, fist=0.52, foot=0.57)
PIVOTS = {'pelvis': ((0.5, 0.3), (0.5, 0.85)), 'cape': ((0.55, 0.45), (0.4, 0.95)), 'tail': ((0.55, 0.92), (0.5, 0.9)), 'foot': ((0.28, 0.59), (0.8, 0.85))}
FIX = dict(face=-1, face_right=set(), overrides={}, leg_w=0.5, pivots=PIVOTS,
           # the 16x16 glove-and-sleeve drawing (794/795 end the electrocution animation; 211 patterns repeat it): the core
           # takes it for an effect, so it would show Ryu's own glove; it becomes a mascot forearm + fist
           arm_pieces=same_as('pl02', [794]))
# the core only calls fix_pose for patterns it posed or that a hook borrows a pose for: borrow the idle's
BORROW = {p: 1 for p in FIX['arm_pieces']}
# Props keep their drawing: Ryu's sandbag from the intro (724-733), which stays on the stage
EFFECT = [(724, 733)]


def fix_pose(p, J, rec, ctx):
    return fix(p, J, rec, ctx, FIX)


# ----------------------------------------------------------------------------------------------------------------------
# shared group-A helpers

THR = 0.3
KP = dict(nose=0, sho=(5, 6), elb=(7, 8), wri=(9, 10), hip=(11, 12), kne=(13, 14), ank=(15, 16))
SIDE = dict(l=0, r=1)
_cache = {}


def bank_stats(ctx):
    """Typical bone lengths of this fighter (median over confidently read frames)."""
    key = ctx['bank_name']
    if key in _cache: return _cache[key]
    L = {k: [] for k in ('ua', 'fa', 'th', 'sh', 'tor', 'shw', 'hipw')}
    for r in ctx['pose'].values():
        if 'kp' not in r: continue
        k = np.array(r['kp']); s = np.array(r['score'])
        for a, b, n in ((5, 7, 'ua'), (6, 8, 'ua'), (7, 9, 'fa'), (8, 10, 'fa'), (11, 13, 'th'), (12, 14, 'th'),
                        (13, 15, 'sh'), (14, 16, 'sh'), (5, 6, 'shw'), (11, 12, 'hipw')):
            if s[a] > .6 and s[b] > .6: L[n].append(np.linalg.norm(k[a] - k[b]))
        if s[[5, 6, 11, 12]].min() > .5:
            L['tor'].append(np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2))
    st = {n: float(np.median(v)) if v else 20.0 for n, v in L.items()}
    _cache[key] = st
    return st


def silhouette(ctx, p):
    f = MOD / 'build' / 'skin' / ctx['bank_name'] / 'frames' / f'{p:04d}.png'
    if not f.exists(): return None
    return np.asarray(Image.open(f).convert('RGBA'))[:, :, 3] > 0


def ground_feet(sil, rec, face, st, prm):
    """Ankle targets from the original's planted feet. On a grounded frame (silhouette bottom 0-7 px under the
    character's position line) the soles are flat runs of columns near the bottom row (SF3 draws the front foot a
    few px higher than the back one). Runs are split where the bottom edge steps, scored by width and lowness;
    the best two are the soles. One ankle target per sole, above its heel half."""
    ys, xs = np.nonzero(sil)
    if not len(ys): return []
    bottom = int(ys.max())
    oy = rec.get('origin', [0, 0])[1]
    if not 0 <= bottom + oy <= prm.get('ground_tol', 7): return []
    low = np.where(sil.any(0), sil.shape[0] - 1 - np.argmax(sil[::-1], 0), -1)
    tol = prm.get('sole_tol', 4)
    runs = []; x = 0; W = len(low)
    while x < W:
        if low[x] >= bottom - tol:
            x0 = x
            while x + 1 < W and low[x + 1] >= bottom - tol and abs(int(low[x + 1]) - int(low[x])) <= 1: x += 1
            runs.append((x0, x, int(np.median(low[x0:x + 1]))))
        x += 1
    runs = [r for r in runs if r[1] - r[0] + 1 >= 5]
    runs.sort(key=lambda r: -((r[1] - r[0] + 1) - 2.5 * (bottom - r[2])))
    flen = prm.get('foot_px', 0.62 * st['sh'])
    if len(runs) == 1 and runs[0][1] - runs[0][0] + 1 > 1.8 * flen:
        a, b, y = runs[0]; m = (a + b) // 2; runs = [(a, m, y), (m + 1, b, y)]
    sole = prm.get('sole_px', 0.2 * st['sh'])
    out = []
    for a, b, y in runs[:2]:
        w = min(b - a + 1, flen * 1.3); c = (a + b) / 2
        out.append(np.array([c - face * prm.get('heel_shift', 0.2) * w, y - sole]))
    return out


class Model:
    """Capsule body drawn at sprite resolution on a padded canvas."""

    def __init__(self, sil, st, prm, face):
        self.P = P = 24
        H, W = sil.shape
        self.sil = np.zeros((H + 2 * P, W + 2 * P), np.uint8); self.sil[P:P + H, P:P + W] = sil
        self.nsil = int(self.sil.sum()); self.silb = self.sil.astype(bool)
        self.wsil = np.ones(self.nsil); self.wsum = float(max(1, self.nsil))
        self.dout = cv2.distanceTransform((1 - self.sil).astype(np.uint8), cv2.DIST_L2, 3)  # px outside the drawing
        tor = st['tor']
        self.aw = max(3, int(round(prm.get('arm_w', 0.2) * tor)))
        self.lw = max(4, int(round(prm.get('leg_w', 0.40) * tor)))
        self.hr = max(4, int(round(prm.get('head_r', 0.30) * tor)))
        self.tw = max(4, int(round(prm.get('torso_w', 0.75) * st['shw'])))
        self.fl = prm.get('foot_l', 0.45) * st['sh']
        self.hl = prm.get('hand_l', 0.45) * st['fa']
        self.face = face
        self.canvas = np.zeros_like(self.sil)

    def draw(self, Q):
        c = self.canvas; c[:] = 0; P = self.P
        names = [('l', 'sho'), ('r', 'sho'), ('r', 'hip'), ('l', 'hip')]
        arr = [Q['l']['sho'], Q['r']['sho'], Q['r']['hip'], Q['l']['hip'], Q['sho'], Q['hip'], Q['head']]
        for s in 'lr':
            L = Q[s]
            fa = L['wri'] - L['elb']; fa = fa / (np.linalg.norm(fa) or 1.0)
            arr += [L['sho'], L['elb'], L['wri'], L['hip'], L['kne'], L['ank'], L['ank'] + np.array([self.face * self.fl, 0.0]),
                    L['wri'] + fa * self.hl]
        ip = [tuple(v) for v in np.rint(np.asarray(arr, float) + P).astype(int).tolist()]
        poly = np.array(ip[:4], np.int32)
        cv2.fillConvexPoly(c, cv2.convexHull(poly), 1)
        cv2.line(c, ip[4], ip[5], 1, self.tw)
        cv2.circle(c, ip[6], self.hr, 1, -1)
        cv2.line(c, ip[4], ip[6], 1, max(3, self.hr))
        aw = self.aw; lw = self.lw
        for k in (7, 15):
            sho, elb, wri, hip, kne, ank, toe, hand = ip[k:k + 8]
            cv2.line(c, sho, elb, 1, aw)
            cv2.line(c, elb, wri, 1, aw)
            cv2.line(c, wri, hand, 1, max(3, int(aw * 1.2)))  # the fist / open hand beyond the wrist
            cv2.line(c, hip, kne, 1, lw)
            cv2.line(c, kne, ank, 1, max(3, int(lw * 0.85)))
            cv2.line(c, ank, toe, 1, max(3, int(lw * 0.45)))
        return c

    def iou(self, Q):
        m = self.draw(Q)
        inter = int(np.count_nonzero(m & self.sil)); union = int(np.count_nonzero(m | self.sil))
        return inter / max(1, union)

    def score(self, Q):
        """(IoU, chamfer): chamfer = mean distance (px) from the drawing's pixels to the nearest model pixel, so a
        fist or foot far out at the end of a thin limb still pulls the model's limb all the way there."""
        m = self.draw(Q)
        inter = int(np.count_nonzero(m & self.sil)); union = int(np.count_nonzero(m | self.sil))
        dt = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
        if not self.nsil: return inter / max(1, union), 0.0
        return inter / max(1, union), float((dt[self.silb] * self.wsil).sum() / self.wsum)

    def weigh(self, centre, tl):
        """Chamfer weights: pixels far from the body centre (fists and feet at full reach) count more."""
        ys, xs = np.nonzero(self.silb)
        r2 = ((xs - self.P - centre[0]) ** 2 + (ys - self.P - centre[1]) ** 2) / max(1.0, tl) ** 2
        self.wsil = 1.0 + r2; self.wsum = float(self.wsil.sum()) / (1.0 + float(r2.mean()))


BONES = (('sho', 'elb', 'ua'), ('elb', 'wri', 'fa'), ('hip', 'kne', 'th'), ('kne', 'ank', 'sh'))


def energy(model, Q, det, w, st, prm, tgt=None):
    iou, ch = model.score(Q)
    e = -prm.get('iou_w', 150.0) * iou + prm.get('chamfer_w', 20.0) * ch
    for s, t in (tgt or {}).items():
        d = Q[s]['ank'] - t
        e += prm.get('feet_w', 3.0) * float(d @ d) / 9.0
    sig2 = prm.get('prior_px', 5.0) ** 2
    for (s, n), q in det.items():
        d = (Q[s][n] if s in ('l', 'r') else Q[n]) - q
        # robust (Cauchy) pull: close to the read it holds like a spring, far away it stops growing, so a joint
        # the detector put on the wrong limb can still leave for the pixels that need it
        d2 = float(d @ d) / sig2
        e += w[(s, n)] * prm.get('prior_k', 4.0) * math.log1p(d2)
        # a joint the detector is sure of (score >= ~0.6) also keeps a plain spring
        e += max(0.0, w[(s, n)] - 0.6) * prm.get('sure_k', 1.0) * d2
    # a knee / elbow / hand / foot off the drawing (a knee under the floor) costs per pixel it is out
    H, W = model.dout.shape; ow = prm.get('out_w', 1.5)
    for s in 'lr':
        for n in ('elb', 'wri', 'kne', 'ank'):
            x, y = int(round(Q[s][n][0] + model.P)), int(round(Q[s][n][1] + model.P))
            d = float(model.dout[min(max(y, 0), H - 1), min(max(x, 0), W - 1)]) + max(0, -x, x - W + 1, -y, y - H + 1)
            if d > 1.5: e += ow * (d - 1.5)
    for s in 'lr':
        L = Q[s]
        for a, b, n in BONES:
            r = np.linalg.norm(L[b] - L[a]) / st[n]
            hi = prm.get('arm_stretch', 1.8) if n in ('ua', 'fa') else prm.get('leg_stretch', 1.3)
            if r > hi: e += (15 if n in ('ua', 'fa') else 40) * (r - hi) ** 2
            lo = 0.45 if n in ('th', 'sh') else 0.35
            if r < lo: e += 80 * (lo - r) ** 2  # a bone folded to a stub draws as a blob
        # an arm folded flat on itself (forearm lying back along the upper arm) only ever hides a missed reach
        u = L['sho'] - L['elb']; v = L['wri'] - L['elb']
        c = float(u @ v) / ((np.linalg.norm(u) * np.linalg.norm(v)) or 1)
        if c > 0.75: e += prm.get('fold_w', 30.0) * (c - 0.75)
    return e


def copyJ(J):
    out = {k: (np.array(v, float) if isinstance(v, np.ndarray) else v) for k, v in J.items() if k not in ('l', 'r')}
    for s in 'lr': out[s] = {n: np.array(v, float) for n, v in J[s].items()}
    return out


def fit(J, rec, sil, st, prm, face, feet=(), core=None):
    """Coordinate-descent fit of the free joints to the silhouette."""
    model = Model(sil, st, prm, face)
    Q = copyJ(J)
    model.weigh((np.asarray(J['sho']) + np.asarray(J['hip'])) / 2, st['tor'])
    # joints that start off the drawing (a guessed leg hanging below a tucked jump) are pulled onto the nearest
    # drawn pixel first, or the descent has nothing to climb
    ys, xs = np.nonzero(sil)
    if len(xs):
        pts = np.stack([xs, ys], 1).astype(float)
        for s in 'lr':
            for n in ('kne', 'ank', 'elb', 'wri'):
                q = Q[s][n]; d2 = ((pts - q) ** 2).sum(1); i = int(d2.argmin())
                if d2[i] > 9: Q[s][n] = pts[i].copy()
    # priors: the core's cleaned joints (its left/right labels, neighbour fills and smoothing), weighted by the
    # detector's score where the joint is still the raw read, 0.5 where the core moved or filled it
    k = np.array(rec['kp'], float) if 'kp' in rec else None; sc = np.array(rec['score']) if 'kp' in rec else None
    cj = core if core is not None else {}
    det = {}; w = {}
    for s in 'lr':
        for n in ('elb', 'wri', 'kne', 'ank'):
            q = cj.get(s, {}).get(n) if cj else None
            if q is None: continue
            q = np.asarray(q, float); score = 0.5
            if k is not None:
                for i in KP[n]:
                    if np.linalg.norm(k[i] - q) < 1.5: score = float(sc[i]); break
            det[(s, n)] = q
            # a joint the detector is sure of stays close; a guess is free to move
            w[(s, n)] = float(np.clip((score - 0.3) / 0.4, 0.05, 1.0)) * prm.get('prior_w', 1.0)
    det[('', 'head')] = np.array(J['head'], float); w[('', 'head')] = 2.0
    hip0 = np.array(J['hip'], float)
    # planted soles: each pulls the nearest ankle (both, when two soles)
    tgt = {}
    if len(feet) == 2:
        a, b = feet
        d1 = np.linalg.norm(Q['l']['ank'] - a) + np.linalg.norm(Q['r']['ank'] - b)
        d2 = np.linalg.norm(Q['l']['ank'] - b) + np.linalg.norm(Q['r']['ank'] - a)
        tgt = dict(l=a, r=b) if d1 <= d2 else dict(l=b, r=a)
    elif len(feet) == 1:
        s1 = min('lr', key=lambda t: np.linalg.norm(Q[t]['ank'] - feet[0]))
        tgt = {s1: feet[0]}
    for t, q in tgt.items():  # start the ankle (and its knee, half way) on the sole
        dv = q - Q[t]['ank']; Q[t]['ank'] = q.copy(); Q[t]['kne'] = Q[t]['kne'] + dv * 0.5
    free = [(s, n) for n in ('kne', 'ank', 'elb', 'wri') for s in 'lr']
    E = energy(model, Q, det, w, st, prm, tgt)
    # leg templates: thigh down / forward-down / forward (tucked jump, knee raised), shin down / tucked back;
    # whichever explains the drawing best is the start for the descent
    if prm.get('templates', True):
        up = Q['sho'] - Q['hip']; up = up / (np.linalg.norm(up) or 1); fw0 = face * np.array([-up[1], up[0]])
        dn = -up
        for t in 'lr':
            if t in tgt: continue
            for ta in (0.0, 0.8, 1.45):
                th = dn * math.cos(ta) + fw0 * math.sin(ta)
                for sa in (0.0, -0.9, -1.6):
                    a2 = ta + sa
                    sh = dn * math.cos(a2) + fw0 * math.sin(a2)
                    R = copyJ(Q)
                    R[t]['kne'] = Q[t]['hip'] + th * st['th']; R[t]['ank'] = R[t]['kne'] + sh * st['sh']
                    e = energy(model, R, det, w, st, prm, tgt)
                    if e < E - 0.5: E, Q = e, R
    # a fist or foot at the far end of a punch / kick: try every hull point of the drawing within reach as the
    # limb's end (elbow / knee half way, bent the natural way) and keep whichever explains the drawing best
    if prm.get('reach', True) and len(xs) > 10:
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)).reshape(-1, 2).astype(float)
        cands = []
        for q in hull:
            if all(np.linalg.norm(q - c) > 5 for c in cands): cands.append(q)
        up = Q['sho'] - Q['hip']; up = up / (np.linalg.norm(up) or 1); fw = face * np.array([-up[1], up[0]])
        for t in 'lr':
            for root, midn, end, b1, b2, bend in (('sho', 'elb', 'wri', 'ua', 'fa', -up), ('hip', 'kne', 'ank', 'th', 'sh', fw)):
                if t in tgt and end == 'ank': continue
                full = st[b1] + st[b2]
                for q in cands:
                    d = np.linalg.norm(q - Q[t][root])
                    if not 0.7 * full < d < (2.2 if end == 'wri' else 1.35) * full: continue
                    R = copyJ(Q)
                    R[t][end] = q.copy()
                    sag = max(0.0, (full * full / 4 - d * d / 4)) ** 0.5 * 0.5
                    R[t][midn] = (Q[t][root] + q) / 2 + bend * sag
                    e = energy(model, R, det, w, st, prm, tgt)
                    if e < E - 0.5: E, Q = e, R
    dirs = [np.array(d, float) for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]

    def moved(Q, key, dv):
        R = copyJ(Q)
        s, n = key
        if n == 'hip':  # move the pelvis as a whole
            R['hip'] = R['hip'] + dv
            for t in 'lr': R[t]['hip'] = R[t]['hip'] + dv
        elif n == 'head':
            R['head'] = R['head'] + dv
        elif n in ('wri+', 'ank+'):  # a hand / foot dragging its elbow / knee half way
            e_, m_ = ('wri', 'elb') if n == 'wri+' else ('ank', 'kne')
            R[s][e_] = R[s][e_] + dv; R[s][m_] = R[s][m_] + dv * 0.5
        else:
            R[s][n] = R[s][n] + dv
            if n == 'kne' and prm.get('carry', True): R[s]['ank'] = R[s]['ank'] + dv * 0.5
            if n == 'elb' and prm.get('carry', True): R[s]['wri'] = R[s]['wri'] + dv * 0.5
        return R

    keys = free + [(t, 'wri+') for t in 'lr'] + [(t, 'ank+') for t in 'lr'] + [('', 'hip'), ('', 'head')]
    def tot(R):
        dh = R['hip'] - hip0
        return energy(model, R, det, w, st, prm, tgt) + float(dh @ dh) / prm.get('hip_px', 4.0) ** 2

    def limb(k):
        n = k[1].rstrip('+')
        return (k[0], 'arm' if n in ('elb', 'wri') else 'leg' if n in ('kne', 'ank') else n)
    E = tot(Q)
    for step in prm.get('steps', (4.0, 2.0, 1.0)):
        ds = dirs if step > 1.5 else dirs[:4]
        active = list(keys)
        for _ in range(4):
            moved_keys = []
            for key in active:
                best = None
                for d in ds:
                    R = moved(Q, key, d * step)
                    e = tot(R)
                    if e < E - 1e-6 and (best is None or e < best[0]): best = (e, R)
                if best is not None:
                    E, Q = best; moved_keys.append(key)
            if not moved_keys: break
            hot = {limb(k) for k in moved_keys}
            active = [k for k in keys if limb(k) in hot or k[1] in ('hip', 'head')]
    # keep the core's limb unless the fit's version explains the drawing clearly better (an arm that only gains a
    # little by moving is noise, and noise flickers from frame to frame)
    for t in 'lr':
        for mid_, end, gain in (('elb', 'wri', prm.get('arm_gain', 8.0)), ('kne', 'ank', prm.get('leg_gain', 4.0))):
            if end == 'ank' and t in tgt: continue
            R = copyJ(Q); R[t][mid_] = np.array(J[t][mid_], float); R[t][end] = np.array(J[t][end], float)
            e = tot(R)
            if e - E < gain: Q, E = R, e
    Q['sho'] = (Q['l']['sho'] + Q['r']['sho']) / 2
    Q['_planted'] = {t: True for t in tgt}
    return Q, model


def apply_overrides(J, ov):
    for key, val in ov.items():
        if '.' in key:
            s, n = key.split('.'); J[s][n] = np.array(val, float)
        else:
            J[key] = val if key == 'face' else np.array(val, float)
    if any(k.endswith('.hip') for k in ov): J['hip'] = (J['l']['hip'] + J['r']['hip']) / 2
    if any(k.endswith('.sho') for k in ov): J['sho'] = (J['l']['sho'] + J['r']['sho']) / 2
    return J


def _skin_module():
    import sys
    for name in ('skin', '__main__'):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, 'frame_links') and hasattr(m, 'fill_limbs'): return m
    import skin
    return skin


_fits = {}


def first_fit(p, J, ctx, prm):
    """Silhouette fit of one frame (no temporal pass), cached per bank + frame. J None: build it from the core."""
    key = (ctx['bank_name'], id(ctx['pose']), p)
    if key in _fits: return _fits[key]
    if J is None:
        core = ctx.get('core', {}).get(p)
        if core is None: _fits[key] = None; return None
        sk = _skin_module()
        J = sk.fill_limbs(sk.copy_J(core), ctx['G'])
    rec = ctx['pose'][str(p)]
    J = copyJ(J)
    # the core's facing (bank facing, plus runs drawn turned the other way); FIX['face_right'] flips single frames
    face = int(J.get('face') or prm.get('face', -1))
    if p in prm.get('face_right', ()): face = -face
    J['face'] = face
    sil = silhouette(ctx, p)
    st = bank_stats(ctx)
    # shoulders / head off the top of the drawing (a core neighbour-fill of a crouch from a standing frame): bring
    # the upper body down so the head sits in the drawing's top blob, keeping the hips
    if sil is not None and sil.any():
        top = int(np.nonzero(sil.any(1))[0][0])
        sy = float(np.asarray(J['sho'])[1])
        if sy < top - 2:
            dv = np.array([0.0, top + 0.55 * st['tor'] - sy])
            J['sho'] = J['sho'] + dv; J['head'] = np.asarray(J['head'], float) + dv
            for t in 'lr':
                for n in ('sho', 'elb', 'wri'): J[t][n] = J[t][n] + dv
    # tucked jumps / rolls: the detector reads the hips down at the knees, stretching the torso; pull the pelvis up
    # to a normal torso length (the thighs then come forward from the belt, as drawn)
    tv = np.asarray(J['hip'], float) - np.asarray(J['sho'], float); tl = np.linalg.norm(tv)
    if tl > prm.get('torso_max', 1.25) * st['tor']:
        dv = tv / tl * st['tor'] * 1.1 - tv
        J['hip'] = J['hip'] + dv
        for t in 'lr': J[t]['hip'] = J[t]['hip'] + dv
    if sil is not None and prm.get('fit', True) and p not in prm.get('no_fit', ()):
        feet = ground_feet(sil, rec, face, st, prm) if prm.get('feet', True) else []
        J, _ = fit(J, rec, sil, st, prm, face, feet, ctx.get('core', {}).get(p))
    _fits[key] = J
    return J


def neighbours(ctx):
    key = ('links', ctx['bank_name'], id(ctx['pose']))
    if key not in _fits:
        sk = _skin_module()
        try:
            _fits[key] = sk.frame_links(ctx['bank_name'], ctx['bank'], ctx['pose'])[0]
        except Exception:
            _fits[key] = {}
    return _fits[key]


LIMBS = (('sho', 'elb', 'wri', 'ua', 'fa'), ('hip', 'kne', 'ank', 'th', 'sh'))


def temporal(p, J, ctx, prm):
    """A limb whose fitted end jumps away from where both animation neighbours (which agree) put it is an outlier
    of the fit (the arm that stretched out for one frame of an idle): take the neighbours' average, relative to the
    limb's root, if that lands on the drawing. Small differences are left alone (the original animation moves too)."""
    occ = neighbours(ctx)
    pairs = [(a, b) for a, b in occ.get(p, ()) if a is not None and b is not None]
    if not pairs: return J
    st = bank_stats(ctx); org = lambda q: np.array(ctx['pose'][str(q)]['origin'], float)
    sil = silhouette(ctx, p)
    if sil is None: return J
    dout = cv2.distanceTransform((~sil).astype(np.uint8), cv2.DIST_L2, 3)
    H, W = sil.shape

    def off(q):
        x, y = int(round(q[0])), int(round(q[1]))
        if not (0 <= x < W and 0 <= y < H): return 99.0
        return float(dout[y, x])
    fits = [(a, b, first_fit(a, None, ctx, prm), first_fit(b, None, ctx, prm)) for a, b in sorted(pairs)]
    for side in 'lr':
        for root, mid_, end, b1, b2 in LIMBS:
            if end == 'ank' and J.get('_planted', {}).get(side): continue
            L = st[b1] + st[b2]
            rp = np.asarray(J[side][root], float); ep = np.asarray(J[side][end], float) - rp
            best = None
            for prev, nxt, A, B in fits:
                if A is None or B is None or A.get('face') != J.get('face'): continue
                ra = lambda X, n: np.asarray(X[side][n], float) - np.asarray(X[side][root], float)
                ea, eb = ra(A, end), ra(B, end)
                gap = np.linalg.norm(ea - eb)
                if gap > prm.get('agree', 0.25) * L: continue
                if best is None or gap < best[0]: best = (gap, (ea + eb) / 2, (ra(A, mid_) + ra(B, mid_)) / 2)
            if best is None: continue
            _, e_avg, m_avg = best
            if np.linalg.norm(ep - e_avg) > prm.get('outlier', 0.35) * L and off(rp + e_avg) < 2 and off(rp + m_avg) < 3:
                J[side][end] = rp + e_avg; J[side][mid_] = rp + m_avg
    return J


_parts = {}


def puppet(ctx, prm):
    """The fighter's puppet parts (with its PIVOTS), cached."""
    key = (ctx['cfg']['mascot'], repr(sorted((prm.get('pivots') or {}).items())))
    if key not in _parts:
        sk = _skin_module()
        _parts[key] = sk.load_puppet(ctx['cfg']['mascot'], prm.get('pivots'))
    return _parts[key]


def _covers(part, A, ang, sx, sy, q, R):
    """Is sprite point q on the opaque pixels of `part` pasted like skin.paste_affine(part, A(canvas), ang, sx, sy)?"""
    c, s = math.cos(ang), math.sin(ang)
    Mx = np.array([[c * sx, s * sy], [-s * sx, c * sy]])
    loc = part.a + np.linalg.solve(Mx, (np.asarray(q, float) + MARGIN) * R - A)
    al = part.alpha
    x, y = int(round(loc[0])), int(round(loc[1]))
    return 0 <= y < al.shape[0] and 0 <= x < al.shape[1] and bool(al[y, x])


MARGIN = 32


def connect(J, ctx, prm):
    """One connected body (the user's 'hips and legs separated'): every thigh top must sit on the pelvis or torso,
    every upper arm root on the torso, the head on the torso's top. A root that would float free is pulled in along
    the line to the body centre (the limb stretches a little instead of detaching)."""
    sk = _skin_module(); R = sk.R; G = ctx['G']; cfg = ctx['cfg']
    global MARGIN; MARGIN = sk.M
    pk = (ctx['cfg']['mascot'], repr(sorted((prm.get('pivots') or {}).items())), J['face'] > 0)
    if pk not in _parts:
        parts = puppet(ctx, prm)
        P = {k: (v if J['face'] > 0 else v.flipped()) for k, v in parts.items()}
        for v in P.values(): v.alpha = np.asarray(v.im)[:, :, 3] > 0
        _parts[pk] = P
    P = _parts[pk]
    S = lambda q: (np.asarray(q, float) + MARGIN) * R
    sho, hip = np.asarray(J['sho'], float), np.asarray(J['hip'], float)

    def torso_ok(q):
        A, B = S(sho), S(hip); t = P['torso']
        d = np.linalg.norm(t.b - t.a); L = np.linalg.norm(B - A)
        ang = math.atan2(B[0] - A[0], B[1] - A[1])
        return _covers(t, A, ang, G, max(L / d, G * 0.6), q, R)

    def pelvis_ok(q):
        tor = S(hip) - S(sho); tang = math.atan2(tor[0], tor[1])
        return _covers(P['pelvis'], S(hip), tang, G, G, q, R)

    def pull(root_key, far_key, centre, ok, side):
        # the limb part is drawn this wide around its root: it touches the body when the root, moved half a limb
        # width towards the body centre, is on the body
        L = J[side]
        r0 = np.asarray(L[root_key], float)
        part = P['thigh' if root_key == 'hip' else 'upperarm']
        half = part.width * G * cfg['limb_w'] / R / 2 * 0.8

        def probe(r):
            d = centre - r; n_ = np.linalg.norm(d)
            return r if n_ < 1e-6 else r + d / n_ * min(half, n_)
        if ok(probe(r0)): return []
        for t in np.linspace(0.1, 1.0, 10):
            r = r0 + (centre - r0) * t
            if ok(probe(r)):
                L[root_key] = r; return [f'{side}.{root_key}:{t:.1f}']
        L[root_key] = centre.copy(); return [f'{side}.{root_key}:x']
    n = []
    for side in 'lr':
        n += pull('hip', 'kne', hip, lambda q: pelvis_ok(q) or torso_ok(q), side)
        n += pull('sho', 'elb', sho, torso_ok, side)
    # head: the neck point a third of the way from the shoulders must be on the head or the torso
    hp = P['head']; hs = G * cfg['head']
    for _ in range(8):
        head = np.asarray(J['head'], float); nk = S(head) - S(sho)
        hc = S(head) - nk / (np.linalg.norm(nk) or 1) * hp.im.height * hs * cfg.get('head_sink', 0.12)
        neck = sho + (head - sho) * 0.35
        ang = math.atan2(-nk[0], -nk[1]) if np.linalg.norm(nk) else 0.0
        ang *= float(np.clip((abs(ang) - 0.9) / 0.6, 0, 1))
        if torso_ok(neck) or _covers(hp, hc, ang, hs, hs, neck, R): break
        J['head'] = sho + (head - sho) * 0.85; n.append('head')
    J['_connect'] = n
    return J


def arm_piece(p, J, ctx, prm, kind='arm'):
    """A loose glove / arm drawing becomes the mascot forearm + fist laid along it, a loose face the mascot head
    (skin.compose_piece)."""
    sk = _skin_module()
    m = silhouette(ctx, p)
    if m is None or not m.any(): return None
    org = np.array(ctx['pose'][str(p)]['origin'], float)
    P = sk._piece_J(kind, m, int(J.get('face') or prm.get('face', -1)), anchor=-org)
    P['_src'] = (ctx['bank_name'], p)
    return P


def fix(p, J, rec, ctx, prm):
    """The group-A per-frame fix: silhouette fit, temporal outlier pass, head offset, overrides, then connect."""
    if p in prm.get('skip', ()): return None
    if J is None: return None
    if p in prm.get('head_pieces', ()): return arm_piece(p, J, ctx, prm, 'head')
    if p in prm.get('arm_pieces', ()): return arm_piece(p, J, ctx, prm)
    J = copyJ(first_fit(p, J, ctx, prm))
    if prm.get('temporal', True): J = temporal(p, J, ctx, prm)
    face = J['face']
    hx, hy = prm.get('head_off', (0.0, 0.0))
    J['head'] = np.array(J['head'], float) + np.array([-face * hx, hy])
    if p in prm.get('overrides', {}): J = apply_overrides(J, prm['overrides'][p])
    if prm.get('connect', True) and not __import__('os').environ.get('GROUPA_NOCONNECT'): J = connect(J, ctx, prm)
    return J
