"""CLAUDIA on Chun-Li (pl16).

Chun-Li is a kicker: the pose model often reads her kicking leg as an arm (the leg comes up at shoulder height) or
loses the standing leg under the qipao, so the puppet kicked low or lost its legs (363-367, 373-374). Her boots are
the one thing that is always easy to see: big white blobs, the shaft running up the shin. Each leg is rebuilt from
its boot: ankle at the boot's far end, shin along the boot shaft, knee one shin length up it, thigh from the hip.

This file also holds the leg helpers the other group-B hooks share (pl04 / pl01 / pl08 / pl13 load it with lib()).
"""
from pathlib import Path
import importlib.util
import math
import numpy as np
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
MOD = HERE.parent

# Feet: the shared foot size made the platform boots ~2x the original's and hung the sole ~10 px under the floor.
# Size set so the boot is about the original's length; ankle pivot at the top of the shoe part of the boot art
# (median ankle->sole distance over the base-move frames: 10.4 px).
# Thighs a little fuller (Chun-Li's legs are the widest part of her silhouette).
CONFIG = dict(foot=0.62, limb_w=0.85)
PIVOTS = {'foot': ((0.3, 0.49), (0.8, 0.85))}


# ---------------------------------------------------------------- shared helpers (group B)

_cache = {}


def lib(name='pl16'):
    """This module (shared helpers), loaded once."""
    import sys
    key = f'_groupB_{name}'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f'{name}.py')
        m = importlib.util.module_from_spec(spec); sys.modules[key] = m; spec.loader.exec_module(m)
    return sys.modules[key]


def frame_rgba(ctx, p):
    """The original frame as an int RGBA array (sprite px), cached."""
    key = (ctx['bank_name'], p)
    if key not in _cache:
        f = MOD / 'build' / 'skin' / ctx['bank_name'] / 'frames' / f'{p:04d}.png'
        _cache[key] = np.asarray(Image.open(f).convert('RGBA')).astype(int) if f.exists() else None
    return _cache[key]


def blobs(mask, min_area):
    """Connected blobs of a bool mask: list of dict(area, c=centroid, xs, ys), largest first."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if st[i][4] < min_area: continue
        ys, xs = np.nonzero(lab == i)
        out.append(dict(area=int(st[i][4]), c=np.array(cen[i]), xs=xs, ys=ys, box=tuple(st[i][:4])))
    out.sort(key=lambda b: -b['area'])
    return out


def leg_lengths(ctx):
    """Typical (thigh, shin) of the original from confidently read frames."""
    key = ('legs', ctx['bank_name'])
    if key not in _cache:
        th, sh = [], []
        for rec in ctx['pose'].values():
            if 'kp' not in rec: continue
            kp = np.array(rec.get('kp', [[0, 0]] * 17), float); sc = np.array(rec.get('score', [0] * 17), float)
            for h, k, a in ((11, 13, 15), (12, 14, 16)):
                if min(sc[h], sc[k], sc[a]) < 0.6: continue
                th.append(np.linalg.norm(kp[k] - kp[h])); sh.append(np.linalg.norm(kp[a] - kp[k]))
        _cache[key] = (float(np.percentile(th, 60)), float(np.percentile(sh, 60)))
    return _cache[key]


def thickness(b):
    """Largest inscribed radius of a blob (px)."""
    x0, y0 = b['xs'].min(), b['ys'].min()
    m = np.zeros((b['ys'].max() - y0 + 3, b['xs'].max() - x0 + 3), np.uint8)
    m[b['ys'] - y0 + 1, b['xs'] - x0 + 1] = 1
    return float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max())


def ik_knee(hip, ank, l1, l2, bend):
    """Knee of a two-bone leg hip->ank on the side `bend` points to; out of reach -> on the straight line."""
    d = ank - hip; L = float(np.linalg.norm(d))
    if L < 1e-3: return hip + bend / (np.linalg.norm(bend) or 1) * l1
    u = d / L
    if L >= l1 + l2 - 1e-3: return hip + u * L * l1 / (l1 + l2)
    L = max(L, abs(l1 - l2) + 1e-3)
    a = (l1 * l1 - l2 * l2 + L * L) / (2 * L); h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    n = np.array([-u[1], u[0]])
    if np.dot(n, bend) < 0: n = -n
    return hip + u * a + n * h


def shin_from_blob(b, hip, l2, sole=7.0, lift=3.0):
    """(ankle, knee) from a boot / lower-leg blob: ankle near the blob's far end from the hip, the shin running
    from there toward the blob's near end, knee one shin length up it."""
    pts = np.stack([b['xs'], b['ys']], 1).astype(float)
    d = np.linalg.norm(pts - hip, axis=1)
    far = pts[d >= d.max() - sole].mean(0)
    near = pts[d <= d.min() + 6].mean(0)
    u = near - far; n = float(np.linalg.norm(u))
    if n < 3: u = hip - far; n = float(np.linalg.norm(u)) or 1.0
    u = u / n
    ank = far + u * lift
    return ank, ank + u * l2


def assign(J, cands, sides=('l', 'r')):
    """Match candidate (ankle, knee) pairs to the legs by least total ankle+knee distance. Returns {side: cand}."""
    def cost(s, c):
        L = J[s]
        return np.linalg.norm(np.asarray(L['ank'], float) - c[0]) + 0.5 * np.linalg.norm(np.asarray(L['kne'], float) - c[1])
    if len(cands) >= 2:
        a, b = cands[:2]
        if cost('l', a) + cost('r', b) <= cost('l', b) + cost('r', a): return {'l': a, 'r': b}
        return {'l': b, 'r': a}
    if len(cands) == 1:
        s = min(sides, key=lambda s: cost(s, cands[0]))
        return {s: cands[0]}
    return {}


def set_leg(J, side, ank, kne, l2, tol_ang=0.6):
    """Ankle onto the target; the knee too, unless the current knee already sits on the target shin line
    (direction within tol_ang rad, length within 0.65-1.45 shin), so good knee reads are kept."""
    L = J[side]
    cur = np.asarray(L['kne'], float)
    v_cur = cur - ank; v_new = kne - ank
    n1, n2 = float(np.linalg.norm(v_cur)), float(np.linalg.norm(v_new))
    keep = False
    if n1 > 1e-3 and n2 > 1e-3 and 0.65 * l2 <= n1 <= 1.45 * l2:
        keep = math.acos(float(np.clip(v_cur @ v_new / (n1 * n2), -1, 1))) < tol_ang
    L['ank'] = ank
    if not keep: L['kne'] = kne
    return J


def fix_thigh(J, side, l1, stretch=1.25):
    """Keep the knee from flying off: a thigh longer than stretch * typical is pulled in along hip->knee, the
    shin following (ankle stays)."""
    L = J[side]; hip = np.asarray(L['hip'], float); kne = np.asarray(L['kne'], float)
    v = kne - hip; n = float(np.linalg.norm(v))
    if n > stretch * l1:
        L['kne'] = hip + v / n * stretch * l1
    return J


def _hsv_feat(c):
    h = cv2.cvtColor(c.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    a = h[:, 0] / 90 * np.pi
    return np.stack([np.cos(a) * h[:, 1], np.sin(a) * h[:, 1], h[:, 2] * 0.5], 1).astype(np.float32)


def body_likeness(ctx, p, K=6):
    """How much a frame's colour mix looks like the fighter's idle drawing (histogram intersection of 6 material
    groups, 0..1). Bodies score >= 0.6 (also hit flashes in the fighter's own colours); pure effects (spirals,
    sparks, music notes, thrown clothes, trails) score <= 0.35."""
    bn = ctx['bank_name']
    key = ('mat', bn)
    if key not in _cache:
        idle = []
        for q in ctx['bank'].standing_patterns(8):
            a = frame_rgba(ctx, q)
            if a is not None: idle.append(a[a[..., 3] > 0][:, :3])
        idle = np.concatenate(idle)
        cols, cnt = np.unique(idle, axis=0, return_counts=True)
        feat = np.repeat(_hsv_feat(cols), np.minimum(cnt, 50), 0)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        cv2.setRNGSeed(1)
        _, _, cen = cv2.kmeans(feat, K, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        grp = lambda c: ((_hsv_feat(c)[:, None] - cen[None]) ** 2).sum(2).argmin(1)
        ref = np.bincount(grp(idle), minlength=K) / len(idle)
        _cache[key] = (grp, ref)
    grp, ref = _cache[key]
    a = frame_rgba(ctx, p)
    if a is None: return 0.0
    c = a[a[..., 3] > 0][:, :3]
    if len(c) < 30: return 0.0
    mix = np.bincount(grp(c), minlength=len(ref)) / len(c)
    return float(np.minimum(mix, ref).sum())


def _key(c):
    return (c[:, 0].astype(np.int64) << 16) | (c[:, 1].astype(np.int64) << 8) | c[:, 2]


def foot_palette(ctx, near=12, share=0.7, min_count=8):
    """Colours that are (almost) only used on the shoes: >= `share` of their idle-frame pixels lie within `near` px
    of the floor (the idle frames' bottom row). Empty for barefoot fighters (their feet share the skin colours)."""
    key = ('footpal', ctx['bank_name'])
    if key not in _cache:
        tot, low = {}, {}
        for q in ctx['bank'].standing_patterns(8):
            a = frame_rgba(ctx, q)
            if a is None: continue
            m = a[..., 3] > 0; ys = np.nonzero(m.any(1))[0]
            if not len(ys): continue
            k = _key(a[m][:, :3]); yy = np.nonzero(m)[0]
            for v, c in zip(*np.unique(k, return_counts=True)): tot[v] = tot.get(v, 0) + c
            kl = k[yy >= ys.max() - near]
            for v, c in zip(*np.unique(kl, return_counts=True)): low[v] = low.get(v, 0) + c
        def ok(v):  # neutral greys / black / white and bright tints also draw outlines, shirts and skin highlights
            r, g, b = (v >> 16) & 255, (v >> 8) & 255, v & 255
            return max(r, g, b) <= 180 and max(r, g, b) - min(r, g, b) >= 12
        _cache[key] = np.array(sorted(v for v, c in tot.items()
                                      if c >= min_count and low.get(v, 0) / c >= share and ok(v)), np.int64)
    return _cache[key]


def shoe_mask(ctx, p, grow=1):
    """Pixels in the fighter's shoe colours (closed, grown by `grow` px into the outline)."""
    a = frame_rgba(ctx, p); pal = foot_palette(ctx)
    if a is None or not len(pal): return None
    m = a[..., 3] > 0
    out = np.zeros(m.shape, bool); out[m] = np.isin(_key(a[m][:, :3]), pal)
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if grow: out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=grow) & m.astype(np.uint8)
    return out > 0


def legs_from_blobs(J, ctx, found, l1, l2, **kw):
    """Rebuild legs from lower-leg blobs (boots / shoes): ankle at each blob's far end from the hip, shin along the
    blob, knee kept when the read agreed. Returns J."""
    hip = np.asarray(J['hip'], float)
    cands = [shin_from_blob(b, hip, l2, **kw) for b in found]
    for s, (ank, kne) in assign(J, cands).items():
        set_leg(J, s, ank, kne, l2)
        fix_thigh(J, s, l1)
    return connect(J)


def shoe_ankle(b, hip, depth=3.0):
    """Ankle inside a low shoe: the shoe's opening (its pixels nearest the hip), pushed `depth` px into the shoe."""
    pts = np.stack([b['xs'], b['ys']], 1).astype(float)
    d = np.linalg.norm(pts - hip, axis=1)
    top = pts[d <= d.min() + 3].mean(0)
    u = pts.mean(0) - top; n = float(np.linalg.norm(u)) or 1.0
    return top + u / n * min(depth, n)


def shoe_blobs(ctx, p, J, min_area=15):
    """Up to two shoe blobs (shoe_mask) that can be feet: within leg reach of the hip and, unless the pose model
    put an ankle on them (kicks, flips), below the hips along the body (the same leather tones turn up in hair,
    moustaches and glove shadows)."""
    m = shoe_mask(ctx, p)
    if m is None: return []
    l1, l2 = leg_lengths(ctx)
    hip = np.asarray(J['hip'], float); sho = np.asarray(J['sho'], float)
    down = (hip - sho) / (np.linalg.norm(hip - sho) or 1)
    rec = ctx['pose'][str(p)]
    kp = np.array(rec.get('kp', [[0, 0]] * 17), float); sc = np.array(rec.get('score', [0] * 17), float)
    anks = [kp[i] for i in (15, 16) if sc[i] >= 0.4]
    out = []
    for b in blobs(m, min_area)[:4]:
        d = np.linalg.norm(b['c'] - hip)
        if d > 1.9 * (l1 + l2) or d < 0.5 * l1: continue
        if np.dot(b['c'] - hip, down) < 0.3 * l1 and not any(np.linalg.norm(b['c'] - k) < 12 for k in anks): continue
        out.append(b)
        if len(out) == 2: break
    return out


def shoe_legs(J, ctx, found, l1, l2):
    """Ankles onto low-shoe blobs; a knee that no longer fits the leg (too long / too short a shin) is re-solved by
    IK, bending the way the read knee did (else forward)."""
    hip = np.asarray(J['hip'], float)
    cands = [(a, a) for a in (shoe_ankle(b, hip) for b in found)]
    for s, (ank, _) in assign(J, cands).items():
        L = J[s]
        L['ank'] = ank
        kne = np.asarray(L['kne'], float); h = np.asarray(L['hip'], float)
        sl = float(np.linalg.norm(kne - ank)); tl = float(np.linalg.norm(kne - h))
        if not (0.35 * l2 <= sl <= 1.8 * l2) or tl > 1.8 * l1:
            d = ank - h; n = np.array([-d[1], d[0]])
            off = float(np.dot(kne - h, n))
            bend = n * np.sign(off) if abs(off) > 1e-3 else np.array([J['face'] * 1.0, -0.3])
            L['kne'] = ik_knee(h, ank, l1, l2, bend)
    return J


def top_palette(ctx, frac=0.18, share=0.7, min_count=8, pred=None):
    """Colours (almost) only used in the top `frac` of the idle frames (hair, headband...). pred(r, g, b) filters."""
    key = ('toppal', ctx['bank_name'], frac, share)
    if key not in _cache:
        tot, top = {}, {}
        for q in ctx['bank'].standing_patterns(8):
            a = frame_rgba(ctx, q)
            if a is None: continue
            m = a[..., 3] > 0; ys = np.nonzero(m.any(1))[0]
            if not len(ys): continue
            k = _key(a[m][:, :3]); yy = np.nonzero(m)[0]
            lim = ys.min() + frac * (ys.max() - ys.min())
            for v, c in zip(*np.unique(k, return_counts=True)): tot[v] = tot.get(v, 0) + c
            for v, c in zip(*np.unique(k[yy <= lim], return_counts=True)): top[v] = top.get(v, 0) + c
        _cache[key] = np.array(sorted(v for v, c in tot.items()
                                      if c >= min_count and top.get(v, 0) / c >= share), np.int64)
    pal = _cache[key]
    if pred is not None:
        pal = np.array([v for v in pal if pred((v >> 16) & 255, (v >> 8) & 255, v & 255)], np.int64)
    return pal


def hair_front(ctx, p, face, pal, min_area=12, front=0.35):
    """Centroid of the front part (the `front` share along the facing direction) of the largest hair blob."""
    a = frame_rgba(ctx, p)
    if a is None or not len(pal): return None
    m = a[..., 3] > 0
    hm = np.zeros(m.shape, bool); hm[m] = np.isin(_key(a[m][:, :3]), pal)
    hm = cv2.morphologyEx(hm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    bl = blobs(hm, min_area)
    if not bl: return None
    b = bl[0]
    x = b['xs'] * face
    sel = x >= np.quantile(x, 1 - front)
    return np.array([b['xs'][sel].mean(), b['ys'][sel].mean()])


def near_palette(ctx, p, pt, pal, r):
    """True when a pixel of one of the colours in pal lies within r px of pt."""
    a = frame_rgba(ctx, p)
    if a is None or pt is None or not len(pal): return False
    x, y = int(round(pt[0])), int(round(pt[1])); h, w = a.shape[:2]
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    if x0 >= x1 or y0 >= y1: return False
    win = a[y0:y1, x0:x1]; m = win[..., 3] > 0
    return bool(m.any() and np.isin(_key(win[m][:, :3]), pal).any())


def head_offset(ctx, pal, face):
    """Median (head - hair_front) over the idle frames, from the cleaned poses."""
    key = ('headoff', ctx['bank_name'])
    if key not in _cache:
        core = ctx.get('core', {}); offs = []
        for q in ctx['bank'].standing_patterns(8):
            J = core.get(q)
            if J is None or J.get('head') is None: continue
            h = hair_front(ctx, q, face, pal)
            if h is not None: offs.append(np.asarray(J['head'], float) - h)
        _cache[key] = np.median(offs, 0) if offs else np.zeros(2)
    return _cache[key]


def connect(J, sho_r=0.45, hip_r=0.35):
    """Keep every limb rooted on the body: a limb's shoulder more than sho_r torso lengths from the shoulder centre
    (or a hip more than hip_r from the hip centre) is pulled in toward it, so the puppet stays one piece."""
    if J is None or J.get('_piece'): return J
    sho, hip = np.asarray(J['sho'], float), np.asarray(J['hip'], float)
    tl = float(np.linalg.norm(hip - sho)) or 20.0
    for s in 'lr':
        L = J[s]
        for k, c, r in (('sho', sho, sho_r), ('hip', hip, hip_r)):
            if L.get(k) is None: continue
            v = np.asarray(L[k], float) - c; n = float(np.linalg.norm(v))
            if n > r * tl: v = v / n * r * tl
            # and never above the top of the torso (shoulder) / below the bottom of the pelvis (hip), where the
            # limb's root would hang off the art
            u = (hip - sho) / tl
            out = float(v @ (-u if k == 'sho' else u))
            lim = 0.1 * tl if k == 'sho' else 0.15 * tl
            if out > lim: v = v - (-u if k == 'sho' else u) * (out - lim)
            L[k] = c + v
    return J


def borrow(ctx, src, p, face=None):
    """Frame src's cleaned pose moved into frame p's sprite space (drawings' centroids aligned), for frames whose own read is
    hopeless but that are drawn in the same pose as another frame (shock x-rays, holds). None if src has none."""
    import skin
    core = ctx.get('core', {})
    if src not in core: return None
    J = skin.fill_limbs(skin.copy_J(core[src]), ctx['G'])
    d = np.asarray(ctx['pose'][str(src)]['origin'], float) - np.asarray(ctx['pose'][str(p)]['origin'], float)
    for k in ('sho', 'hip', 'head'):
        if J.get(k) is not None: J[k] = np.asarray(J[k], float) + d
    for s in 'lr':
        for k, v in J[s].items():
            if v is not None: J[s][k] = np.asarray(v, float) + d
    # align by the drawings' centroids (jump / throw frames move the anchor between frames)
    ma, mb = frame_rgba(ctx, src), frame_rgba(ctx, p)
    if ma is not None and mb is not None and (ma[..., 3] > 0).any() and (mb[..., 3] > 0).any():
        ya, xa = np.nonzero(ma[..., 3] > 0); yb, xb = np.nonzero(mb[..., 3] > 0)
        d2 = np.array([xb.mean() - xa.mean(), yb.mean() - ya.mean()]) - d
        for k in ('sho', 'hip', 'head'):
            if J.get(k) is not None: J[k] = np.asarray(J[k], float) + d2
        for s in 'lr':
            for k, v in J[s].items():
                if v is not None: J[s][k] = np.asarray(v, float) + d2
    if 'ground' in J and 'ground' in core.get(p, {}): J['ground'] = core[p]['ground']
    if face is not None: J['face'] = face
    return J


def not_body(ctx, p, keep=(), thr=0.45):
    """True for effect frames (sparks, spirals, trails, props) that the pose model / animation fill still gave a
    skeleton: they keep their own drawing. `keep` lists body-shaped frames in odd colours (shock, morph, flash)."""
    return p not in keep and body_likeness(ctx, p) < thr


# ---------------------------------------------------------------- Chun-Li / CLAUDIA

# Body-shaped frames in effect colours: the super's white-blue kick silhouette (1518), the flesh-coloured morph into
# Chun-Li in her intro (1633-1642).
KEEP = {1518} | set(range(1633, 1643))
# Effects drawn in her own skin / white colours, so the colour test can't tell them: Kikosho ball (1539-1571),
# Kikoken fireballs (1583-1603), swooshes and dust (1610-1631).
NOT_BODY = set(range(1539, 1572)) | set(range(1583, 1604)) | set(range(1610, 1632))
# ... and the completion pass must not draw CLAUDIA on them either: Kikoken / Kikosho balls, fireballs, swooshes, dust
EFFECT = [(1536, 1632)]

def boot_mask(a):
    """White / light grey neutral pixels (her boots, also hair ribbons and wristband studs)."""
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    return (al > 0) & (np.abs(r - g) <= 6) & (np.abs(g - b) <= 6) & (r >= 120)


def boots(ctx, p, J):
    a = frame_rgba(ctx, p)
    if a is None: return []
    l1, l2 = leg_lengths(ctx)
    hip = np.asarray(J['hip'], float); head = np.asarray(J['head'], float)
    out = []
    for b in blobs(boot_mask(a), 70)[:5]:
        if np.linalg.norm(b['c'] - hip) > 1.9 * (l1 + l2) or np.linalg.norm(b['c'] - hip) < 0.45 * l1: continue
        # hair ribbons / buns are thin strands (<= 3 px from an edge); a boot shaft is >= 3.7 px thick
        if thickness(b) < 3.3: continue
        if np.linalg.norm(b['c'] - head) < 12 and b['area'] < 160: continue
        out.append(b)
        if len(out) == 2: break
    return out


def fix_pose(p, J, rec, ctx):
    if p in NOT_BODY or not_body(ctx, p, KEEP): return None
    l1, l2 = leg_lengths(ctx)
    hip = np.asarray(J['hip'], float)
    bl = boots(ctx, p, J)
    cands = [shin_from_blob(b, hip, l2) for b in bl]
    for s, (ank, kne) in assign(J, cands).items():
        set_leg(J, s, ank, kne, l2)
        fix_thigh(J, s, l1)
    return connect(J)

# Loose pieces of the original fighter drawn as separate objects (busts, arms, blurs): redrawn as the mascot.
REDRAW_EXTRA = [775, 1280, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604]
