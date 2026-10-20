"""PUMP COMBAT fighter hooks: pl05 Necro -> pepe (see skin.fighter for the hook contract)."""
import numpy as np
from pathlib import Path
from PIL import Image

FOOT_H = 679  # keyed foot art height (part px)
FIST_H = 844
# Tuned against the original silhouettes (IoU over every 9th pattern): Necro is long and thin, so smaller
# feet/fists and a slightly bigger head than the default; pelvis hung lower so the shorts cover the hip joint.
CONFIG = dict(limb_w=0.9, head=0.75, head_sink=0.18, fist=0.4, foot=0.45,
              foot_ax=0.3, foot_ay=0.25, pelvis_ay=0.4)
PIVOTS = {'foot': ((CONFIG['foot_ax'], CONFIG['foot_ay']), (0.8, 0.85)),
          'pelvis': ((0.5, CONFIG['pelvis_ay']), (0.5, 0.85))}

# SF3 sprites are drawn facing one way (skin.core_poses gives every frame the bank's facing); these inclusive
# pattern ranges are drawn facing the other way in the original (turn-arounds, back-turned frames).
FACE_FLIP = [(612, 617)]
BORROW = {960: 964, 961: 964}  # pattern -> pattern whose pose it reuses
# Pure effects (no body pixels): registered in skin.FORCE_EFFECT below so the completion pass keeps their
# drawing instead of fitting a mascot onto them.
EFFECT = [(1131, 1142),   # ground trail
          (1166, 1167)]   # '48E' / '48F' debug labels
OTHER = [(1034, 1115), (1196, 1218)]  # Effie (Necro's win-pose partner): another character, keeps her drawing
SKIP = [(1034, 1115), (1196, 1218)] + EFFECT
# pattern ranges above are left as colour-mapped originals


# ---- shared helpers (identical in pl05/pl09/pl10/pl14/pl17) ----
def _in(p, ranges):
    return any(a <= p <= b for a, b in ranges)


_masks = {}


def _mask(bank_name, p):
    k = (bank_name, p)
    if k not in _masks:
        f = Path(__file__).resolve().parents[1] / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
        _masks[k] = np.asarray(Image.open(f).convert('RGBA'))[:, :, 3] > 0 if f.exists() else None
    return _masks[k]


def _inside(m, q, pad=2):
    x, y = int(round(q[0])), int(round(q[1]))
    H, W = m.shape
    return m[max(0, y - pad):min(H, y + pad + 1), max(0, x - pad):min(W, x + pad + 1)].any()


def clip_limbs(p, J, ctx, gap=4):
    """A limb the detector threw into empty space (no original pixels under it) is cut back to where the
    original's silhouette ends, so the puppet never grows an arm or leg the frame doesn't have."""
    m = _mask(ctx['bank_name'], p)
    if m is None: return J
    for side in 'lr':
        if side in J.get('_noclip', ()): continue  # a hook placed this side's limbs by hand
        L = J[side]
        for chain in (('sho', 'elb', 'wri'), ('hip', 'kne', 'ank')):
            pts = [L[k] for k in chain]
            out = 0; last = pts[0]; cut = None
            for si in range(2):
                A, B = pts[si], pts[si + 1]; n = max(2, int(np.linalg.norm(B - A)))
                for t in np.linspace(0, 1, n + 1)[1:]:
                    q = A + (B - A) * t
                    if _inside(m, q): last = q; out = 0
                    else:
                        out += 1
                        if out > gap: cut = (si, last); break
                if cut: break
            if not cut: continue
            si, q = cut
            if si == 0:  # the elbow/knee itself is off the body: fold the limb into the stretch that exists
                L[chain[1]] = (pts[0] + q) / 2; L[chain[2]] = q.copy()
            else:
                L[chain[2]] = q.copy()
    return J
def effect_frame(p, ctx):
    """Hit sparks, lightning, ribbons, kanji, dithered after-images: thin (mean half-thickness < 2.6 px) or broken-up
    (largest blob < 70%) silhouettes are not a body; the puppet stays off them (colour-mapped original)."""
    import cv2
    m = _mask(ctx['bank_name'], p)
    if m is None or m.sum() < 20: return False
    m8 = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    big = st[1:, cv2.CC_STAT_AREA].max() / m8.sum()
    thick = cv2.distanceTransform(m8, cv2.DIST_L2, 3)[m].mean()
    return thick < 2.6 or big < 0.7
def borrow(p, q, J, ctx):
    """Frame p's pose taken from frame q of the same pose (anchor coordinates line up), for frames the detector
    read upside down or scrambled. Keeps p's facing."""
    import skin
    core = ctx.get('core') or {}
    if q not in core: return J
    K = skin.fill_limbs(skin.copy_J(core[q]), ctx['G'])
    d = np.array(ctx['pose'][str(q)]['origin'], float) - np.array(ctx['pose'][str(p)]['origin'], float)
    for k, v in K.items():
        if isinstance(v, dict): K[k] = {kk: (None if vv is None else vv + d) for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.shape == (2,): K[k] = v + d
    K['face'] = J['face']
    for k in ('ground', 'ankle_h'):
        if k in J: K[k] = J[k]
    return K
def attach_roots(J, k_sho=0.32, k_hip=0.3):
    """One connected body: every shoulder / hip joint stays within reach of the torso's own shoulder / hip centre
    (a fraction of the torso length); a root read out on the drawing's far side is pulled in and its bone stretches
    to the elbow / knee, which stay on the drawing."""
    tl = float(np.linalg.norm(J['hip'] - J['sho'])) or 1.0
    for side in 'lr':
        L = J[side]
        for key, ctr, k in (('sho', J['sho'], k_sho), ('hip', J['hip'], k_hip)):
            if L.get(key) is None: L[key] = ctr.copy(); continue
            d = L[key] - ctr; n = float(np.linalg.norm(d)); r = max(3.0, k * tl)
            if n > r: L[key] = ctr + d / n * r
    return J
def cfg_effect_gate(ctx):
    """effect_frame() drops thin / broken-up silhouettes; off by default now that skin.core_poses has its own effect
    gate and the completion pass gives after-images a mascot (cfg effect_gate=True turns it back on)."""
    return bool(ctx['cfg'].get('effect_gate', False))
def register_effects(bank_name, ranges):
    """Hand a hook's pure-effect pattern ranges to the core (skin.FORCE_EFFECT), whichever name skin runs under."""
    import sys
    for name in ('skin', '__main__'):
        m = sys.modules.get(name)
        if m is not None and isinstance(getattr(m, 'FORCE_EFFECT', None), dict):
            m.FORCE_EFFECT[bank_name] = tuple(tuple(r) for r in ranges)
# ---- end shared helpers ----


ARM, LEG = 41.0, 64.0  # Necro's idle shoulder->wrist and hip->ankle lengths (sprite px)


def _skeleton_paths(m, J):
    """Thin the original's silhouette and walk it from the torso: returns [(path px list root->tip)] for every
    skeleton branch tip (geodesic BFS from the pixels near the shoulder-hip line)."""
    import cv2
    from collections import deque
    sk = cv2.ximgproc.thinning(m.astype(np.uint8) * 255) > 0
    ys, xs = np.nonzero(sk)
    if not len(ys): return []
    pts = set(zip(ys.tolist(), xs.tolist()))
    a, b = J['sho'], J['hip']; ab = b - a; L2 = max(1e-6, ab @ ab)
    R = max(6.0, 0.4 * np.sqrt(L2))
    par = {}; dist = {}; q = deque()
    for y, x in pts:
        t = np.clip(((np.array([x, y]) - a) @ ab) / L2, 0, 1)
        if np.linalg.norm(a + t * ab - (x, y)) <= R:
            par[(y, x)] = None; dist[(y, x)] = 0.0; q.append((y, x))
    if not q: return []
    while q:  # BFS (unit steps; good enough for path recovery)
        c = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (c[0] + dy, c[1] + dx)
                if (dy or dx) and n in pts and n not in par:
                    par[n] = c; dist[n] = dist[c] + (1.414 if dy and dx else 1.0); q.append(n)
    # branch tips = leaves of the BFS tree that are the farthest point of their neighbourhood (corner pixels of
    # an 8-connected skeleton make extra 1-pixel leaves)
    parents = {v for v in par.values() if v is not None}
    tips = []
    for c in par:
        if c in parents or dist[c] <= 8: continue
        if all(dist.get((c[0] + dy, c[1] + dx), -1) <= dist[c] for dy in range(-3, 4) for dx in range(-3, 4)):
            tips.append(c)
    out = []
    for t in tips:
        path = []; c = t
        while c is not None: path.append((c[1], c[0])); c = par[c]
        out.append(np.array(path[::-1], float))
    return out


def stretch_limbs(p, J, ctx):
    """Necro's rubber arms and legs: when the original's limb runs well past the detected wrist/ankle, slide the
    wrist/ankle out to the tip of the limb and put the elbow/knee half way along it, so the puppet stretches too."""
    m = _mask(ctx['bank_name'], p)
    if m is None: return J
    try:
        paths = _skeleton_paths(m, J)
    except Exception:
        return J
    used = set()
    jobs = [(s, 'sho', 'elb', 'wri', ARM) for s in 'lr'] + [(s, 'hip', 'kne', 'ank', LEG) for s in 'lr']
    best = []
    for s, root, midj, tip, normal in jobs:
        L = J[s]; d = L[tip]
        for i, path in enumerate(paths):
            dd = np.linalg.norm(path - d, axis=1); k = int(dd.argmin())
            if dd[k] > 7: continue
            seg = np.linalg.norm(np.diff(path[k:], axis=0), axis=1).sum()  # skeleton length past the joint
            reach = np.linalg.norm(path[-1] - L[root])
            if seg > 12 and reach > 1.2 * normal:
                best.append((seg, i, s, root, midj, tip, k))
    # long branches no detected wrist/ankle sits on: give them to the nearest limb of the right kind
    a, b = J['sho'], J['hip']; ab = b - a; L2 = max(1e-6, ab @ ab)
    for i, path in enumerate(paths):
        t = ((path[0] - a) @ ab) / L2
        kind = [j for j in jobs if (j[1] == 'sho') == (t < 0.55)]
        normal = kind[0][4]
        if np.linalg.norm(path[-1] - path[0]) < 1.4 * normal: continue
        for s, root, midj, tip, _ in kind:
            dmin = np.linalg.norm(path - J[s][tip], axis=1).min()
            best.append((-1000 - dmin, i, s, root, midj, tip, len(path) // 2))
    for seg, i, s, root, midj, tip, k in sorted(best, key=lambda x: -x[0]):
        if i in used or (s, tip) in used: continue
        used.add(i); used.add((s, tip))
        path = paths[i]; L = J[s]
        # the fist/foot art hangs past its pivot: pull the pivot back from the tip by that much
        cfg = ctx['cfg']; g = ctx['G'] / 4
        pull = 0.8 * FIST_H * cfg['fist'] * g if tip == 'wri' else (1 - cfg['foot_ay']) * FOOT_H * cfg['foot'] * g
        run = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[::-1], axis=0), axis=1))]
        j = min(int(np.searchsorted(run, pull)), len(path) - 1 - k)
        ti = len(path) - 1 - j; L[tip] = path[ti]
        # elbow/knee half way along the limb's own centreline (follows a bend), from where it leaves the root
        ri = int(np.linalg.norm(path[:ti + 1] - L[root], axis=1).argmin())
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[ri:ti + 1], axis=0), axis=1))]
        L[midj] = path[ri + int(np.searchsorted(arc, arc[-1] / 2))] if arc[-1] > 4 else (L[root] + L[tip]) / 2
    return J


def head_forward(p, J, ctx):
    """Necro hunches: his head hangs in front of and level with the shoulders, and the detector often puts it on
    the shoulder hump behind. A head behind the shoulders (upright frames) is mirrored to the front, if the
    original has pixels there."""
    if J.get('head') is None: return J
    t = J['sho'] - J['hip']; tl = float(np.linalg.norm(t))
    if tl < 6: return J
    u = t / tl
    if u[1] > -0.6: return J  # lying, flipping: leave it
    fw = J['face'] * np.array([-u[1], u[0]])
    if fw[0] * J['face'] < 0: fw = -fw
    d = J['head'] - J['sho']; a = float(d @ fw)
    if a > -0.1 * tl: return J
    cand = J['sho'] + d - 2 * a * fw
    m = _mask(ctx['bank_name'], p)
    if m is not None and _inside(m, cand, pad=3): J['head'] = cand
    return J


def fix_pose(p, J, rec, ctx):
    if _in(p, SKIP) or (cfg_effect_gate(ctx) and effect_frame(p, ctx)): return None
    if _in(p, FACE_FLIP): J['face'] = -J['face']
    cfg = ctx['cfg']
    if p in BORROW: J = borrow(p, BORROW[p], J, ctx)
    J = head_forward(p, J, ctx)
    if cfg.get('stretch', True):
        J = stretch_limbs(p, J, ctx)
    if cfg.get('clip', True):
        J = clip_limbs(p, J, ctx)
    J = attach_roots(J)
    return J


register_effects('pl05', EFFECT)

# Loose pieces of the original fighter drawn as separate objects (busts, arms, blurs): redrawn as the mascot.
REDRAW_EXTRA = [996, 997]
