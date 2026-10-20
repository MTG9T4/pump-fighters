"""PUMP COMBAT fighter hooks: pl10 Yang -> ponke (see skin.fighter for the hook contract)."""
import numpy as np
from pathlib import Path
from PIL import Image

FOOT_H = 664  # keyed foot art height (part px)
# Tuned against the original silhouettes (IoU over every 10th pattern): thicker limbs for Yang's baggy trousers,
# smaller fists/feet; the monkey's pelvis hangs lower so its shorts cover the hip joint.
CONFIG = dict(limb_w=1.15, head=0.68, head_sink=0.06, fist=0.4, foot=0.55,
              foot_ax=0.3, foot_ay=0.35, pelvis_ay=0.4)
PIVOTS = {'foot': ((CONFIG['foot_ax'], CONFIG['foot_ay']), (0.8, 0.85)),
          'pelvis': ((0.5, CONFIG['pelvis_ay']), (0.5, 0.85))}

# SF3 sprites are drawn facing one way (skin.core_poses gives every frame the bank's facing); these inclusive
# pattern ranges are drawn facing the other way in the original (turn-arounds, back-turned frames).
FACE_FLIP = []
BORROW = {1229: 1228}  # faded dash after-image: the detector's read is scrambled  # pattern -> pattern whose pose it reuses
# Pure effects (no body pixels): registered in skin.FORCE_EFFECT below so the completion pass keeps their
# drawing instead of fitting a mascot onto them.
EFFECT = [(1135, 1143), (1209, 1216),   # claw slashes
          (1292, 1297)]                  # intro: the cloth bundle Yang drops out of
SKIP = EFFECT
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


def fix_pose(p, J, rec, ctx):
    if _in(p, SKIP) or (cfg_effect_gate(ctx) and effect_frame(p, ctx)): return None
    if _in(p, FACE_FLIP): J['face'] = -J['face']
    cfg = ctx['cfg']
    if p in BORROW: J = borrow(p, BORROW[p], J, ctx)
    if cfg.get('clip', True):
        J = clip_limbs(p, J, ctx)
    J = attach_roots(J)
    return J


register_effects('pl10', EFFECT)

# Loose pieces of the original fighter drawn as separate objects (busts, arms, blurs): redrawn as the mascot.
REDRAW_EXTRA = [474, 475, 477, 478]
