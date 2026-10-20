"""PUMP COMBAT fighter hooks: pl14 Akuma -> goat (see skin.fighter for the hook contract)."""
import numpy as np
from pathlib import Path
from PIL import Image

FOOT_H = 734  # keyed foot art height (part px)
# Tuned against the original silhouettes (IoU over every 9th pattern): Akuma's broad gi wants thicker limbs and a
# slightly bigger puppet; the goat's hooves sit lower on the ankle pivot (foot_ay) and its fists are smaller.
CONFIG = dict(scale=1.05, limb_w=1.15, head=0.68, head_sink=0.12, fist=0.4, foot=0.75,
              foot_ax=0.3, foot_ay=0.45, pelvis_ay=0.4,
              mesh_foot=0.8)  # mesh renderer: the goat's hooves are drawn big
PIVOTS = {'foot': ((CONFIG['foot_ax'], CONFIG['foot_ay']), (0.8, 0.85)),
          'pelvis': ((0.5, CONFIG['pelvis_ay']), (0.5, 0.85))}

# SF3 sprites are drawn facing one way (skin.core_poses gives every frame the bank's facing); these inclusive
# pattern ranges are drawn facing the other way in the original (turn-arounds, back-turned frames).
FACE_FLIP = []
BORROW = {p: 49 for p in (42, 43, 44, 45, 48, 50, 51, 60)}  # crouch frames reach fix_pose (crouch_template)
# Pure effects (no body pixels): registered in skin.FORCE_EFFECT below so the completion pass keeps their
# drawing instead of fitting a mascot onto them.
EFFECT = [(824, 839),     # ten-kanji
          (848, 896),     # Gou Hadou / Gou Shouryuu fireballs and fire bursts
          (1024, 1055),   # Shun Goku Satsu screen flash + kanji
          (1088, 1155),   # outline wisps and flame pillars
          (758, 761)]     # the small ten-kanji appearing on his back (the completion pass drew GOAT heads on them)
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


# Akuma's crouch (front leg out forward, rear knee down): the detector folds both legs onto one side. Hand pose
# in anchor coordinates (sprite px + origin of pattern 49), used for every pattern whose drawing matches it.
CROUCH = dict(head=(52, 10), sho=(48, 24), hip=(46, 46),
              near=dict(sho=(40, 26), elb=(30, 37), wri=(22, 34), hip=(40, 46), kne=(18, 49), ank=(11, 71)),
              far=dict(sho=(57, 22), elb=(52, 34), wri=(40, 32), hip=(52, 47), kne=(60, 69), ank=(80, 73)))
CROUCH_REF = 49
_tmpl_ok = {}


def _matches(p, q, ctx, thr=0.8):
    """IoU of the drawings of p and q in anchor coordinates."""
    k = (p, q)
    if k in _tmpl_ok: return _tmpl_ok[k]
    a, b = _mask(ctx['bank_name'], p), _mask(ctx['bank_name'], q)
    ok = False
    if a is not None and b is not None and abs(a.shape[0] - b.shape[0]) <= 4:
        oa = np.array(ctx['pose'][str(p)]['origin']); ob = np.array(ctx['pose'][str(q)]['origin'])
        x0, y0 = np.minimum(oa, ob); x1 = max(oa[0] + a.shape[1], ob[0] + b.shape[1]); y1 = max(oa[1] + a.shape[0], ob[1] + b.shape[0])
        A = np.zeros((y1 - y0, x1 - x0), bool); B = A.copy()
        A[oa[1] - y0:oa[1] - y0 + a.shape[0], oa[0] - x0:oa[0] - x0 + a.shape[1]] = a
        B[ob[1] - y0:ob[1] - y0 + b.shape[0], ob[0] - x0:ob[0] - x0 + b.shape[1]] = b
        ok = (A & B).sum() / max(1, (A | B).sum()) >= thr
    _tmpl_ok[k] = ok
    return ok


def crouch_template(p, J, ctx):
    o = np.array(ctx['pose'][str(CROUCH_REF)]['origin'], float); org = np.array(ctx['pose'][str(p)]['origin'], float)
    q = lambda v: np.array(v, float) + o - org
    if ctx['cfg'].get('near_rule', 'physical') == 'legacy': near = 'l' if J['face'] > 0 else 'r'
    else: near = 'r' if J['face'] > 0 else 'l'
    far = 'r' if near == 'l' else 'l'
    J['head'], J['sho'], J['hip'] = q(CROUCH['head']), q(CROUCH['sho']), q(CROUCH['hip'])
    J[near] = {k: q(v) for k, v in CROUCH['near'].items()}; J[far] = {k: q(v) for k, v in CROUCH['far'].items()}
    J['_noclip'] = ('l', 'r')
    return J


def fix_pose(p, J, rec, ctx):
    if _in(p, SKIP) or (cfg_effect_gate(ctx) and effect_frame(p, ctx)): return None
    if _in(p, FACE_FLIP): J['face'] = -J['face']
    cfg = ctx['cfg']
    if p in BORROW and not (40 <= p <= 70): J = borrow(p, BORROW[p], J, ctx)
    if 40 <= p <= 70 and _matches(p, CROUCH_REF, ctx): J = crouch_template(p, J, ctx)
    if cfg.get('clip', True):
        J = clip_limbs(p, J, ctx)
    J = attach_roots(J)
    return J


register_effects('pl14', EFFECT)
