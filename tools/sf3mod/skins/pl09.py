"""PUMP COMBAT fighter hooks: pl09 Oro -> slerf (see skin.fighter for the hook contract)."""
import numpy as np
from pathlib import Path
from PIL import Image

FOOT_H = 585  # keyed foot art height (part px)
# Tuned against the original silhouettes (IoU over every 10th pattern): Oro is a big hunched body in a robe, so the
# slender sloth puppet is scaled up with thick limbs and a bigger head.
CONFIG = dict(scale=1.2, limb_w=1.05, head=0.75, head_sink=0.06, fist=0.5, foot=0.55,
              foot_ax=0.3, foot_ay=0.25, pelvis_ay=0.4)
PIVOTS = {'foot': ((CONFIG['foot_ax'], CONFIG['foot_ay']), (0.8, 0.85)),
          'pelvis': ((0.5, CONFIG['pelvis_ay']), (0.5, 0.85))}

# SF3 sprites are drawn facing one way (skin.core_poses gives every frame the bank's facing); these inclusive
# pattern ranges are drawn facing the other way in the original (turn-arounds, back-turned frames).
FACE_FLIP = [(72, 72)]
BORROW = {}  # pattern -> pattern whose pose it reuses (filled in below)
# Pure effects (no body pixels): registered in skin.FORCE_EFFECT below so the completion pass keeps their
# drawing instead of fitting a mascot onto them.
EFFECT = [(451, 451), (464, 473), (490, 492),   # 'xxx missing' debug labels
          (1232, 1283), (1285, 1400)]                 # Nichirin / Yagyou orbs and the dome
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


_rgb = {}


def _frame(bank_name, p):
    k = (bank_name, p)
    if k not in _rgb:
        f = Path(__file__).resolve().parents[1] / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
        _rgb[k] = np.asarray(Image.open(f).convert('RGBA')) if f.exists() else None
    return _rgb[k]


def _skin_frac(im, L):
    """Share of samples along shoulder->elbow->wrist that land on Oro's bare yellow skin (his tied arm is wrapped
    in the red/purple sash, so the free arm is the one whose bones run over skin)."""
    pts = [L['sho'] + (L['elb'] - L['sho']) * t for t in np.linspace(.3, 1, 6)] +           [L['elb'] + (L['wri'] - L['elb']) * t for t in np.linspace(0, 1, 8)]
    hits = []
    for x, y in pts:
        x, y = int(round(x)), int(round(y))
        win = im[max(0, y - 1):y + 2, max(0, x - 1):x + 2].astype(int)
        if not win.size: hits.append(False); continue
        r, g, b, a = win[..., 0], win[..., 1], win[..., 2], win[..., 3]
        hits.append(bool(((r > 140) & (g > 100) & (r - b > 70) & (g - b > 40) & (a > 0)).any()))
    return float(np.mean(hits))


def one_arm(p, J, rec, ctx):
    """Oro fights with one arm; the other is tied inside his robe. Keep the free arm (drawn as the near arm, over
    the body) and fold the tied one inside the torso, where the torso art hides it."""
    im = _frame(ctx['bank_name'], p)
    sc = np.array(rec['score']) if 'score' in rec else np.zeros(17)  # frames the core filled from neighbours have no read
    # the side skin.compose draws in front ('physical' rule: facing right shows COCO 'r')
    if ctx['cfg'].get('near_rule', 'physical') == 'legacy': near = 'l' if J['face'] > 0 else 'r'
    else: near = 'r' if J['face'] > 0 else 'l'
    far = 'r' if near == 'l' else 'l'
    fl, fr = (_skin_frac(im, J[s]) if im is not None else 0.0 for s in 'lr')
    # the tied arm is the bundle on the back shoulder: the free hand is the more forward wrist, unless that one
    # runs over cloth while the other runs over skin
    fwd = 'l' if J['l']['wri'][0] * J['face'] > J['r']['wri'][0] * J['face'] else 'r'
    frac = dict(l=fl, r=fr); other = 'r' if fwd == 'l' else 'l'
    vis = other if frac[fwd] < 0.15 and frac[other] > 0.5 else fwd
    if vis != near:
        for k in ('sho', 'elb', 'wri'):
            J[near][k], J[far][k] = J[far][k], J[near][k]
    hide = J['sho'] + 0.4 * (J['hip'] - J['sho'])
    for k in ('sho', 'elb', 'wri'):
        J[far][k] = hide.copy()
    return J


# Hand-placed poses for frames the detector can't read (Oro's crouch: the cape and sash pull the hips out to the
# back and lay the torso flat). Joints in anchor coordinates (sprite px + the pattern's origin), so one template
# serves every pattern of the pose; 'near' = the free arm / front leg side.
CROUCH = dict(head=(30, 14), sho=(41, 24), hip=(60, 40),
              near=dict(sho=(37, 25), elb=(29, 39), wri=(13, 40), hip=(57, 40), kne=(45, 47), ank=(41, 57)),
              far=dict(sho=(45, 23), elb=(45, 23), wri=(45, 23), hip=(63, 40), kne=(83, 43), ank=(78, 57)))
# Close-up bust (Oro hovering cross-legged, arms folded): legs tucked under the torso
BUST = dict(head=(38, 14), sho=(36, 29), hip=(36, 48),
            near=dict(sho=(31, 31), elb=(17, 40), wri=(8, 47), hip=(33, 48), kne=(24, 51), ank=(32, 53)),
            far=dict(sho=(40, 28), elb=(40, 28), wri=(40, 28), hip=(39, 48), kne=(48, 51), ank=(40, 53)))
BUSTS = [(928, 973), (978, 991), (1002, 1023)]
KEEP = [p for a, b in BUSTS for p in range(a, b + 1)]  # never dropped by the core's effect gate
TEMPLATES = {p: (CROUCH, (-48, -57), 0) for p in (51, 52, 53, 74)}
TEMPLATES.update({p: (BUST, (-37, -93), 0) for p in KEEP})
TEMPLATES.update({73: (CROUCH, (-48, -57), 3), 72: (CROUCH, (-48, -57), 4)})  # crouch turn-around
MIRRORED = {72}
BORROW.update({p: 940 for p in KEEP if p != 940})  # bust frames the core dropped still reach fix_pose
TEMPLATES.update({50: (CROUCH, (-48, -57), 0), 55: (CROUCH, (-48, -57), 6)})  # (template, origin, body lift px)


def apply_template(p, J, ctx):
    t, o, lift = TEMPLATES[p]
    org = np.array(ctx['pose'][str(p)]['origin'], float)
    mx = -1.0 if p in MIRRORED else 1.0  # mirrored template: anchor x -> -x (the pose turned the other way)
    def q(v, up=1.0):
        a = np.array(v, float) + np.array(o, float); a[0] *= mx
        return a - org - np.array([0.0, lift * up])
    if ctx['cfg'].get('near_rule', 'physical') == 'legacy': near = 'l' if J['face'] > 0 else 'r'
    else: near = 'r' if J['face'] > 0 else 'l'
    far = 'r' if near == 'l' else 'l'
    J['head'], J['sho'], J['hip'] = q(t['head']), q(t['sho']), q(t['hip'])
    up = dict(sho=1, elb=1, wri=1, hip=1, kne=0.5, ank=0)  # a lifted body keeps its feet on the floor
    for side, key in ((near, 'near'), (far, 'far')):
        J[side] = {k: q(v, up[k]) for k, v in t[key].items()}
    J['_template'] = True
    if t is BUST: J['plant'] = False
    return J


def fix_pose(p, J, rec, ctx):
    if _in(p, SKIP) or (cfg_effect_gate(ctx) and effect_frame(p, ctx)): return None
    if _in(p, FACE_FLIP): J['face'] = -J['face']
    cfg = ctx['cfg']
    if p in BORROW: J = borrow(p, BORROW[p], J, ctx)
    if p in TEMPLATES:
        return apply_template(p, J, ctx)
    if cfg.get('one_arm', True):
        J = one_arm(p, J, rec, ctx)
    if cfg.get('clip', True):
        J = clip_limbs(p, J, ctx)
    J = attach_roots(J)
    return J


register_effects('pl09', EFFECT)

# Loose pieces of the original fighter drawn as separate objects (busts, arms, blurs): redrawn as the mascot.
REDRAW_EXTRA = [242, 809, 940, 945, 949, 952, 953, 987, 988, 990, 991, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1216, 1217, 1218]
