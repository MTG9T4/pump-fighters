"""Full-body mascot skins: pose a 2D cut-out puppet onto every sprite frame of an SF3 fighter.

Inputs per fighter: build/skin/<bank>/pose.json (sf3mod/pose.py) and the puppet parts in
assets/puppet/<mascot>/*.png (Codex art, black background). For every frame the puppet's bones are laid
along the detected skeleton, rendered at 4x, shrunk to sprite size, clipped to the frame's existing tiles
(the bank can't grow) and written in a new 64-colour mascot palette (all 13 colour buttons).

    python sf3mod/skin.py pl02 swordcat [--frames 8,104] [--preview]
build(ext, out, only=None) writes plNN.bin + plNNpl.bin into the overrides dir.

Poses are cleaned up per animation, not per frame (core_poses: the fighter's animation scripts are parsed by
sf3mod/anims.py into build/skin/<bank>/anims.json): one facing per bank, left/right labels kept consistent between
neighbouring frames, failed frames posed from their neighbours, feet planted on the original's floor line.
After the hooks, classify_and_complete() makes sure no pattern shows the original fighter: every body pattern left
unposed gets a fitted pose, a puppet part (head icon, loose arm / leg, bust) or a toned silhouette; only effects and
other characters keep their drawing (python sf3mod/skin_audit.py <bank|all> checks it, plus puppet connectivity).
Hook module attributes the core reads: CONFIG, PIVOTS, fix_pose, KEEP (never effect-gated), BORROW (pose source for
frames the core couldn't pose), OTHER / EFFECT (patterns that keep their drawing, even when the detector posed them:
props like Ryu's sandbag or Dudley's car go in EFFECT), PIECES ({'head'|'arm'|'leg'|'bust'|
'body': ranges}), SILHOUETTE (dark body inside an effect). J extras a hook may set: plant=False, near='l'|'r',
ground=None, tail_scale, tail_angle, _treat (None = keep mascot colours).

Drawing: when the mascot has mesh layer art (assets/mesh/<m>/body.png, arm.png, bind.json; sf3mod/mesh_art.py +
sf3mod/mesh.py) compose() hands each posed frame to mesh.compose_mesh(): one continuous body surface (torso, pelvis,
both legs) and two arm surfaces deformed by smooth bone weights, so hips, knees and shoulders bend without seams.
CONFIG renderer='puppet' keeps the old rigid cut-out paste (the default when no mesh art exists); loose pieces
(head icons, loose arms / legs, busts) and frames whose body a hook parked off the canvas always use the cut-out parts.
Mesh widths follow the cut-out's tuned widths (limb_w, foot, fist); mesh_torso_w / mesh_leg_w / mesh_arm_w /
mesh_foot / mesh_fist / mesh_hip_follow / mesh_far_arm_shade fine-tune them per fighter.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageFilter

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD / 'tools'))
if str(MOD) not in sys.path: sys.path.insert(0, str(MOD))
from spritebank import SpriteBank, Palettes, load_palettes  # noqa

R = 4  # render supersampling
NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP, LKNE, RKNE, LANK, RANK = range(17)
THR = 0.3

# Which mascot skins which bank, and puppet proportions (multiples of the global scale, which is set so the
# puppet torso matches the fighter's idle torso).
SKINS = {
    'pl02': dict(mascot='swordcat', head=0.8, head_sink=0.06, limb_w=0.85, fist=0.72, foot=0.85),
    'pl11': dict(mascot='inu', head=0.68, head_sink=0.14, limb_w=0.85, fist=0.7, foot=0.85),
    'pl04': dict(mascot='jeanphil', head=0.62, head_sink=0.06, limb_w=0.85, fist=0.55, foot=0.85),
    'pl13': dict(mascot='elon', head=0.6, head_sink=0.06, limb_w=0.85, fist=0.55, foot=0.85),
    'pl05': dict(mascot='pepe', head=0.62, head_sink=0.08, limb_w=0.8, fist=0.55, foot=0.85),
    'pl16': dict(mascot='claudia', head=0.6, head_sink=0.06, limb_w=0.75, fist=0.6, foot=0.8),
    'pl03': dict(mascot='wif', head=0.68, head_sink=0.12, limb_w=0.85, fist=0.7, foot=0.85),
    'pl17': dict(mascot='popcat', head=0.7, head_sink=0.14, limb_w=0.85, fist=0.7, foot=0.85),
    'pl18': dict(mascot='pnut', head=0.66, head_sink=0.1, limb_w=0.72, fist=0.7, foot=0.85),
    'pl06': dict(mascot='moodeng', head=0.66, head_sink=0.08, limb_w=1.0, fist=0.8, foot=0.9),
    # spare slots
    'pl01': dict(mascot='giga', head=0.6, head_sink=0.05, limb_w=1.0, fist=0.8, foot=0.9),
    'pl14': dict(mascot='goat', head=0.72, head_sink=0.12, limb_w=0.85, fist=0.7, foot=0.85),
    'pl12': dict(mascot='chillguy', head=0.68, head_sink=0.08, limb_w=0.85, fist=0.7, foot=0.85),
    'pl10': dict(mascot='ponke', head=0.68, head_sink=0.08, limb_w=0.85, fist=0.7, foot=0.85),
    'pl07': dict(mascot='mew', head=0.7, head_sink=0.14, limb_w=0.8, fist=0.65, foot=0.8),
    'pl08': dict(mascot='pippin', head=0.7, head_sink=0.14, limb_w=0.8, fist=0.65, foot=0.8),
    'pl09': dict(mascot='slerf', head=0.66, head_sink=0.06, limb_w=0.9, fist=0.75, foot=0.85),
    'pl20': dict(mascot='bonk', head=0.68, head_sink=0.12, limb_w=0.85, fist=0.7, foot=0.85),
    'pl19': dict(mascot='fwog', head=0.8, head_sink=0.06, limb_w=0.9, fist=0.7, foot=0.85),
    # final boss: Gill -> PILL (palette has two halves; palette_bytes writes the mascot into both)
    'pl00': dict(mascot='pill', head=0.6, head_sink=0.06, limb_w=0.95, fist=0.7, foot=0.8, cape=0.95),
}


def border_background(bgmask):
    """The near-black pixels connected (4-way) to ANY border pixel: the flat background. Enclosed dark pixels
    (eyes, hair, outlines inside the shape) stay part of the drawing."""
    n, lab = cv2.connectedComponents(bgmask.astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    return np.isin(lab, edge)


def key_part(path):
    """Black background -> alpha, keeping dark outlines that are inside the shape."""
    a = np.array(Image.open(path).convert('RGB'))
    bg = border_background(a.max(axis=2) < 14)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    # soften the cut edge one pixel
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
    rgba = np.dstack([a, alpha])
    im = Image.fromarray(rgba, 'RGBA')
    return im.crop(im.getbbox())


class Part:
    def __init__(self, im, top=None, bottom=None):
        self.im = im
        w, h = im.size
        al = np.asarray(im)[:, :, 3] > 0
        ys = np.nonzero(al.any(1))[0]

        def cx(y):
            xs = np.nonzero(al[int(np.clip(y, 0, h - 1))])[0]
            return (xs.min() + xs.max()) / 2 if len(xs) else w / 2
        # joint centres sit half a limb-width in from each rounded end
        wt = al[: max(1, h // 5)].sum(1).max(); wb = al[-max(1, h // 5):].sum(1).max()
        ty = top if top is not None else min(h * 0.4, wt * 0.45)
        by = bottom if bottom is not None else max(h * 0.6, h - wb * 0.45)
        self.a = np.array([cx(ty), ty]); self.b = np.array([cx(by), by])
        self.width = al.sum(1).max()

    def flipped(self):
        if getattr(self, '_flip', None) is not None and self._flip_key == (self.a.tobytes(), self.b.tobytes()):
            return self._flip
        p = Part.__new__(Part)
        p.im = self.im.transpose(Image.FLIP_LEFT_RIGHT); w = self.im.width
        p.a = np.array([w - self.a[0], self.a[1]]); p.b = np.array([w - self.b[0], self.b[1]]); p.width = self.width
        self._flip, self._flip_key = p, (self.a.tobytes(), self.b.tobytes())  # parts are flipped every frame: keep it
        return p


# Joint pivots as (x, y) fractions of each keyed part: a = joint nearest the body, b = far joint.
PIVOTS = {
    'head': ((0.5, 0.45), (0.5, 0.95)), 'torso': ((0.5, 0.07), (0.5, 0.93)), 'pelvis': ((0.5, 0.38), (0.5, 0.85)),
    'upperarm': ((0.55, 0.12), (0.5, 0.88)), 'forearm': ((0.5, 0.08), (0.5, 0.92)), 'fist': ((0.42, 0.12), (0.5, 0.9)),
    'thigh': ((0.5, 0.1), (0.5, 0.86)), 'shin': ((0.5, 0.08), (0.5, 0.92)), 'foot': ((0.3, 0.42), (0.8, 0.85)),
    'cape': ((0.6, 0.05), (0.4, 0.95)), 'tail': ((0.5, 0.03), (0.3, 0.9)),
}


def fighter(bank_name):
    """Per-fighter settings: SKINS[bank] merged with the optional hook module sf3mod/skins/<bank>.py, which may define
    CONFIG (dict, overrides SKINS), PIVOTS (dict, overrides part pivots) and fix_pose(p, J, rec, ctx) -> J."""
    import importlib.util
    cfg = dict(SKINS[bank_name]); hooks = None
    f = MOD / 'skins' / f'{bank_name}.py'
    if f.exists():
        spec = importlib.util.spec_from_file_location(f'skins_{bank_name}', f)
        hooks = importlib.util.module_from_spec(spec); spec.loader.exec_module(hooks)
        cfg.update(getattr(hooks, 'CONFIG', {}))
    return cfg, hooks


class Parts(dict):
    """The puppet's cut-out parts by name; .mesh is the mascot's mesh model (sf3mod/mesh.py) when the mesh renderer
    is on (cfg['renderer'] == 'mesh', the default for mascots with layer art in assets/mesh/<m>/), else None."""
    mesh = None


def load_puppet(mascot, pivots=None):
    d = MOD / 'assets' / 'puppet' / mascot
    piv = dict(PIVOTS); piv.update(pivots or {})
    parts = Parts()
    for f in d.glob('*.png'):
        im = key_part(f); p = Part(im)
        if f.stem in piv:
            (ax, ay), (bx, by) = piv[f.stem]
            p.a = np.array([ax * im.width, ay * im.height]); p.b = np.array([bx * im.width, by * im.height])
        parts[f.stem] = p
    return parts


def paste_bone(canvas, part, A, B, sw, min_len=0.5, shade=1.0):
    """Map part joint a->A and b->B: rotate to the bone, stretch along it, sw = across-scale."""
    d = np.linalg.norm(part.b - part.a)
    L = np.linalg.norm(np.asarray(B) - np.asarray(A))
    sl = max(L / d, sw * min_len)
    ang = math.atan2(B[0] - A[0], B[1] - A[1])  # 0 = pointing down
    paste_affine(canvas, part, A, ang, sw, sl, shade)


def paste_affine(canvas, part, A, ang, sx, sy, shade=1.0):
    c, s = math.cos(ang), math.sin(ang)
    # forward: p -> A + Rot(-ang) * diag(sx, sy) * (p - part.a)  (y down, ang measured from +y toward +x)
    M = np.array([[c * sx, s * sy], [-s * sx, c * sy]])
    off = np.asarray(A, float) - M @ part.a
    im = part.im
    if shade != 1.0:
        cache = part.__dict__.setdefault('_shade', {})
        if shade not in cache:
            arr = np.asarray(im).astype(np.float32).copy(); arr[:, :, :3] *= shade
            cache[shade] = Image.fromarray(arr.clip(0, 255).astype(np.uint8), 'RGBA')
        im = cache[shade]
    _paste_xform(canvas, im, M, off)


def _paste_xform(canvas, im, M, off):
    """Composite `im` mapped by x -> M x + off onto the canvas, transforming only the covered rectangle."""
    w, h = im.size
    cor = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) @ M.T + off
    x0, y0 = np.floor(cor.min(0)).astype(int) - 1; x1, y1 = np.ceil(cor.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, canvas.width), min(y1, canvas.height)
    if x1 <= x0 or y1 <= y0: return
    Mi = np.linalg.inv(M); oi = Mi @ (np.array([x0, y0], float) - off)
    layer = im.transform((x1 - x0, y1 - y0), Image.AFFINE, (Mi[0, 0], Mi[0, 1], oi[0], Mi[1, 0], Mi[1, 1], oi[1]),
                         resample=Image.BILINEAR)
    canvas.alpha_composite(layer, (int(x0), int(y0)))


def pt(kp, sc, i):
    return np.array(kp[i]) if sc[i] >= THR else None


def mid(a, b):
    if a is None: return b
    if b is None: return a
    return (a + b) / 2


def pose_frame(rec):
    """Clean a detection into the joints the puppet needs, or None when it can't be trusted."""
    if 'kp' not in rec: return None
    kp, sc = np.array(rec['kp']), np.array(rec['score'])
    if sc[[LSHO, RSHO, LHIP, RHIP]].max() < THR or np.mean(sc) < 0.33: return None
    sho = mid(pt(kp, sc, LSHO), pt(kp, sc, RSHO)); hip = mid(pt(kp, sc, LHIP), pt(kp, sc, RHIP))
    if sho is None or hip is None: return None
    headpts = [kp[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if sc[i] >= THR]
    head = np.mean(headpts, 0) if headpts else None
    ears = [kp[i] for i in (LEAR, REAR) if sc[i] >= THR]
    face = 1
    if sc[NOSE] >= THR and ears:
        face = 1 if kp[NOSE][0] >= np.mean(ears, 0)[0] else -1
    elif head is not None:
        face = 1 if head[0] >= sho[0] else -1
    J = dict(sho=sho, hip=hip, head=head, face=face)
    for side, (s, e, w, h, k, a) in dict(l=(LSHO, LELB, LWRI, LHIP, LKNE, LANK),
                                         r=(RSHO, RELB, RWRI, RHIP, RKNE, RANK)).items():
        J[side] = {n: pt(kp, sc, i) for n, i in dict(sho=s, elb=e, wri=w, hip=h, kne=k, ank=a).items()}
    return J


def fill_limbs(J, G):
    """Guess missing joints so every limb can be drawn."""
    torso = J['hip'] - J['sho']; tl = np.linalg.norm(torso) or G
    down = torso / tl
    for side in 'lr':
        L = J[side]
        L['sho'] = L['sho'] if L['sho'] is not None else J['sho']
        L['hip'] = L['hip'] if L['hip'] is not None else J['hip']
        if L['elb'] is None:
            L['elb'] = (L['sho'] + L['wri']) / 2 + down * tl * 0.1 if L['wri'] is not None else L['sho'] + down * tl * 0.45
        if L['wri'] is None:
            L['wri'] = L['elb'] + (L['elb'] - L['sho']) * 0.9
        if L['kne'] is None:
            L['kne'] = (L['hip'] + L['ank']) / 2 if L['ank'] is not None else L['hip'] + down * tl * 0.55
        if L['ank'] is None:
            L['ank'] = L['kne'] + (L['kne'] - L['hip'])
    if J['head'] is None:
        J['head'] = J['sho'] - down * tl * 0.45
    return J


# ------------------------------------------------------------------------------------------------------------
# Core clean-up shared by every fighter: the detector reads each sprite frame on its own, so its left/right
# labels, facing and weak joints jump around from one frame of an animation to the next. Here every frame is
# looked at together with its neighbours in the fighter's real animation scripts (sf3mod/anims.py):
#   * facing is the bank's drawing direction (SF3 draws every frame of a fighter facing one way; the game mirrors),
#     so the puppet never flips mid-animation; rotations (flips, knockdowns) come from the bones alone
#   * left/right labels of legs and arms are chosen to minimise motion between neighbouring frames
#   * crossed legs and backward knees are repaired against the original's silhouette
#   * missing or weak joints, and frames the detector failed on, are filled from neighbouring frames
#   * small detector jitter is smoothed, limb joints are kept on the original's pixels
#   * each frame carries the original's floor line (J['ground']) so planted feet sit flat on it
# Coordinates inside are 'anchor' coordinates (sprite px + pattern origin): the same for every frame of a fighter.

LEG_PAIRS = ((LHIP, RHIP), (LKNE, RKNE), (LANK, RANK))
ARM_PAIRS = ((LSHO, RSHO), (LELB, RELB), (LWRI, RWRI))
LIMB_JOINTS = (LELB, RELB, LWRI, RWRI, LKNE, RKNE, LANK, RANK)
CORE_DEFAULTS = dict(
    face_mode='bank',   # 'bank': one facing for the whole bank (cfg 'face' = +1/-1 forces it); 'detect': per frame
    temporal=True,      # label consistency, joint fill and smoothing over animation neighbours
    fill_frames=True,   # pose frames the detector failed on from their animation neighbours
    smooth=0.35,        # pull of a joint towards its neighbours' midpoint when it only jitters (0 = off)
    leg_repair=True,    # fix crossed legs / backward knees against the silhouette
    snap=True,          # limb joints that fall just outside the original's pixels are moved onto them
    plant=True,         # feet near the floor line sit flat on it (compose); one frame can opt out: J['plant'] = False
    effect_gate=True,   # leave pure effect frames (sparks, spirals, trails) colour-mapped
    keep_frames=(),     # patterns the effect gate must not drop
    turned_runs=True,   # runs of an animation clearly drawn facing the other way get the other facing
    head_fix=True,      # heads read behind the neck or far off on a glove go back to the head's usual place
)
_core_cache = {}


_mask_files = {}


def _alpha(bank_name, p, cache):
    """The original frame's opaque mask (sprite px) or None. All masks of a bank are cached in one file
    (build/skin/<bank>/masks.npz, rebuilt when the frames folder changes) because opening ~1500 PNGs is slow."""
    if p in cache: return cache[p]
    d = MOD / 'build' / 'skin' / bank_name
    if bank_name not in _mask_files:
        fr = d / 'frames'; npz = d / 'masks.npz'; data = {}
        stamp = fr.stat().st_mtime if fr.exists() else 0
        try:
            if npz.exists() and npz.stat().st_mtime >= stamp:
                z = np.load(npz)
                data = {int(k[1:]): z[k] for k in z.files}
                if len(data) != sum(1 for _ in fr.glob('*.png')): data = {}
        except Exception:
            data = {}
        if not data and fr.exists():
            for f in fr.glob('*.png'):
                a = np.asarray(Image.open(f).convert('RGBA'))[:, :, 3] > 0
                data[int(f.stem)] = a
            try:
                np.savez_compressed(npz, **{f'p{k}': v for k, v in data.items()})
            except Exception:
                pass
        _mask_files[bank_name] = data
    m = _mask_files[bank_name].get(p)
    cache[p] = None if m is None else m.astype(bool)
    return cache[p]


def effect_shape(mask):
    """Hit sparks, lightning, ribbons, kanji, dithered after-images: a thin (mean half-thickness < 2.6 px) or
    broken-up (largest blob < 70% of the pixels) silhouette is not a body."""
    if mask is None or mask.sum() < 20: return False
    m8 = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n < 2: return False
    big = st[1:, cv2.CC_STAT_AREA].max() / m8.sum()
    pad = np.pad(m8, 1)  # a frame filled edge to edge has no background for the distance transform
    thick = float(cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1][mask].mean())
    return thick < 2.6 or big < 0.7


def _hsv_feat(c):
    h = cv2.cvtColor(c.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    a = h[:, 0] / 90 * np.pi
    return np.stack([np.cos(a) * h[:, 1], np.sin(a) * h[:, 1], h[:, 2] * 0.5], 1).astype(np.float32)


_part_cache = {}


def main_part(bank, p):
    """The palette part most of a pattern's pixels use. Part 0 is the fighter's selected colour (his body); the fixed
    parts hold effects but also bodies drawn in other colours (Urien's win pose and suit, Necro's burn), so a colour
    likeness against the idle drawing is only meaningful on part 0."""
    key = id(bank)
    if key not in _part_cache: _part_cache[key] = {}
    c = _part_cache[key]
    if p not in c:
        cnt = {}
        try:
            for t in bank.tiles(p):
                cnt[t['attr'] & 0x1ff] = cnt.get(t['attr'] & 0x1ff, 0) + int((t['pixels'] > 0).sum())
        except Exception:
            pass
        c[p] = max(cnt, key=cnt.get) if cnt else 0
    return c[p]


def _solidity(mf):
    cs, _ = cv2.findContours(mf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = sum(cv2.contourArea(cv2.convexHull(c)) for c in cs)
    return float(mf.sum()) / max(hull, 1.0)


_like_cache = {}


def body_likeness(bank_name, bank, p, K=6):
    """How much a frame's colour mix looks like the fighter's idle drawing (histogram intersection of 6 material
    groups, 0..1; group B's measure). Bodies, also hit flashes in the fighter's own colours, score >= 0.6; pure
    effects (spirals, sparks, notes, thrown clothes, trails) <= 0.45."""
    d = MOD / 'build' / 'skin' / bank_name / 'frames'
    rgba = lambda q: np.asarray(Image.open(d / f'{q:04d}.png').convert('RGBA')) if (d / f'{q:04d}.png').exists() else None
    if bank_name not in _like_cache:
        idle = []
        try:
            for q in bank.standing_patterns(8):
                a = rgba(q)
                if a is not None: idle.append(a[a[..., 3] > 0][:, :3])
        except Exception:
            pass
        if not idle: _like_cache[bank_name] = None; return 1.0
        idle = np.concatenate(idle)
        cols, cnt = np.unique(idle, axis=0, return_counts=True)
        feat = np.repeat(_hsv_feat(cols), np.minimum(cnt, 50), 0)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        cv2.setRNGSeed(1)
        _, _, cen = cv2.kmeans(feat, K, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        grp = lambda c: ((_hsv_feat(c)[:, None] - cen[None]) ** 2).sum(2).argmin(1)
        _like_cache[bank_name] = (grp, np.bincount(grp(idle), minlength=K) / len(idle), {})
    if _like_cache[bank_name] is None: return 1.0
    grp, ref, memo = _like_cache[bank_name]
    if p not in memo:
        a = rgba(p)
        c = a[a[..., 3] > 0][:, :3] if a is not None else np.zeros((0, 3))
        memo[p] = 0.0 if len(c) < 30 else float(np.minimum(np.bincount(grp(c), minlength=len(ref)) / len(c), ref).sum())
    return memo[p]


def frame_links(bank_name, bank, pose):
    """Animation neighbours. Returns (occ, nb): occ[p] = set of (prev, next) patterns around p in some animation
    (None at an end), nb[p] = {q: count}. Patterns no animation references are linked to p-1 / p+1 when those
    look like the same kind of frame (similar size and position)."""
    import anims
    occ, nb = {}, {}

    def link(a, b, w=1):
        if a is None or b is None or a == b: return
        nb.setdefault(a, {}); nb[a][b] = nb[a].get(b, 0) + w
        nb.setdefault(b, {}); nb[b][a] = nb[b].get(a, 0) + w
    try:
        an = anims.load(bank_name, bank)
    except Exception as e:  # no character data: fall back to pattern order only
        print('anims:', e); an = []
    for a in an:
        seq = []
        for p, _ in a['frames']:
            if not seq or seq[-1] != p: seq.append(p)
        if len(seq) < 2: continue
        loop = a.get('loop') and len(seq) > 2
        if loop and seq[0] == seq[-1]: seq = seq[:-1]
        for i, p in enumerate(seq):
            prev = seq[i - 1] if i > 0 else (seq[-1] if loop else None)
            nxt = seq[i + 1] if i + 1 < len(seq) else (seq[0] if loop else None)
            occ.setdefault(p, set()).add((prev, nxt))
            link(p, nxt)

    def similar(a, b):
        ra, rb = pose.get(str(a)), pose.get(str(b))
        if not ra or not rb or 'skip' in ra or 'skip' in rb: return False
        (wa, ha), (wb, hb) = ra['size'], rb['size']
        if min(ha, hb) < 40 or max(ha, hb) / min(ha, hb) > 1.3 or max(wa, wb) / max(1, min(wa, wb)) > 1.5: return False
        return abs(ra['origin'][1] - rb['origin'][1]) < 0.2 * max(ha, hb) and abs(ra['origin'][0] - rb['origin'][0]) < 0.3 * max(wa, wb)
    for p in range(bank.count):
        if p in occ: continue
        prev = p - 1 if p > 0 and similar(p, p - 1) else None
        nxt = p + 1 if p + 1 < bank.count and similar(p, p + 1) else None
        if prev is None and nxt is None: continue
        occ.setdefault(p, set()).add((prev, nxt))
        link(p, prev, 0.5); link(p, nxt, 0.5)
    return occ, nb


def bank_facing(pose):
    """+1 / -1: the way this bank's frames are drawn, by vote of the confident upright head reads."""
    vote = 0.0
    for rec in pose.values():
        if 'kp' not in rec or rec.get('rot', 0) != 0: continue
        kp, sc = np.array(rec['kp']), np.array(rec['score'])
        ears = [kp[i] for i in (LEAR, REAR) if sc[i] >= 0.5]
        if sc[NOSE] < 0.5 or not ears: continue
        d = kp[NOSE][0] - np.mean(ears, 0)[0]
        vote += np.clip(d / 4, -1, 1)
    return 1 if vote >= 0 else -1


def _ok(sc):
    return sc[[LSHO, RSHO]].max() >= THR and sc[[LHIP, RHIP]].max() >= THR


def _swap(k, s, pairs):
    for a, b in pairs:
        k[[a, b]] = k[[b, a]]; s[[a, b]] = s[[b, a]]


def _rel(k, s, pairs):
    """Joints of a limb pair relative to the midpoint of their roots (hips / shoulders), with validity."""
    (ra, rb) = pairs[0]
    roots = [k[i] for i in (ra, rb) if s[i] >= THR]
    if not roots: return None
    c = np.mean(roots, 0)
    return {i: (k[i] - c) for a, b in pairs for i in (a, b) if s[i] >= THR}


def _consistent_labels(K, nb, pairs):
    """Choose per frame whether to swap the detector's left/right labels of a limb pair so that neighbouring frames
    of an animation move as little as possible. Greedy max-confidence spanning forest with parity (union-find),
    then each connected group keeps the labelling most of its upright frames had."""
    rel = {p: _rel(k, s, pairs) for p, (k, s, _) in K.items()}
    edges = []
    for p, qs in nb.items():
        for q, w in qs.items():
            if q <= p or rel.get(p) is None or rel.get(q) is None: continue
            A, B = rel[p], rel[q]
            same = swap = 0.0; n = 0
            for a, b in pairs:
                if a in A and b in A and a in B and b in B:
                    same += np.linalg.norm(A[a] - B[a]) + np.linalg.norm(A[b] - B[b])
                    swap += np.linalg.norm(A[a] - B[b]) + np.linalg.norm(A[b] - B[a]); n += 1
            if n == 0: continue
            d = (swap - same) / n  # > 0: keep the same labels
            if abs(d) >= 1.0: edges.append((abs(d) * min(w, 3), p, q, d > 0))
    edges.sort(key=lambda e: -e[0])
    parent, par = {}, {}  # par = parity relative to parent

    def find(x):
        if parent.setdefault(x, x) == x: par.setdefault(x, 0); return x, 0
        r, pr = find(parent[x])
        par[x] ^= pr; parent[x] = r
        return r, par[x]
    for _, p, q, keep in edges:
        (rp, pp), (rq, pq) = find(p), find(q)
        if rp == rq: continue
        parent[rq] = rp; par[rq] = pp ^ pq ^ (0 if keep else 1)
    groups = {}
    for p in rel:
        if rel[p] is None: continue
        r, s = find(p)
        groups.setdefault(r, []).append((p, s))
    swapped = 0
    for r, mem in groups.items():
        vote = sum((1.0 if K[p][2] == 0 else 0.3) * (1 if s == 0 else -1) for p, s in mem)
        flip = vote < 0
        for p, s in mem:
            if s ^ flip:
                _swap(K[p][0], K[p][1], pairs); swapped += 1
    return swapped


def _body_axes(k, s, face):
    """(up, forward) unit vectors of the body in sprite space: up = hip -> shoulder; forward = the way the
    fighter faces, turned with the body (so it is still right in flips and knockdowns)."""
    sho = np.mean([k[i] for i in (LSHO, RSHO) if s[i] >= THR], 0)
    hip = np.mean([k[i] for i in (LHIP, RHIP) if s[i] >= THR], 0)
    u = sho - hip; n = np.linalg.norm(u)
    u = u / n if n > 1e-3 else np.array([0.0, -1.0])
    return u, face * np.array([-u[1], u[0]]), sho, hip, n


def _inside(mask, pts):
    """Fraction of points that land on opaque pixels."""
    if mask is None: return 1.0
    H, W = mask.shape
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    return float((mask[ys, xs] & ok).mean())


def _crosses(a, b, c, d):
    """Segments ab and cd properly intersect."""
    def o(p, q, r): return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def _leg_cost(legs, fw, mask):
    """Backward knee bend (px) + bone pixels off the silhouette (px) + legs crossing each other, for
    [(hip, knee, ankle), ...] in sprite px."""
    cost = 0.0
    if len(legs) == 2:
        (h1, k1, a1), (h2, k2, a2) = legs
        for s1 in ((h1, k1), (k1, a1)):
            for s2 in ((h2, k2), (k2, a2)):
                if _crosses(*s1, *s2): cost += 12.0
    for h, kn, a in legs:
        v = a - h; L2 = float(v @ v) or 1.0
        t = np.clip((kn - h) @ v / L2, 0, 1)
        off = (kn - (h + t * v)) @ fw
        cost += max(0.0, -off - 2.0)
        if mask is not None:
            pts = np.concatenate([h + (kn - h) * np.linspace(0, 1, 8)[:, None], kn + (a - kn) * np.linspace(0, 1, 8)[:, None]])
            ln = np.linalg.norm(kn - h) + np.linalg.norm(a - kn)
            cost += (1 - _inside(mask, pts)) * ln * 0.5
    return cost


def _repair_legs(k, s, face, mask):
    """Backward knees read off the drawing are mirrored to the front of the hip -> ankle line (sprite px)."""
    idx = (LHIP, LKNE, LANK, RHIP, RKNE, RANK)
    if s[list(idx)].min() < THR: return 0
    u, fw, *_ = _body_axes(k, s, face)

    # Swapping a knee or an ankle between the legs was tried here and dropped: SF3 stances really do cross the
    # legs (Chun-Li's idle, jump tucks) and the detector follows the pixels, so 'uncrossing' made things worse.
    fixed = 0
    # a knee still bent backwards: mirror it to the front of the hip -> ankle line if that sits on the drawing
    for h, kn, a in ((LHIP, LKNE, LANK), (RHIP, RKNE, RANK)):
        v = k[a] - k[h]; L2 = float(v @ v) or 1.0
        t = np.clip((k[kn] - k[h]) @ v / L2, 0, 1); foot = k[h] + t * v
        off = (k[kn] - foot) @ fw
        ln = np.linalg.norm(k[kn] - k[h]) + np.linalg.norm(k[a] - k[kn])
        # only a knee read off the drawing and clearly bent the wrong way: in a 3/4 stance the rear knee really
        # does point back/outward on screen, and there it sits on the trouser pixels
        if off < -max(4.0, 0.12 * ln) and _inside(mask, k[kn][None]) < 1:
            cand = k[kn] - 2 * off * fw
            one = lambda q: _leg_cost([(k[h], q, k[a])], fw, mask)
            if one(cand) < one(k[kn]) - 3:
                k[kn] = cand; fixed += 1
    return fixed


def _snap(k, s, mask, joints, reach=5):
    """Move limb joints that fall just outside the original's pixels onto the nearest opaque pixel."""
    if mask is None or not mask.any(): return 0
    inv = (~mask).astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(inv == 0)
    # label -> coordinates of that zero pixel
    lut = np.zeros((lab.max() + 1, 2)); lut[lab[ys, xs]] = np.stack([xs, ys], 1)
    H, W = mask.shape; n = 0
    for j in joints:
        if s[j] < THR: continue
        x, y = k[j]
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        if mask[yi, xi] or dist[yi, xi] > reach: continue
        k[j] = lut[lab[yi, xi]]; n += 1
    return n


def _head_from_silhouette(J, mask):
    """Head centre from the original's pixels: the top blob above the shoulders, along the body's up axis."""
    if mask is None: return None
    torso = J['hip'] - J['sho']; tl = np.linalg.norm(torso)
    if tl < 4: return None
    up = -torso / tl; side = np.array([-up[1], up[0]])
    ys, xs = np.nonzero(mask)
    rel = np.stack([xs, ys], 1) - J['sho']
    h = rel @ up; lat = rel @ side
    sel = (h > 0.05 * tl) & (h < 1.0 * tl) & (np.abs(lat) < 0.55 * tl)
    if sel.sum() < 6: return None
    top = h[sel].max()
    blob = sel & (h > top - 0.4 * tl)
    return np.array([xs[blob].mean(), ys[blob].mean()])


def _fit(mq, oq, mp, op, step=15):
    """Best rigid fit (rotation about the centroid + shift) of frame q's silhouette onto frame p's.
    Masks in sprite px, o = pattern origins. Returns (IoU, T) with T a 2x3 matrix taking q's anchor coordinates
    to p's, so a neighbour's joints can be carried over to a frame the detector failed on (curled rolls,
    somersaults, recoveries)."""
    yq, xq = np.nonzero(mq); yp, xp = np.nonzero(mp)
    if len(xq) < 20 or len(xp) < 20: return 0.0, None
    area = max(len(xq), len(xp)) / min(len(xq), len(xp))
    if area > 1.8: return 0.0, None
    cq = np.array([xq.mean(), yq.mean()]); cp = np.array([xp.mean(), yp.mean()])
    H, W = mp.shape
    src = mq.astype(np.uint8)
    npix = float(len(xp))

    def score(th):
        c, s_ = math.cos(th), math.sin(th)
        Rm = np.array([[c, -s_], [s_, c]])
        t = cp - Rm @ cq
        w = cv2.warpAffine(src, np.hstack([Rm, t[:, None]]), (W, H), flags=cv2.INTER_NEAREST) > 0
        inter = float((w & mp).sum())
        return inter / (npix + len(xq) - inter), Rm
    # neighbouring frames turn a little at a time: big turns must earn it (a lying body matches itself upside
    # down almost as well, which would put the head on the feet)
    pen = lambda th: 0.2 * abs(th) / math.pi
    best = (-9, 0.0)
    for th in np.radians(np.arange(-180, 180, step)):
        r, _ = score(th)
        if r - pen(th) > best[0]: best = (r - pen(th), th)
    for th in best[1] + np.radians(np.arange(-step + 5, step, 5)):
        r, _ = score(th)
        if r - pen(th) > best[0]: best = (r - pen(th), th)
    r, Rm = score(best[1])
    # sprite px of q -> sprite px of p:  x_p = Rm (x_q - cq) + cp ; anchor coords add the origins
    A = Rm; tq = np.asarray(oq, float); tp = np.asarray(op, float)
    t = cp + tp - A @ (cq + tq)
    return r, np.hstack([A, t[:, None]])


def _anim_seqs(bank_name, bank):
    """[(name, frames without immediate repeats)] for every animation of the bank."""
    import anims
    try:
        an = anims.load(bank_name, bank)
    except Exception:
        return []
    out = []
    for a in an:
        seq = []
        for p, _ in a['frames']:
            if not seq or seq[-1] != p: seq.append(p)
        if seq: out.append((a['name'], seq))
    return out


def _face_vote(k, s):
    """+1 / -1 when this read clearly faces screen-right / screen-left (nose ahead of the ears, measured across the
    body's own up axis so lying and flipping reads count too), else 0."""
    ears = [k[i] for i in (LEAR, REAR) if s[i] >= 0.5]
    if s[NOSE] < 0.5 or not ears: return 0
    u = _body_axes(k, s, 1)[0] if _ok(s) else np.array([0.0, -1.0])
    d = (k[NOSE] - np.mean(ears, 0)) @ np.array([-u[1], u[0]])
    return int(np.sign(d)) if abs(d) >= 1.5 else 0


def _turned_runs(seqs, votes, face):
    """Frames drawn facing away from the bank's facing: within each animation, a run with at least two clear
    opposite reads and no clear agreeing read (frames without a clear read inside the run go along)."""
    out = set()

    def close(run):
        opp = [q for q in run if votes.get(q, 0) == -face]
        if len(opp) >= 2:
            i0 = run.index(opp[0]); i1 = len(run) - 1 - run[::-1].index(opp[-1])
            out.update(run[i0:i1 + 1])
    for _, seq in seqs:
        run = []
        for q in seq:
            if votes.get(q, 0) == face: close(run); run = []
            else: run.append(q)
        close(run)
    return out


def _kick_leg(k, s, arm_len):
    """A raised leg read as an arm: both leg chains sit on the standing leg (knees and ankles together) while an arm
    chain is as long as a leg (> 1.4x the bank's arm), or reaches down to hip height or below. That arm chain becomes
    the free leg; the arm is left to be filled."""
    need = [LHIP, RHIP, LKNE, RKNE, LANK, RANK]
    if s[need].min() < THR or not _ok(s): return False
    ll = (np.linalg.norm(k[LKNE] - k[LHIP]) + np.linalg.norm(k[LANK] - k[LKNE]) +
          np.linalg.norm(k[RKNE] - k[RHIP]) + np.linalg.norm(k[RANK] - k[RKNE])) / 2
    if ll < 8 or np.linalg.norm(k[LKNE] - k[RKNE]) > 0.2 * ll or np.linalg.norm(k[LANK] - k[RANK]) > 0.2 * ll:
        return False
    u, _, sho, hip, tl = _body_axes(k, s, 1)
    best = None
    for so, el, wr in ((LSHO, LELB, LWRI), (RSHO, RELB, RWRI)):
        if min(s[so], s[el], s[wr]) < THR: continue
        ln = np.linalg.norm(k[el] - k[so]) + np.linalg.norm(k[wr] - k[el])
        low = (k[wr] - sho) @ u < -0.6 * tl and ln > 1.1 * arm_len
        if not (ln > 1.4 * arm_len or low): continue
        reach = np.linalg.norm(k[wr] - k[LANK])
        if best is None or reach > best[0]: best = (reach, el, wr)
    if best is None: return False
    _, el, wr = best
    # the leg chain the detector was less sure of gives way
    kn, an = (LKNE, LANK) if s[[LKNE, LANK]].sum() <= s[[RKNE, RANK]].sum() else (RKNE, RANK)
    k[kn], k[an] = k[el].copy(), k[wr].copy()
    s[kn], s[an] = s[el], s[wr]
    s[el] = s[wr] = 0.0
    return True


def core_poses(bank_name, bank, pose, G, cfg):
    """{pattern: J} for every frame of the bank that can be posed, after the shared temporal clean-up (sprite px).
    Cached per (bank, pose object, G); hooks receive copies."""
    key = (bank_name, id(pose), round(float(G), 4), tuple(sorted((k, str(v)) for k, v in cfg.items() if k in CORE_DEFAULTS)))
    if key in _core_cache: return _core_cache[key]
    C = dict(CORE_DEFAULTS); C.update({k: v for k, v in cfg.items() if k in CORE_DEFAULTS})
    masks = {}
    face = int(cfg.get('face', bank_facing(pose)))
    keep = set(C['keep_frames'] or ())
    K = {}  # p -> [kp anchor coords, scores, rot]
    for p in range(bank.count):
        rec = pose.get(str(p))
        if not rec or 'kp' not in rec: continue
        K[p] = [np.array(rec['kp'], float) + rec['origin'], np.array(rec['score'], float), rec.get('rot', 0)]
    occ, nb = frame_links(bank_name, bank, pose) if C['temporal'] else ({}, {})
    seqs = _anim_seqs(bank_name, bank)
    org = lambda p: np.array(pose[str(p)]['origin'], float)
    stats = dict(swap_legs=0, swap_arms=0, leg_fix=0, filled_joints=0, filled_frames=0, refitted=0, snapped=0,
                 rejected=0, effects=0, turned=0, kick_legs=0, xray=0, head_moved=0, no_ground=0)

    # pure effects (sparks, lightning, spirals, trails) are not posed: thin or broken-up shapes, or colours unlike the
    # fighter's idle drawing on a frame the detector isn't sure is a body
    effects = set()
    if C['effect_gate']:
        for p in range(bank.count):
            if p in keep: continue
            rec = pose.get(str(p))
            if not rec or 'skip' in rec: continue
            m = _alpha(bank_name, p, masks)
            if m is None or not m.any(): continue
            mean = float(np.mean(rec['score'])) if 'score' in rec else 0.0
            colour_off = (body_likeness(bank_name, bank, p) < 0.45) if main_part(bank, p) == 0 else (mean < 0.4)
            if effect_shape(m) or (mean < 0.53 and colour_off):
                effects.add(p)
    # shock x-rays: a damage animation a, b, a, c... draws one pose; b and c take a's
    xray = {}
    for name, seq in seqs:
        if name == 'dmca' and len(seq) >= 4 and len(set(seq[0::2])) == 1 and 2 <= len(set(seq)) <= 4:
            for q in seq[1::2]:
                if q != seq[0]: xray[q] = seq[0]
    for q in xray:
        effects.discard(q)
    stats['effects'] = len(effects)

    arms = [np.linalg.norm(k[e] - k[a]) + np.linalg.norm(k[w] - k[e]) for k, s, _ in K.values()
            for a, e, w in ((LSHO, LELB, LWRI), (RSHO, RELB, RWRI)) if min(s[a], s[e], s[w]) > 0.6]
    arm_len = float(np.median(arms)) if len(arms) > 20 else 1e9
    for p in list(K):
        if p in effects: del K[p]; continue
        if _kick_leg(K[p][0], K[p][1], arm_len): stats['kick_legs'] += 1

    # facing: the bank's drawing direction, except runs of an animation clearly drawn facing the other way
    faces = {}
    if C['face_mode'] == 'bank' and C['turned_runs']:
        votes = {p: _face_vote(k, s) for p, (k, s, _) in K.items()}
        # pattern order counts too (many turned frames aren't reached by the parsed scripts), split where the
        # drawing changes size or place
        order, run = [], []
        for p in range(bank.count):
            ra, rb = pose.get(str(p - 1)), pose.get(str(p))
            same = bool(ra and rb and 'skip' not in ra and 'skip' not in rb and
                        max(ra['size'][1], rb['size'][1]) <= 1.3 * min(ra['size'][1], rb['size'][1]) and
                        abs(ra['origin'][1] - rb['origin'][1]) < 0.25 * max(ra['size'][1], rb['size'][1]))
            if not same and run: order.append(('order', run)); run = []
            run.append(p)
        if run: order.append(('order', run))
        for p in _turned_runs(seqs + order, votes, face):
            faces[p] = -face
        stats['turned'] = len(faces)
    face_of = lambda p: faces.get(p, face)

    # torso read upside down against both animation neighbours (which agree with each other): the read is wrong
    if C['temporal']:
        def axis(p):
            k, s, _ = K[p]
            if not _ok(s): return None
            return _body_axes(k, s, 1)[0]
        for p, oc in occ.items():
            if p not in K or axis(p) is None: continue
            for prev, nxt in oc:
                if prev in K and nxt in K and axis(prev) is not None and axis(nxt) is not None:
                    a, b, c = axis(prev), axis(nxt), axis(p)
                    if a @ b > 0.5 and c @ a < -0.2 and c @ b < -0.2:
                        K[p][1] = K[p][1] * 0; stats['rejected'] += 1; break
        # labels only travel between frames drawn facing the same way (a turned run shows the other side)
        nb_same = {p: {q: w for q, w in qs.items() if face_of(q) == face_of(p)} for p, qs in nb.items()}
        stats['swap_legs'] = _consistent_labels(K, nb_same, LEG_PAIRS)
        stats['swap_arms'] = _consistent_labels(K, nb_same, ARM_PAIRS)

    if C['leg_repair']:
        for p, (k, s, _) in K.items():
            if not _ok(s): continue
            kk = k - org(p)
            n = _repair_legs(kk, s, face_of(p), _alpha(bank_name, p, masks))
            if n: K[p][0] = kk + org(p); stats['leg_fix'] += n

    good = {p for p, (k, s, _) in K.items() if _ok(s) and s.mean() >= 0.33}
    # weak reads: a torso but a low average. Keep their confident joints, the rest comes from the neighbours
    weak = {p for p, (k, s, _) in K.items() if _ok(s) and p not in good and p in occ}
    mean0 = {p: float(K[p][1].mean()) for p in K}
    # shaky reads (rolls, handstands, curled frames read from a rotated copy): a neighbour that was read well and fits
    # the silhouette closely (rotation allowed) gives a better pose; tried in the fill below
    def tumbling(p):  # read from a rotated copy, or a torso far from upright
        k, s, rot = K[p]
        return rot != 0 or _body_axes(k, s, 1)[0][1] > -0.5
    shaky = {p for p in good | weak if mean0[p] < 0.45 and p in occ and tumbling(p)} if C['fill_frames'] else set()

    def estimate(p, j, pool):
        """Neighbour estimate of joint j for frame p: midpoint of prev/next where both have it, else the one."""
        pairs, singles = [], []
        for prev, nxt in occ.get(p, ()):
            a = prev if prev in pool and K[prev][1][j] >= THR else None
            b = nxt if nxt in pool and K[nxt][1][j] >= THR else None
            if a is not None and b is not None: pairs.append((K[a][0][j] + K[b][0][j]) / 2)
            elif a is not None: singles.append(K[a][0][j])
            elif b is not None: singles.append(K[b][0][j])
        if pairs: return np.mean(pairs, 0), True
        if singles: return np.mean(singles, 0), False
        return None, False

    if C['temporal']:
        new = {}
        for p in good | weak:
            k, s, _ = K[p]; k2, s2 = k.copy(), s.copy()
            for j in range(17):
                thr = 0.5 if p in weak else THR
                e, both = estimate(p, j, good)
                if s[j] < thr:
                    if e is not None:
                        k2[j] = e; s2[j] = max(THR + 0.01, s[j]); stats['filled_joints'] += 1
                    elif p in weak:
                        s2[j] = min(s[j], THR - 0.01)
                    continue
                if e is None or not both: continue
                d = np.linalg.norm(k[j] - e)
                if d < 3.0 and C['smooth']:
                    k2[j] = k[j] + C['smooth'] * (e - k[j])
            new[p] = (k2, s2)
        for p, (k2, s2) in new.items():
            K[p][0], K[p][1] = k2, s2
        good |= weak

    def lands_inside(kk, s, mp):
        """Shoulders and head of a carried-over pose (sprite px) must land within 6 px of the target's pixels
        (snapping brings them onto it). Ibuki's 69 put them 9+ px above the frame: a pose that misses by up to
        14 px is moved over onto the drawing; returns the (possibly moved) joints, or None to reject the source."""
        pad = 16
        inv = (~np.pad(mp, pad)).astype(np.uint8)
        dist, lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(inv == 0)
        lut = np.zeros((lab.max() + 1, 2)); lut[lab[ys, xs]] = np.stack([xs, ys], 1)
        idx = [i for i in (LSHO, RSHO) if s[i] >= THR]
        hp = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if s[i] >= THR]

        def worst(k):
            pts = [k[i] for i in idx] + ([np.mean([k[i] for i in hp], 0)] if hp else [])
            out = (0.0, None)
            for q in pts:
                x, y = int(round(q[0])) + pad, int(round(q[1])) + pad
                if not (0 <= x < dist.shape[1] and 0 <= y < dist.shape[0]): return (1e9, None)
                if dist[y, x] > out[0]: out = (float(dist[y, x]), lut[lab[y, x]] - pad - np.array([x - pad, y - pad]))
            return out
        d, v = worst(kk)
        if d <= 6: return kk
        if d > 14 or v is None: return None
        moved = kk + v * (1 + 2 / max(d, 1e-3))
        return moved if worst(moved)[0] <= 6 else None

    filled, refit = set(), set()
    if C['temporal'] and C['fill_frames']:
        for it in range(4):
            added = {}
            for p in occ:
                if p in effects or p in added: continue
                if p in good and not (it == 0 and p in shaky): continue
                rec = pose.get(str(p))
                if not rec or 'skip' in rec or rec['size'][1] < 32 or rec['size'][0] < 16: continue
                mp = _alpha(bank_name, p, masks)
                if mp is None or mp.sum() < 200: continue
                cands = []
                for prev, nxt in occ[p]:
                    for q in (prev, nxt):
                        if q is None or q not in good or q in [c[0] for c in cands]: continue
                        # a shaky read is replaced only from a frame that was read well
                        if p in shaky and (q in shaky or mean0.get(q, 0) < 0.5): continue
                        mq = _alpha(bank_name, q, masks)
                        if mq is None: continue
                        r, T = _fit(mq, org(q), mp, org(p))
                        # a frame filled earlier passes its pose on only when the shapes match closely; a shaky
                        # read is only replaced by a close fit
                        need = 0.6 if p in shaky else (0.62 if q in filled else 0.5)
                        if r < need: continue
                        kq = K[q][0] @ T[:, :2].T + T[:, 2]
                        kk = lands_inside(kq - org(p), K[q][1], mp)
                        if kk is None: continue
                        cands.append((q, r, kk + org(p)))
                if not cands: continue
                best = max(c[1] for c in cands)
                use = [c for c in cands if c[1] >= best - 0.08]
                k = np.mean([kq for _, _, kq in use], 0)
                s = np.min([K[q][1] for q, _, _ in use], 0)
                if not _ok(s): continue
                own = K.get(p)
                if own is not None:  # the frame's own confident joints win when they agree with the guess
                    ok = (own[1] >= (0.6 if p in shaky else 0.5)) & (np.linalg.norm(own[0] - k, axis=1) < 10)
                    k[ok] = own[0][ok]
                added[p] = [k, s, own[2] if own is not None else 0]
            if not added: break
            for p, v in added.items():
                if p in good: stats['refitted'] += 1; refit.add(p)
                else: stats['filled_frames'] += 1
                K[p] = v; good.add(p); filled.add(p)

    for q, a in xray.items():  # after the fill, so a filled 'a' also counts
        if a in good:
            K[q] = [K[a][0].copy(), K[a][1].copy(), K[a][2]]; good.add(q); filled.add(q); stats['xray'] += 1

    # typical shoulder -> head-centre offset of confident upright reads, in torso units along (up, forward)
    necks, offs = [], []
    for p in good - filled:
        k, s, _ = K[p]
        hp = [k[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if s[i] >= 0.5]
        if len(hp) >= 3 and s[[LSHO, RSHO]].min() >= 0.5:
            sho = (k[LSHO] + k[RSHO]) / 2; h = np.mean(hp, 0)
            necks.append(np.linalg.norm(h - sho))
            if _ok(s):
                u, fw, _, _, tl = _body_axes(k, s, face_of(p))
                if u[1] < -0.77 and tl > 4: offs.append(((h - sho) @ u / tl, (h - sho) @ fw / tl))
    neck = float(np.median(necks)) if len(necks) > 20 else None
    hoff = np.median(offs, 0) if len(offs) > 20 else None
    sole, ah = _sole_offset(bank_name, bank, pose)
    out = {}
    for p in sorted(good):
        k, s, _ = K[p]
        kk = k - org(p)
        mask = _alpha(bank_name, p, masks)
        if C['snap']:
            # a pose carried over from a neighbour may stick out of a curled frame: pull it all the way in
            far = p in filled
            stats['snapped'] += _snap(kk, s, mask, LIMB_JOINTS + ((NOSE, LEYE, REYE, LEAR, REAR) if far else ()),
                                      reach=30 if far else 5)
        J = _joints(kk, s, face_of(p))
        if J is not None and C['face_mode'] == 'detect':
            J['face'] = _detect_face(kk, s, J)
        if J is None: continue
        if C['head_fix'] and hoff is not None and _ok(s):
            u, fw, _, _, tl = _body_axes(kk, s, J['face'])
            if u[1] < -0.77 and tl > 4:  # upright: torso within ~40 degrees of vertical
                prior = J['sho'] + (hoff[0] * u + hoff[1] * fw) * tl
                h = J['head']
                if h is None or (h - J['sho']) @ fw < -0.1 * tl:
                    # read on the trapezius / back of the neck (behind the shoulder centre): the top blob above the
                    # shoulders when it is where a head goes, else the bank's usual offset
                    blob = _head_from_silhouette(J, mask)
                    J['head'] = blob if blob is not None and np.linalg.norm(blob - prior) < 0.5 * tl else prior
                    stats['head_moved'] += 1
                elif np.linalg.norm(h - prior) > 0.6 * tl:
                    # read far from where a head goes (on a glove in a guard): the top blob above the shoulders, if
                    # it is the more head-like place
                    blob = _head_from_silhouette(J, mask)
                    if blob is not None and np.linalg.norm(blob - prior) < np.linalg.norm(h - prior) - 0.2 * tl:
                        J['head'] = blob; stats['head_moved'] += 1
        if J['head'] is None:
            J['head'] = _head_from_silhouette(J, mask)
        if J['head'] is not None and neck:
            nk = J['head'] - J['sho']; n = np.linalg.norm(nk)
            if n > 1.25 * neck: J['head'] = J['sho'] + nk / n * 1.25 * neck
        J['ground'] = float(-pose[str(p)]['origin'][1] + sole); J['ankle_h'] = ah
        if mask is not None and mask.any() and J['ground'] < np.nonzero(mask.any(1))[0].max() - 6:
            # unusual anchors (throws, specials that move the anchor): the floor line sits far above the drawing
            J['ground'] = None; stats['no_ground'] += 1
        if p in filled: J['_interp'] = True
        if p in refit: J['_refit'] = True
        if p in xray: J['_borrowed'] = xray[p]
        out[p] = J
    _core_cache[key] = (out, stats)
    return out, stats


def _detect_face(kp, sc, J):
    """The old per-frame facing (nose vs ears, else head vs shoulders) for face_mode='detect'."""
    ears = [kp[i] for i in (LEAR, REAR) if sc[i] >= THR]
    if sc[NOSE] >= THR and ears: return 1 if kp[NOSE][0] >= np.mean(ears, 0)[0] else -1
    if J['head'] is not None: return 1 if J['head'][0] >= J['sho'][0] else -1
    return 1


def _joints(kp, sc, face):
    sho = mid(pt(kp, sc, LSHO), pt(kp, sc, RSHO)); hip = mid(pt(kp, sc, LHIP), pt(kp, sc, RHIP))
    if sho is None or hip is None: return None
    headpts = [kp[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if sc[i] >= THR]
    J = dict(sho=sho, hip=hip, head=np.mean(headpts, 0) if headpts else None, face=face)
    for side, (s, e, w, h, k, a) in dict(l=(LSHO, LELB, LWRI, LHIP, LKNE, LANK),
                                         r=(RSHO, RELB, RWRI, RHIP, RKNE, RANK)).items():
        J[side] = {n: pt(kp, sc, i) for n, i in dict(sho=s, elb=e, wri=w, hip=h, kne=k, ank=a).items()}
    return J


_sole_cache = {}


def _sole_offset(bank_name, bank, pose):
    """(sole, ankle_h): pixels from a frame's anchor line (y = 0) down to the soles of the idle stance (SF3 anchors
    sit a few px up), and how high the detected ankles sit above those soles when standing."""
    if bank_name in _sole_cache: return _sole_cache[bank_name]
    try:
        frames = bank.standing_patterns(8)
    except Exception:
        frames = []
    offs, ank = [], []
    for p in frames:
        f = MOD / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
        rec = pose.get(str(p), {})
        if not f.exists() or 'origin' not in rec: continue
        a = np.asarray(Image.open(f).convert('RGBA'))[:, :, 3] > 0
        ys = np.nonzero(a.any(1))[0]
        if not len(ys): continue
        sole = ys.max() + 1 + rec['origin'][1]; offs.append(sole)
        if 'kp' in rec:
            for i in (LANK, RANK):
                if rec['score'][i] >= 0.5: ank.append(ys.max() + 1 - rec['kp'][i][1])
    sole = float(np.median(offs)) if offs else 4.0
    ah = float(np.clip(np.median(ank), 4, 30)) if ank else 10.0
    _sole_cache[bank_name] = (sole, ah)
    return _sole_cache[bank_name]


def copy_J(J):
    out = {}
    for k, v in J.items():
        if isinstance(v, dict): out[k] = {kk: (None if vv is None else np.array(vv, float)) for kk, vv in v.items()}
        elif isinstance(v, np.ndarray): out[k] = v.copy()
        else: out[k] = v
    return out


M = 32  # free margin around the fighter's original canvas (frames get their own tiles, so nothing is clipped)


def _span(part, y, flip=False):
    """Opaque x-extent (part px) of a part's art at row y (clamped into the art), cached per (unflipped) part;
    flip=True gives the extent in the mirrored part."""
    if flip:
        lo, hi = _span(part, y)
        w = part.im.width - 1
        return w - hi, w - lo
    key = id(part)
    if key not in _span_cache:
        al = np.asarray(part.im)[:, :, 3] > 0
        rows = []
        for r in range(al.shape[0]):
            xs = np.nonzero(al[r])[0]
            rows.append((float(xs.min()), float(xs.max())) if len(xs) else None)
        filled = [r for r in rows if r is not None]
        last = filled[0] if filled else (0.0, float(al.shape[1]))
        for i, r in enumerate(rows):  # empty rows take the nearest filled row above
            if r is None: rows[i] = last
            else: last = r
        _span_cache[key] = rows
    rows = _span_cache[key]
    return rows[int(np.clip(round(y), 0, len(rows) - 1))]


_span_cache = {}


def _parked(J, far=1000.0):
    """A hook parked the body off the canvas to draw a loose limb on its own (pl18 leg_only): the mesh body can't
    leave its torso out, so such frames keep the cut-out puppet."""
    for k in ('sho', 'hip'):
        v = J.get(k)
        if v is not None and np.abs(np.asarray(v, float)).max() > far: return True
    return False


def compose(J, parts, G, cfg, size):
    """Draw the puppet for one frame onto a (size + 2M) * R canvas. G = part px -> canvas px."""
    W, H = size
    cv = Image.new('RGBA', ((W + 2 * M) * R, (H + 2 * M) * R))
    f = J['face']
    P = {k: (v if f > 0 else v.flipped()) for k, v in parts.items()}
    S = lambda p: (np.asarray(p) + M) * R
    if J.get('_piece'):
        return compose_piece(cv, J['_piece'], P, G, cfg, S)
    if getattr(parts, 'mesh', None) is not None and cfg.get('renderer', 'mesh') == 'mesh' and not _parked(J):
        import mesh
        return mesh.compose_mesh(J, parts, G, cfg, size, R, M, paste_affine, paste_head)
    J = dict(J); J['l'] = dict(J['l']); J['r'] = dict(J['r'])  # joints may be adjusted below; the caller's J stays
    # The side nearer the camera is drawn in front: a fighter facing right shows his right side (COCO 'r'), one
    # facing left his left side (the detector labels sprites that way too: Ryu's rear fist across his chest is 'l').
    # cfg near_rule='legacy' restores the old inverted choice; a hook may force J['near'].
    if J.get('near') in ('l', 'r'): near = J['near']
    elif cfg.get('near_rule', 'physical') == 'legacy': near = 'l' if f > 0 else 'r'
    else: near = 'r' if f > 0 else 'l'
    far = 'r' if near == 'l' else 'l'
    sho, hip = S(J['sho']), S(J['hip'])
    tor = hip - sho; tang = math.atan2(tor[0], tor[1])
    gw = G * cfg['limb_w']

    def arm(side, shade, upper=True, lower=True):
        L = J[side]
        if upper: paste_bone(cv, P['upperarm'], S(L['sho']), S(L['elb']), gw, shade=shade)
        if not lower: return
        paste_bone(cv, P['forearm'], S(L['elb']), S(L['wri']), gw, shade=shade)
        a = S(L['wri']) - S(L['elb']); ang = math.atan2(a[0], a[1])
        paste_affine(cv, P['fist'], S(L['wri']), ang, G * cfg['fist'], G * cfg['fist'], shade)

    ground = J.get('ground') if cfg.get('plant', True) and J.get('plant', True) else None
    foot = P['foot']; fs = G * cfg['foot']
    sole = (foot.im.height - foot.a[1]) * fs / R  # ankle pivot -> bottom of the foot part, sprite px

    def leg(side, shade, upper=True, lower=True):
        L = J[side]
        kne, ank = np.asarray(L['kne'], float), np.asarray(L['ank'], float)
        s = ank - kne; ang = math.atan2(s[0], s[1])
        if ground is not None:
            # A foot whose ankle is as low as the original's standing ankles stands flat on the original's floor
            # line (sole on the line, toes the way the fighter faces); lifting off, it turns with the shin, fully
            # once the ankle is about a foot's height up.
            ah = J.get('ankle_h', 10.0)
            up = ground - ank[1] - ah  # how far this ankle is above a standing ankle
            w = float(np.clip((up - 3) / max(6.0, ah), 0, 1))
            ang = ang * w
            ank = ank + np.array([0.0, (ground - sole - ank[1]) * (1 - w)])
        if upper: paste_bone(cv, P['thigh'], S(L['hip']), S(kne), gw, shade=shade)
        if not lower: return
        paste_bone(cv, P['shin'], S(kne), S(ank), gw, shade=shade)
        paste_affine(cv, foot, S(ank), ang, fs, fs, shade)

    # cape and tail hang behind everything (far leg included); cfg cape / tail scale them (default 0.75)
    cs, ts = G * cfg.get('cape', 0.75), G * cfg.get('tail', 0.75)
    if 'cape' in P: paste_affine(cv, P['cape'], sho, tang * 0.6 - f * 0.25, cs, cs, 0.9)
    if J.get('tail_scale') is not None: ts = G * J['tail_scale']  # a hook may size / turn the tail per frame
    t_ang = J['tail_angle'] if J.get('tail_angle') is not None else tang - f * 0.2
    if 'tail' in P: paste_affine(cv, P['tail'], hip - np.array([f * 6 * R, 0]), t_ang, ts, ts)
    # Body connectivity: every limb root must sit on the art it hangs from. Shoulder joints outside the torso art
    # are pulled in onto it; the pelvis is stretched across the hips so both thigh tops start under it.
    tp = P['torso']; tlen = np.linalg.norm(hip - sho) or 1.0
    t_sy = max(tlen / (np.linalg.norm(tp.b - tp.a) or 1.0), G * 0.6)
    ex = np.array([math.cos(tang), -math.sin(tang)]); ey = np.array([math.sin(tang), math.cos(tang)])
    arm_half = P['upperarm'].width * gw / 2
    parked = lambda q, c: np.linalg.norm(q - c) > 4 * tlen + 40 * R  # a hook parks unused parts far off the canvas
    for side in 'lr':
        # along the spine too: a shoulder no more than 0.08 torso lengths above the shoulder centre or 0.3 below it,
        # a hip within 0.15 of the hip centre (group D's connect())
        for key, c, lo_, hi_ in (('sho', sho, -0.08, 0.3), ('hip', hip, -0.15, 0.15)):
            if J[side].get(key) is None: continue
            q = S(J[side][key])
            if parked(q, c): continue
            v = (q - c) @ ey; vv = float(np.clip(v, lo_ * tlen, hi_ * tlen))
            if vv != v: J[side][key] = (q + ey * (vv - v)) / R - M
        q = S(J[side]['sho'])
        if parked(q, sho): continue
        d = (q - sho) @ ex; v = (q - sho) @ ey
        lo, hi = _span(parts['torso'], tp.a[1] + v / t_sy, f < 0)
        lim_hi = (hi - tp.a[0]) * G - 0.35 * arm_half; lim_lo = (tp.a[0] - lo) * G - 0.35 * arm_half
        dd = float(np.clip(d, -max(lim_lo, 0.0), max(lim_hi, 0.0)))
        if dd != d: J[side]['sho'] = (q + ex * (dd - d)) / R - M
    pp = P['pelvis']; lo, hi = _span(parts['pelvis'], pp.a[1], f < 0)
    thigh_half = P['thigh'].width * gw / 2
    p_sx = G
    for side in 'lr':
        if parked(S(J[side]['hip']), hip): continue
        d = (S(J[side]['hip']) - hip) @ ex
        ext = (hi - pp.a[0]) if d >= 0 else (pp.a[0] - lo)
        if ext > 0: p_sx = max(p_sx, (abs(d) + 0.5 * thigh_half) / ext)
    p_sx = min(p_sx, 3 * G)
    arm(far, 0.72); leg(far, 0.72)
    paste_bone(cv, P['torso'], sho, hip, G, min_len=0.6)
    leg(near, 1.0, lower=False)  # the near thigh's top goes under the pelvis
    paste_affine(cv, P['pelvis'], hip, tang, p_sx, G)
    leg(near, 1.0, upper=False)
    head = S(J['head']); nk = head - sho; hs = G * cfg['head']
    # a cat head carries its ears above the skull: sink it a little along the neck so they stay in the tiles
    head = head - nk / (np.linalg.norm(nk) or 1) * P['head'].im.height * hs * cfg.get('head_sink', 0.12)
    # and it never floats off the shoulders: its centre stays within ~0.6 head sizes of the shoulder centre (an
    # idle head sits at 0.3-0.5)
    hn = head - sho; hl = np.linalg.norm(hn); hr = 0.6 * min(P['head'].im.width, P['head'].im.height) * hs
    if hl > hr: head = sho + hn / hl * hr
    # near upper arm under the head, forearm + fist over it: a guard shows fists in front of the face,
    # never a sleeve across it
    arm(near, 1.0, lower=False)
    paste_head(cv, P['head'], head, nk, hs)
    arm(near, 1.0, upper=False)
    return cv


def compose_piece(cv, pc, P, G, cfg, S):
    """A loose body piece: the mascot head over a head-only drawing, forearm + fist along a loose arm, head over a
    torso for a close-up bust. Sized to the drawing (close-ups are bigger than the in-game scale)."""
    hp = P['head']; c = np.asarray(pc['c'], float)
    if pc['kind'] == 'leg':  # a loose leg: thigh from the hip end, shin, foot at the far end
        a, b = np.asarray(pc['a'], float), np.asarray(pc['b'], float)
        sw = pc['thick'] * R / max(1.0, P['thigh'].width)
        d = b - a; n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
        kne = a + d * 0.5 + n * 0.05 * np.linalg.norm(d); ank = a + d * 0.85
        paste_bone(cv, P['thigh'], S(a), S(kne), sw)
        paste_bone(cv, P['shin'], S(kne), S(ank), sw)
        s_ = S(ank) - S(kne); fs = sw * cfg['foot'] / cfg['limb_w']
        paste_affine(cv, P['foot'], S(ank), math.atan2(s_[0], s_[1]), fs, fs)
        return cv
    if pc['kind'] == 'arm':
        fa, fist = P['forearm'], P['fist']
        a, b = np.asarray(pc['a'], float), np.asarray(pc['b'], float)
        sw = pc['thick'] * R / max(1.0, fa.width)
        d = b - a; L = np.linalg.norm(d) or 1.0
        wri = a + d * 0.8
        paste_bone(cv, fa, S(a), S(wri), sw)
        ang = math.atan2(d[0], d[1])
        paste_affine(cv, fist, S(wri), ang, sw * cfg['fist'] / cfg['limb_w'], sw * cfg['fist'] / cfg['limb_w'])
        return cv
    if pc['kind'] == 'bust':
        h = np.asarray(pc['head'], float)
        tor = P['torso']; tl = np.linalg.norm(tor.b - tor.a)
        st = pc['w'] * 0.8 * R / max(1.0, tor.width)
        sho = h + np.array([0.0, pc['h'] * 0.12])  # the torso's top tucks under the head
        paste_bone(cv, tor, S(sho), S(sho + np.array([0.0, tl * st / R])), st, min_len=0.6)
        hs = pc['w'] * 0.7 * R / hp.im.width
        paste_head(cv, hp, S(h), np.array([0.0, -1.0]), hs)
        return cv
    # the head fills the drawing it replaces (an icon, a head poking out of a cocoon)
    hs = min(pc['w'] * R / hp.im.width, pc['h'] * R / hp.im.height) * 1.05
    paste_head(cv, hp, S(c), np.array([0.0, -1.0]), hs)
    return cv


def paste_head(cv, hp, centre, nk, hs):
    """Head: the part's centre (joint a) on the detected head centre, tilted along shoulder->head (nk).

    The part's own 'down' (towards the neck) is mapped onto -nk. Small tilts are damped so an upright fighter
    keeps an upright head; flips and knockdowns rotate it fully."""
    n = np.linalg.norm(nk)
    ang = math.atan2(-nk[0], -nk[1]) if n else 0.0  # 0 when the head is straight above the shoulders
    # tilting a tiny face reads as a broken neck: upright while the neck leans less than ~50 degrees, then ease
    # into the full tilt by ~85 degrees so flips and knockdowns turn the head without a one-frame snap
    ang *= float(np.clip((abs(ang) - 0.9) / 0.6, 0, 1))
    c, s = math.cos(ang), math.sin(ang)
    M = np.array([[c * hs, s * hs], [-s * hs, c * hs]])
    off = np.asarray(centre, float) - M @ hp.a
    _paste_xform(cv, hp.im, M, off)


def global_scale(pose, cfg, parts):
    """Part px -> canvas px so the puppet torso spans the fighter's typical shoulder-hip distance
    (70th percentile over all confidently read frames, so crouches and misses don't skew it)."""
    lens = []
    for rec in pose.values():
        if 'kp' not in rec: continue
        sc = np.array(rec['score']); kp = np.array(rec['kp'])
        if sc[[LSHO, RSHO, LHIP, RHIP]].min() < 0.5: continue
        lens.append(np.linalg.norm((kp[LHIP] + kp[RHIP]) / 2 - (kp[LSHO] + kp[RSHO]) / 2))
    t = parts['torso']
    return np.percentile(lens, 70) * R / np.linalg.norm(t.b - t.a)


def shrink(cv, size):
    small = cv.resize(size, Image.LANCZOS)
    a = np.asarray(small).copy()
    a[:, :, 3] = np.where(a[:, :, 3] >= 110, 255, 0)
    a[a[:, :, 3] == 0] = 0
    return a


def make_palette(images):
    """64 colours in 4 banks of 16 (the cell codec packs 2 pixels per byte while they stay in one bank).
    Colours are first split into 4 material groups, then each group gets its own shades; slot 0 of every bank
    is the outline colour (bank 0's slot 0 is transparent, its outline is slot 1)."""
    px = np.concatenate([im.reshape(-1, 4)[im.reshape(-1, 4)[:, 3] > 0][:, :3] for im in images]).astype(np.float32)
    if len(px) > 300000: px = px[np.random.default_rng(0).choice(len(px), 300000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    body = px[px.max(1) >= 40]
    # group by chroma + brightness so one material's ramp lands in one bank
    hsv = cv2.cvtColor(body.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    ang = hsv[:, 0] / 90 * np.pi
    feat = np.stack([np.cos(ang) * hsv[:, 1], np.sin(ang) * hsv[:, 1], hsv[:, 2] * 0.35], 1).astype(np.float32)
    _, grp, _ = cv2.kmeans(feat, 4, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    grp = grp.ravel()
    pal = np.zeros((64, 4), np.uint8)
    outline = (20, 16, 28)
    order = np.argsort([-(grp == g).sum() for g in range(4)])
    for b, g in enumerate(order):
        pts = body[grp == g]
        first = 2 if b == 0 else 1
        k = min(16 - first, len(np.unique(pts.astype(int), axis=0)))
        if k > 0:
            _, _, cent = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            cent = cent[np.argsort(cent.sum(1))]
            pal[b * 16 + first:b * 16 + first + k, :3] = cent.clip(0, 255)
            pal[b * 16 + first:b * 16 + first + k, 3] = 255
        pal[b * 16 + (1 if b == 0 else 0)] = (*outline, 255)
    return pal


def snap(a, pal):
    """RGBA image -> palette indices. Outline-dark pixels take the outline slot of the bank most used around
    them, so runs stay inside one 16-colour bank."""
    idx = np.zeros(a.shape[:2], np.uint8)
    m = a[:, :, 3] > 0
    live = np.flatnonzero(pal[:, 3] > 0)
    cols = pal[live, :3].astype(np.int32)
    px = a[m][:, :3].astype(np.int32)
    idx[m] = live[((px[:, None, :] - cols[None]) ** 2).sum(2).argmin(1)]
    outl = np.isin(idx, [1, 16, 32, 48])
    bank = idx >> 4
    k = np.ones((5, 5), np.float32)
    counts = np.stack([cv2.filter2D(((bank == b) & m & ~outl).astype(np.float32), -1, k) for b in range(4)])
    slot = np.array([1, 16, 32, 48], np.uint8)[counts.argmax(0)]
    idx[outl] = slot[outl]
    return idx


def coverage(tiles, origin, size):
    W, H = size
    cov = np.zeros((H, W), bool)
    for t in tiles:
        x, y = t['x'] - origin[0], t['y'] - origin[1]
        cov[y:y + t['h'], x:x + t['w']] = True
    return cov


def tile_indexes(tiles, origin, idx):
    """Split a full-frame index image into per-tile cells: each opaque pixel goes to the front-most tile."""
    claimed = np.zeros(idx.shape, bool); out = []
    for t in tiles:  # earlier tiles are in front
        x, y = t['x'] - origin[0], t['y'] - origin[1]
        reg = np.s_[y:y + t['h'], x:x + t['w']]
        local = np.where(~claimed[reg], idx[reg], 0).astype(np.uint8)
        claimed[reg] |= local > 0
        if t['attr'] & 0x4000: local = local[::-1]
        if t['attr'] & 0x8000: local = local[:, ::-1]
        out.append(local)
    return out


def remap_cells(tiles, lut):
    """Keep a frame's drawing but move its colours into the mascot palette (frames the puppet can't pose)."""
    out = []
    for t in tiles:
        # only the selected-colour part (0, Gill's second half 8) moves to the mascot palette; tiles drawn in the
        # fixed rows (fire, ice, sparks: parts 1-6 / 9-14, kept by palette_bytes) keep their indices
        if (t['attr'] & 0x1ff) & 7 and (t['attr'] & 0x1ff) < 16:
            local = t['pixels'].astype(np.uint8)
        else:
            local = lut[t['pixels']].astype(np.uint8)
        if t['attr'] & 0x4000: local = local[::-1]
        if t['attr'] & 0x8000: local = local[:, ::-1]
        out.append(local)
    return out


def recolor_lut(bank_name, ext, pal):
    """Old palette index -> nearest new index after the bank's recolor.py rules (skin -> fur etc.)."""
    import recolor
    from common import decode_color
    old = decode_color((ext / f'{bank_name}pl.bin').read_bytes()).reshape(-1, 64, 4)[0].astype(int).tolist()
    rec = np.array(recolor.recolor_row(old, recolor.RULES.get(bank_name, {})))
    cols = pal[1:, :3].astype(int)
    lut = np.zeros(64, np.uint8)
    for i in range(1, 64):
        lut[i] = 1 + int(((cols - rec[i, :3]) ** 2).sum(1).argmin()) if rec[i, 3] else 0
    return lut


def grid_tiles(idx, x0, y0, side=32):
    """Cover the opaque pixels with side x side cells. Returns [(x, y, cell)] in sprite coordinates."""
    ys, xs = np.nonzero(idx)
    if not len(xs): return []
    out = []
    for by in range(ys.min(), ys.max() + 1, side):
        for bx in range(xs.min(), xs.max() + 1, side):
            c = np.zeros((side, side), np.uint8)
            blk = idx[by:by + side, bx:bx + side]
            c[:blk.shape[0], :blk.shape[1]] = blk
            if c.any(): out.append((x0 + bx, y0 + by, c))
    return out


def _encode(job):
    from codecs_sf3 import p6_encode, p6_encode_greedy
    from spritebank import morton
    wh, cell, optimal = job
    raw = np.empty(cell.size, np.uint8); raw[morton(cell.shape[0]).reshape(-1)] = cell.reshape(-1)
    return bytes([wh]) + (p6_encode if optimal else p6_encode_greedy)(raw.tobytes())


def rebuild_bank(bank, new_cells, optimal=False, layouts=None):
    """Rebuild the whole texture table (dedup, drop unused cells).
    new_cells: {pattern: [cell per existing tile]} keeps a pattern's tile list and swaps its pixels.
    layouts: {pattern: [(x, y, 32x32 cell)]} gives a pattern a brand-new tile list, written at the end of the file."""
    import struct
    from multiprocessing import Pool
    layouts = layouts or {}
    WH32 = 0x93  # 32x32 cell: w=4<<5, h=4<<2, side code 3
    jobs = {}; plan = {}; lay_keys = {}
    for p, tl in layouts.items():
        lay_keys[p] = []
        for x, y, c in tl:
            key = (WH32, c.tobytes()); jobs.setdefault(key, (WH32, c)); lay_keys[p].append((x, y, key))
    for p in range(bank.count):
        if p in layouts: continue
        for i, (dx, dy, attr, code) in enumerate(bank.entries(p)):
            if p in new_cells:
                wh = bank.cell(code)[0]; side = ((wh & 3) + 1) * 8
                full = np.zeros((side, side), np.uint8)
                c = new_cells[p][i]; full[:c.shape[0], :c.shape[1]] = c
                key = (wh, full.tobytes())
                jobs.setdefault(key, (wh, full))
                plan[(p, i)] = ('new', key)
            else:
                plan[(p, i)] = ('old', code)
    keys = list(jobs)
    with Pool() as pool:
        blobs_new = dict(zip(keys, pool.map(_encode, [(*jobs[k], optimal) for k in keys], chunksize=16)))
    blobs = []; index = {}
    def add(blob):
        if blob not in index: index[blob] = len(blobs); blobs.append(blob)
        return index[blob]
    result = bytearray(bank.data)
    for (p, i), (kind, v) in plan.items():
        if kind == 'old':
            pos = bank.tex + bank.offsets[v]; blob = bytes(bank.data[pos:pos + bank.cell(v)[2]])
        else:
            blob = blobs_new[v]
        code = add(blob)
        if code > 65535: raise ValueError('cell code limit')
        struct.pack_into('<H', result, bank.pattern_offsets[p] + 2 + i * 8 + 6, code)
    for tl in lay_keys.values():
        for _, _, key in tl:
            if add(blobs_new[key]) > 65535: raise ValueError('cell code limit')
    # Cells fill the original texture slot first; whatever doesn't fit goes after the character data at the
    # end of the file. The loader reads the whole file into a heap block sized from the file, and cell offsets
    # are 32-bit from the texture table, so cells past to_chd load fine (Q's stock bank is already 3.8 MB).
    budget = bank.end - bank.tex
    table = len(blobs) * 4
    if table >= budget: raise ValueError('cell table alone exceeds the texture slot')
    tail_base = len(bank.data) - bank.tex
    tail_base += (-tail_base) % 4
    offsets = []; inner = bytearray(); tail = bytearray()
    for blob in blobs:
        if table + len(inner) + len(blob) <= budget:
            offsets.append(table + len(inner)); inner.extend(blob)
        else:
            offsets.append(tail_base + len(tail)); tail.extend(blob)
    texture = struct.pack(f'<{len(offsets)}I', *offsets) + inner
    result[bank.tex:bank.end] = texture + b'\0' * (budget - len(texture))
    result += b'\0' * ((-len(result)) % 4) + tail
    # new tile lists for re-laid-out patterns: count, then (dx, dy, attr, code) with x -= dx, y -= dy per tile
    for p, tl in lay_keys.items():
        result += b'\0' * ((-len(result)) % 4)
        struct.pack_into('<I', result, p * 4, len(result))
        rec = bytearray(struct.pack('<H', len(tl))); x = y = 0
        for tx, ty, key in tl:
            code = index[blobs_new[key]]
            rec += struct.pack('<hhHH', x - tx, y - ty, 0, code); x, y = tx, ty
        result += rec
    total = table + len(inner) + len(tail)
    print(f'{len(blobs)} cells, {total:,} texture bytes (slot {budget:,}); file {len(bank.data):,} -> {len(result):,}')
    return bytes(result)


def palette_bytes(orig, pal):
    """Rows 0-12 (colour buttons) become the mascot palette. Rows 22-25 are hit tints (burn, shock...) that
    follow row 0's brightness: rebuild them by looking up each new colour's brightness in the old row 0.
    A two-half palette (pl00 Gill: 2 x 28 rows, the second body set reads half 1) gets the mascot in both halves."""
    from common import encode_color, decode_color
    src = np.frombuffer(orig, '<u2').reshape(-1, 64)
    rows = src.copy()
    old = decode_color(orig).reshape(-1, 64, 4).astype(int)
    enc = np.frombuffer(encode_color(pal), '<u2')
    lumn = pal[:, :3].astype(int).mean(1)
    for h in range(len(rows) // 28):
        b = h * 28
        for r in range(13): rows[b + r] = enc
        rows[b:b + 13, 0] = src[b:b + 13, 0]
        lum0 = old[b, 2:, :3].mean(1)
        # every effect row (hit tints 22-25, and the special-move rows such as Yun's Genei-Jin after-images
        # 16-18) is indexed for the original sprite's colours: rebuild it by brightness for the mascot's
        for r in range(13, 28):
            if b + r >= len(rows) or not src[b + r, 2:].any(): continue
            for i in range(2, 64):
                rows[b + r, i] = src[b + r, 2 + int(np.abs(lum0 - lumn[i]).argmin())]
    return rows.astype('<u2').tobytes()


# ------------------------------------------------------------------------------------------------------------
# No original body on screen. Every pattern is classified; every one that shows the fighter's body (any pose, roll,
# after-image, shadow, x-ray, burn, intro/win pose, a loose arm or a head-only close-up) is drawn as the mascot,
# whatever the detector or a hook made of it. Only pure effects and other characters keep their own drawing.
#   body   : posed by the core / hook, or its silhouette fits a posed frame (rotation allowed) -> that pose, fitted
#   piece  : a loose head / arm / bust drawing -> the matching puppet part(s) over it
#   effect : no body pixels (fireball, spark, dust, kanji, trail) -> keeps its drawing
#   other  : another character (Poison, Effie, Q's dummies...) -> keeps its drawing, listed by the audit
# After-images, silhouettes, x-rays and burns keep the original's look: the mascot's pixels are re-coloured from the
# original frame (tone ramp, dither pattern, outline only) in render().

# other characters drawn in a fighter's bank (pattern ranges, inclusive)
OTHER_CHARACTERS = {
    'pl05': ((1034, 1115), (1196, 1218)),   # Effie, Necro's win-pose partner
    'pl18': ((1130, 1177),),                # dummy head icons and the other characters running away
}
# Loose body pieces whose shape can't be told from an effect drawn in the fighter's own colours (from the
# review of every unposed pattern): {bank: {'head' | 'arm' | 'bust' | 'body': ranges}}
PIECES = {}
# Effects in the fighter's own colours that the shape test would take for a body: {bank: ranges}
FORCE_EFFECT = {}
# A body drawn as a dark silhouette inside an effect (fire pillar): the darkest colour group becomes a toned mascot,
# the rest of the drawing stays. Hooks can add their own as SILHOUETTE = [(a, b), ...].
SILHOUETTES = {'pl13': ((1300, 1328),), 'pl14': ((1024, 1053),)}
_audit = {}
_last_core = {}


def _in_ranges(p, ranges):
    return any((a <= p <= b) if isinstance(a, int) else False for a, b in ranges)


def _hook_list(hooks, name):
    """A hook's pattern list (set of ints or list of (a, b) ranges) as a membership test."""
    v = getattr(hooks, name, None) if hooks is not None else None
    if not v: return lambda p: False
    v = list(v)
    if v and isinstance(v[0], (tuple, list)): return lambda p: _in_ranges(p, v)
    s = set(v)
    return lambda p: p in s


def _filled(m):
    """Opaque mask with dither gaps closed and outline interiors filled (after-images drawn as an outline)."""
    m8 = cv2.morphologyEx(np.pad(m, 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ff = m8.copy(); h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    return ((ff != 2) | (m8 > 0))[2:-2, 2:-2]


def _rgba(bank_name, p):
    f = MOD / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
    return np.asarray(Image.open(f).convert('RGBA')) if f.exists() else None


def _other_char(bank_name, p, hooks):
    if _in_ranges(p, OTHER_CHARACTERS.get(bank_name, ())): return True
    if _hook_list(hooks, 'OTHER')(p): return True
    if bank_name == 'pl06':  # Poison, drawn in blue (same test as skins/pl06.py)
        a = _rgba(bank_name, p)
        if a is not None:
            px = a[a[..., 3] > 0].astype(int)
            if len(px) >= 20 and (px[:, 2] > px[:, 0] + 25).mean() > 0.35: return True
    return False


def _treatment(bank_name, bank, p, mask):
    """How the original draws this body frame when it isn't in the fighter's colours: None, or dict(tone=True,
    dither=(kind, parity) | None, outline=bool). Tone = colours unlike the idle drawing (burn, x-ray, shadow)."""
    if mask is None or mask.sum() < 30: return None
    filled = _filled(mask)
    outline = filled.sum() > 1.8 * mask.sum()
    ys, xs = np.nonzero(mask)
    dither = None
    for kind, cls in (('check', (xs + ys) % 2), ('grid', (xs % 2) * 2 + (ys % 2))):
        share = np.bincount(cls, minlength=4) / len(xs)
        if share.max() >= 0.85 and filled.sum() > 1.6 * mask.sum(): dither = (kind, int(share.argmax())); break
    tone = body_likeness(bank_name, bank, p) < 0.45
    if not (tone or dither or outline): return None
    return dict(tone=tone or bool(dither) or outline, dither=dither, outline=outline and not dither)


def _move_J(J, T, o_from, o_to):
    """J (sprite px of frame `from`) through the 2x3 anchor transform T into sprite px of frame `to`."""
    def f(v):
        if v is None: return None
        a = np.asarray(v, float) + o_from
        return a @ T[:, :2].T + T[:, 2] - o_to
    out = {}
    for k, v in J.items():
        if k.startswith('_'): out[k] = v  # bookkeeping (_treat, _src, ...) is not joints
        elif k in ('l', 'r') and isinstance(v, dict): out[k] = {kk: f(vv) for kk, vv in v.items()}
        elif k in ('sho', 'hip', 'head') and v is not None: out[k] = f(v)
        else: out[k] = v
    return out


_colsets = {}


def _ckey(c):
    c = np.asarray(c).astype(np.int64)
    return (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]


def _colour_sets(bank_name, bank, pose):
    """Colours of the idle drawing: all of them, the head's, the arms', and 'head-only' ones (hair, eyes, headgear:
    colours that sit in the head region at least 80% of the time). Also the idle head's area in px."""
    if bank_name in _colsets: return _colsets[bank_name]
    tot, inhead, inarm = {}, {}, {}
    head_areas = []
    try:
        idle = bank.standing_patterns(8)
    except Exception:
        idle = []
    for q in idle:
        a = _rgba(bank_name, q); rec = pose.get(str(q), {})
        if a is None or 'kp' not in rec: continue
        m = a[..., 3] > 0; ys, xs = np.nonzero(m); keys = _ckey(a[m][:, :3])
        kp, sc = np.array(rec['kp']), np.array(rec['score'])
        hp = [kp[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if sc[i] >= 0.5]
        sho = [kp[i] for i in (LSHO, RSHO) if sc[i] >= 0.5]
        pts = np.stack([xs, ys], 1).astype(float)
        hm = np.zeros(len(pts), bool); am = np.zeros(len(pts), bool)
        if hp and sho:
            hc = np.mean(hp, 0); r = 0.6 * np.linalg.norm(hc - np.mean(sho, 0))
            hm = np.linalg.norm(pts - hc, axis=1) < r
            head_areas.append(int(hm.sum()))
        for s_, e_, w_ in ((LSHO, LELB, LWRI), (RSHO, RELB, RWRI)):
            for i, j in ((s_, e_), (e_, w_)):
                if min(sc[i], sc[j]) < 0.4: continue
                A, B = kp[i], kp[j]; v = B - A; L2 = float(v @ v) or 1.0
                t = np.clip((pts - A) @ v / L2, 0, 1)
                am |= np.linalg.norm(pts - (A + t[:, None] * v), axis=1) < 4
        for k, h, ar in zip(keys, hm, am):
            tot[k] = tot.get(k, 0) + 1
            if h: inhead[k] = inhead.get(k, 0) + 1
            if ar: inarm[k] = inarm.get(k, 0) + 1
    allc = set(tot)
    head = {k for k in inhead}
    arm = {k for k in inarm}
    head_only = {k for k, n in inhead.items() if n >= 0.8 * tot[k] and tot[k] >= 3}
    _colsets[bank_name] = dict(all=allc, head=head, arm=arm, head_only=head_only,
                               head_area=float(np.median(head_areas)) if head_areas else 300.0)
    return _colsets[bank_name]


def _piece_kind(bank_name, bank, pose, p, m):
    """('head' | 'arm' | 'bust', erase mask) for a loose body piece, or None. By colour: the piece is drawn in the
    fighter's idle colours (head-only colours such as hair for a head or bust; arm colours along an elongated
    shape for an arm). A head poking out of something else (Oro's cocoon) is a head piece over that drawing."""
    cs = _colour_sets(bank_name, bank, pose)
    a = _rgba(bank_name, p)
    if a is None: return None
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    # crumbling / scattered drawings (fire, explosions) are not a loose bust or arm (heads and icons may be framed)
    scattered = effect_shape(m) or (n_ > 1 and st_[1:, cv2.CC_STAT_AREA].max() < 0.85 * m.sum())
    keys = _ckey(a[..., :3]); keys[~m] = -1
    flat = keys[m]
    inall = np.isin(keys, list(cs['all'])) & m
    hair = np.isin(keys, list(cs['head_only'])) & m
    f_body = inall.sum() / m.sum()
    n_hair = int(hair.sum())
    if f_body >= 0.6:
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1).astype(float)
        ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)) if len(pts) > 3 else np.array([1, 1])
        elong = float(np.sqrt(max(ev.max(), 1e-6) / max(ev.min(), 1e-6)))
        f_arm = np.isin(flat, list(cs['arm'])).mean()
        f_head = np.isin(flat, list(cs['head'])).mean()
        big = m.sum() > 3 * cs['head_area']
        if n_hair >= max(12, 0.05 * m.sum()):
            if not big: return 'head', inall
            return None if scattered else ('bust', inall)
        if elong >= 1.5 and f_arm >= 0.6 and not scattered:
            return 'arm', inall
        if f_head >= 0.75 and not big:
            return 'head', inall
        if f_arm >= 0.7 and not big and not scattered:
            return 'arm', inall
        return None
    # mostly something else: a head sticking out of it?
    if n_hair >= max(25, 0.3 * cs['head_area']):
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(hair.astype(np.uint8), np.ones((5, 5), np.uint8)))
        if n > 1:
            i = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
            blob = (lab == i)
            erase = cv2.dilate(blob.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            return 'head', erase & inall
    return None


def _piece_J(kind, erase, face, anchor=None):
    """A loose head / arm / bust drawing: the J compose() turns into the matching puppet part(s); render() keeps the
    rest of the drawing (outside `erase`) under it."""
    ys, xs = np.nonzero(erase)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    piece = dict(kind=kind, c=c, w=float(np.ptp(xs) + 1), h=float(np.ptp(ys) + 1), erase=erase)
    if kind in ('arm', 'leg'):
        ev, evec = np.linalg.eigh(np.cov((pts - c).T))
        ax = evec[:, ev.argmax()]
        t = (pts - c) @ ax; nrm = np.array([-ax[1], ax[0]])
        a, b = c + ax * t.min(), c + ax * t.max()
        span = t.max() - t.min()

        def width(lo, hi):
            sel = (t >= lo) & (t <= hi)
            return np.ptp((pts[sel] - c) @ nrm) if sel.sum() > 2 else 0
        # a loose arm hangs from the shoulder at the fighter's anchor: the hand is the far end (without an anchor,
        # the end whose last fifth is wider)
        if anchor is not None:
            if np.linalg.norm(a - anchor) > np.linalg.norm(b - anchor): a, b = b, a
        elif width(t.min(), t.min() + span / 5) > width(t.max() - span / 5, t.max()): a, b = b, a
        piece.update(a=a, b=b, thick=float(np.percentile(np.abs((pts - c) @ nrm), 75)) * 2 + 2)
    if kind == 'bust':
        sel = ys < ys.min() + 0.45 * piece['h']
        piece['head'] = np.array([xs[sel].mean(), ys[sel].mean()])
    z = {n: c.copy() for n in ('sho', 'elb', 'wri', 'hip', 'kne', 'ank')}
    return dict(sho=c.copy(), hip=c.copy(), head=c.copy(), face=face, l=dict(z), r={k: v.copy() for k, v in z.items()},
                ground=None, plant=False, _piece=piece)


def classify_and_complete(bank_name, bank, pose, G, cfg, hooks, out, core, frames=None):
    """Classify every pattern (or `frames`) and give every body pattern that is still unposed a mascot pose.
    Fills `out` in place; returns {pattern: class}. Classes: posed, body, piece, effect, other, none.
    The pose library is what is posed after the hooks (a hook that turned a misread arm piece away keeps it out)."""
    masks = {}
    org = lambda p: np.array(pose[str(p)]['origin'], float)
    face = int(cfg.get('face', bank_facing(pose)))
    occ, nb = frame_links(bank_name, bank, pose)
    lib = {q: J for q, J in out.items() if '_piece' not in J and '_borrowed' not in J and pose[str(q)]['size'][1] >= 32}
    if frames is not None:  # a preview of a few frames: the core's poses stand in for the frames not run through the hook
        lib.update({q: fill_limbs(copy_J(J), G) for q, J in core.items() if q not in frames and q not in lib
                    and pose[str(q)]['size'][1] >= 32})
    info = {}
    for q in lib:
        m = _alpha(bank_name, q, masks)
        if m is not None and m.any(): info[q] = (m.shape[1] / max(1, m.shape[0]), float(_filled(m).sum()))
    areas = sorted(a for _, a in info.values())
    body_area = areas[len(areas) // 2] if areas else 2000.0
    sole, ah = _sole_offset(bank_name, bank, pose)
    fcache = {}
    filled = lambda q: fcache.setdefault(q, _filled(_alpha(bank_name, q, masks)))
    sides = (('l', (LELB, LWRI, LKNE, LANK)), ('r', (RELB, RWRI, RKNE, RANK)))

    def fit_body(p, m, mf, need=0.45, wide=False, confident=True):
        """The silhouette (outline / dither filled in) against posed frames: animation neighbours first, then
        pattern neighbours, then posed frames of a similar size and shape anywhere in the bank -> fitted pose."""
        near, seen, frontier = [], {p}, [p]
        for _ in range(3):
            nxt = []
            for x in frontier:
                for y in nb.get(x, {}):
                    if y not in seen: seen.add(y); nxt.append(y)
            near += [y for y in nxt if y in info]; frontier = nxt
        near += [q for d in range(1, 9) for q in (p - d, p + d) if q in info and q not in near]
        near = near[:6]
        asp = m.shape[1] / max(1, m.shape[0]); area = float(mf.sum())
        lo_a, hi_a, lo_r, hi_r, n_al = (0.6, 1.6, 0.5, 2.0, 30) if wide else (0.75, 1.33, 0.6, 1.6, 8)
        alike = sorted((q for q, (a2, ar) in info.items() if q not in near and lo_a < a2 / asp < hi_a and lo_r < ar / area < hi_r),
                       key=lambda q: abs(info[q][1] - area))[:n_al]
        best = (0.0, None, None)
        for i, q in enumerate(near + alike):
            r, T = _fit(filled(q), org(q), mf, org(p), step=30)
            if T is not None and r >= (need if i < len(near) or wide else need + 0.1) and r > best[0]: best = (r, q, T)
        if best[1] is None: return None
        if best[1] not in near and need < 0.5 and not confident:
            return None  # only shape-alike frames elsewhere in the bank: too weak for a drawing nobody reads as a body
        q, T = best[1], best[2]
        J = _move_J(lib[q], T, org(q), org(p))
        kk = np.zeros((17, 2)); sc = np.zeros(17)
        for side, ids in sides:  # pull the carried pose onto the drawing
            for n, i in zip(('elb', 'wri', 'kne', 'ank'), ids):
                if J[side][n] is not None: kk[i] = J[side][n]; sc[i] = 1
        _snap(kk, sc, mf, LIMB_JOINTS, reach=30)
        for side, ids in sides:
            for n, i in zip(('elb', 'wri', 'kne', 'ank'), ids):
                if sc[i]: J[side][n] = kk[i]
        J['_borrowed'] = q; J['_fit'] = round(float(best[0]), 3)
        return J

    sil_hook = _hook_list(hooks, 'SILHOUETTE')
    hook_effect = _hook_list(hooks, 'EFFECT')
    hook_pieces = dict(getattr(hooks, 'PIECES', {}) or {}) if hooks is not None else {}

    def placeholder(p):
        """Debug placeholder cells (red (246,0,0) tiles / dark grey (41,41,41) with hex numbers): not a drawing."""
        a = _rgba(bank_name, p)
        if a is None: return False
        px = a[a[..., 3] > 0][:, :3].astype(int)
        if not len(px): return False
        red = ((px[:, 0] >= 230) & (px[:, 1] <= 20) & (px[:, 2] <= 20)).mean()
        grey = ((np.abs(px - 41) <= 3).all(1)).mean()
        return red + grey >= 0.35

    def split_body(p, m, relaxed=False):
        a = _rgba(bank_name, p)
        if a is None or m.sum() < 300: return None
        px = a[m][:, :3].astype(np.float32)
        if len(np.unique(px, axis=0)) < 3: return None
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(2)
        groups = []
        lum = px @ np.array([0.299, 0.587, 0.114], np.float32)
        for cut in (60, 90, 120):  # a dark silhouette inside bright fire / energy
            groups.append(lum < cut)
        for k in (2, 3):
            _, lab, cen = cv2.kmeans(px, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
            lab = lab.ravel()
            groups += [lab == c for c in np.argsort(-np.bincount(lab, minlength=k))]
        for g in groups:
                part = np.zeros_like(m); part[m] = g
                part = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
                if part.sum() < 0.12 * m.sum() or part.sum() < 200 or part.sum() > 0.9 * m.sum(): continue
                pf = _filled(part)
                if not relaxed:
                    if effect_shape(part) and pf.sum() < 1.8 * part.sum(): continue
                    if _solidity(pf) > 0.9: continue
                J = fit_body(p, part, pf, need=0.3 if relaxed else 0.5, wide=True)
                if J is not None:
                    J['_keep'] = m & ~pf  # the effect's pixels, drawn behind the puppet
                    J['_tone_mask'] = part
                    return J
        return None

    def is_icon(p, m):
        """A small face / portrait icon: icon-sized, compact, solid and many-coloured (sparks and dust are thin or
        few-coloured)."""
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n_ > 2: m = lab_ == 1 + int(st_[1:, cv2.CC_STAT_AREA].argmax())
        ys, xs = np.nonzero(m)
        H, W = np.ptp(ys) + 1, np.ptp(xs) + 1
        if max(H, W) > 72 or min(H, W) < 12 or not 0.6 <= W / H <= 1.7: return False
        box = (np.ptp(xs) + 1) * (np.ptp(ys) + 1)
        if m.sum() < 0.55 * box or m.sum() < 120: return False
        a = _rgba(bank_name, p)
        return a is not None and len(np.unique(_ckey(a[m][:, :3]))) >= 10

    # shock x-rays (a, b, a, c damage loops): b and c always keep the x-ray look
    xray = set()
    for name, seq in _anim_seqs(bank_name, bank):
        if name == 'dmca' and len(seq) >= 4 and len(set(seq[0::2])) == 1 and 2 <= len(set(seq)) <= 4:
            xray |= {q for q in seq[1::2] if q != seq[0]}

    dmg = {q for name, seq in _anim_seqs(bank_name, bank) if name == 'dmca' for q in seq}

    def force_tone(tr):
        tr = dict(tr or {})
        return dict(tone=True, dither=tr.get('dither'), outline=tr.get('outline', False))
    classes = {}
    todo = list(range(bank.count))
    for _pass in range(4):
        added = []
        for p in todo:
            if frames is not None and p not in frames: continue
            rec = pose.get(str(p))
            if p in out:
                classes[p] = 'posed'; continue
            m = _alpha(bank_name, p, masks) if rec and 'skip' not in rec else None
            if m is None or m.sum() < 40:
                classes[p] = 'none'; continue
            if _other_char(bank_name, p, hooks):
                classes[p] = 'other'; continue
            silhouette = _in_ranges(p, SILHOUETTES.get(bank_name, ())) or sil_hook(p)
            if not silhouette and (_in_ranges(p, FORCE_EFFECT.get(bank_name, ())) or hook_effect(p) or placeholder(p)):
                classes[p] = 'effect'; continue
            forced = next((k for k, rs in list(PIECES.get(bank_name, {}).items()) + list(hook_pieces.items())
                           if (_in_ranges(p, rs) if rs and isinstance(next(iter(rs)), (tuple, list)) else p in set(rs))), None)
            mf = filled(p)
            J = None
            if p in core and rec and 'score' in rec and float(np.mean(rec['score'])) >= 0.6 and not effect_shape(m) \
                    and _solidity(mf) < 0.95 and (main_part(bank, p) != 0 or body_likeness(bank_name, bank, p) >= 0.45):
                J = fill_limbs(copy_J(core[p]), G); J['_borrowed'] = 'core'; J['_fit'] = 1.0
            # a drawing in colours unlike the fighter (flame swooshes, energy) only counts as a body when it is drawn
            # like an after-image (outline / dither), is a damage frame (burn, x-ray) or is forced
            mp_ = main_part(bank, p)
            det = float(np.mean(rec['score'])) if rec and 'score' in rec else 0.0
            solid = _solidity(mf)
            lookalike = (forced == 'body' or p in xray
                         or (mp_ == 0 and body_likeness(bank_name, bank, p) >= 0.45)
                         or (mp_ != 0 and det >= 0.4)
                         or (mf.sum() > 1.6 * m.sum() and solid < 0.9))
            lookalike = lookalike and (solid < 0.95 or forced == 'body')  # panels and pillars are not bodies
            dissolve = effect_shape(m) and det >= 0.6 and mp_ is not None
            if dissolve:  # a body crumbling into dust / dithering out: fit the closed silhouette
                mf = _filled(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0)
            if J is None and forced in (None, 'body') and (lookalike or dissolve) and \
                    (dissolve or not (effect_shape(m) and mf.sum() < 1.8 * m.sum())):
                J = fit_body(p, m, mf, confident=det >= 0.35 or p in xray or forced == 'body' or mf.sum() > 1.6 * m.sum())
                if J is None and forced == 'body':
                    q = min(info, key=lambda x: abs(x - p))
                    cq = np.argwhere(filled(q))[:, ::-1].mean(0) + org(q); cp = np.argwhere(mf)[:, ::-1].mean(0) + org(p)
                    J = _move_J(lib[q], np.hstack([np.eye(2), (cp - cq)[:, None]]), org(q), org(p))
                    J['_borrowed'] = q; J['_fit'] = 0.0
            if J is None and forced is None and is_icon(p, m):
                # a face / portrait icon: the mascot's head fitted to it (a label printed over it stays)
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                face_px = (lab_ == 1 + int(st_[1:, cv2.CC_STAT_AREA].argmax())) if n_ > 2 else m
                J = _piece_J('head', face_px, face)
            if J is None and silhouette:
                J = split_body(p, m, relaxed=True)
            if J is None and forced is None and not effect_shape(m) and m.sum() > 0.5 * body_area:
                # a body inside an effect (Urien's dark silhouette in the fire pillar): split the drawing by colour and
                # look for a body-shaped colour group; the rest of the drawing stays, behind the puppet
                J = split_body(p, m)
            if J is None:
                pk = _piece_kind(bank_name, bank, pose, p, m)
                if forced in ('head', 'arm', 'leg', 'bust'): pk = (forced, pk[1] if pk else m)
                if pk is not None:
                    J = _piece_J(pk[0], pk[1] if pk[1].any() else m, face, anchor=-org(p))
            if J is None:
                classes[p] = 'effect'; continue
            if '_piece' not in J:
                J['ground'] = float(-pose[str(p)]['origin'][1] + sole); J['ankle_h'] = ah
                if J['ground'] < np.nonzero(m.any(1))[0].max() - 6: J['ground'] = None
                J = fill_limbs(J, G)
            J['_src'] = (bank_name, p)
            if '_piece' not in J:
                tr = _treatment(bank_name, bank, p, J['_tone_mask'] if '_tone_mask' in J else m)
                if p in xray or '_tone_mask' in J: tr = force_tone(tr)
                if dissolve and J.get('_borrowed') != 'core': tr = dict(tr or {}, mask_src=True)
                if tr: J['_treat'] = tr
            out[p] = J
            classes[p] = 'piece' if '_piece' in J else 'body'
            if '_piece' not in J and J.get('_fit', 0) >= 0.5 and pose[str(p)]['size'][1] >= 32:
                lib[p] = {k: v for k, v in J.items() if k not in ('_keep', '_tone_mask', '_treat', '_src')}
                lib[p].pop('_borrowed', None)
                msk = J.get('_tone_mask'); msk = m if msk is None else msk
                info[p] = (msk.shape[1] / max(1, msk.shape[0]), float(_filled(msk).sum()))
                fcache[p] = _filled(msk); added.append(p)
        # frames next to a newly fitted one get another try (a run of unposed frames chains from its first fit)
        todo = sorted({y for x in added for y in nb.get(x, {}) if classes.get(y) == 'effect'})
        if not todo: break

    # posed frames drawn in another look (burn, x-ray, after-image) keep it too
    for p, J in out.items():
        if classes.get(p) != 'posed' or '_treat' in J or '_piece' in J: continue
        tr = _treatment(bank_name, bank, p, _alpha(bank_name, p, masks))
        if p in xray: tr = force_tone(tr)
        if tr: J['_treat'] = tr; J['_src'] = (bank_name, p)
    _audit.setdefault(bank_name, {}).update(classes)
    return classes


def apply_treatment(a, J, size):
    """Re-colour a rendered puppet (RGBA on the (size + 2M) canvas) the way the original frame is drawn; a body piece
    keeps the rest of its original drawing under the puppet part."""
    src = J.get('_src')
    o = _rgba(*src) if src else None
    if o is None: return a
    a = a.copy()
    pc = J.get('_piece')
    if pc is not None:
        keep = (o[..., 3] > 0) & ~pc['erase']
        if keep.any():
            base = np.zeros_like(a); H, W = keep.shape
            base[M:M + H, M:M + W][keep] = o[keep]
            top = a[:, :, 3:4] > 0
            a = np.where(top, a, base)
        return a
    tr = J.get('_treat')
    if not tr and J.get('_keep') is None: return a
    om = o[..., 3] > 0
    if J.get('_tone_mask') is not None: om = om & J['_tone_mask']
    if om.sum() < 10: return a
    if not tr: tr = {}
    vis = a[:, :, 3] > 0
    if tr.get('outline'):
        vis = vis & ~(cv2.erode(vis.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if tr.get('mask_src'):
        H, W = om.shape
        src = np.zeros(vis.shape, bool)
        src[M:M + H, M:M + W] = cv2.dilate((o[..., 3] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        vis = vis & src
    if tr.get('dither'):
        kind, par = tr['dither']
        yy, xx = np.indices(vis.shape); xx = xx - M; yy = yy - M  # same parity as the original's pixels
        cls = (xx + yy) % 2 if kind == 'check' else (xx % 2) * 2 + (yy % 2)
        vis = vis & (cls == par)
    if tr.get('tone'):
        cols = o[om][:, :3].astype(float)
        cols = cols[np.argsort(cols @ np.array([0.299, 0.587, 0.114]))]
        pc_ = a[vis][:, :3].astype(float)
        if len(pc_):
            rank = np.argsort(np.argsort(pc_ @ np.array([0.299, 0.587, 0.114]))) / max(1, len(pc_) - 1)
            a[vis, :3] = cols[(rank * (len(cols) - 1)).astype(int)].astype(np.uint8)
    a[~vis] = 0
    keep = J.get('_keep')
    if keep is not None and keep.any():  # the effect around a body (fire pillar) stays, behind the puppet
        base = np.zeros_like(a); H, W = keep.shape
        base[M:M + H, M:M + W][keep] = o[keep]
        a = np.where(a[:, :, 3:4] > 0, a, base)
    return a


def prepare_poses(bank_name, bank, pose, G, cfg, hooks, frames=None):
    """Every frame's cleaned joints, {pattern: J}; frames the puppet can't pose are left out.
    Shared clean-up for all fighters lives here (core_poses: animation-aware facing, labels, fills, smoothing);
    per-fighter fixes go in sf3mod/skins/<bank>.py fix_pose, which gets the cleaned, filled J."""
    out = {}
    ctx = dict(bank_name=bank_name, bank=bank, pose=pose, G=G, cfg=cfg)
    # a hook's KEEP patterns are never dropped by the core's effect gate
    ccfg = dict(cfg)
    hk = set(getattr(hooks, 'KEEP', ()) or ()) if hooks is not None else set()
    if hk: ccfg['keep_frames'] = tuple(sorted(set(cfg.get('keep_frames', ()) or ()) | hk))
    core, stats = core_poses(bank_name, bank, pose, G, ccfg)
    ctx['core'] = core; ctx['core_stats'] = stats; _last_core[bank_name] = core
    borrow = getattr(hooks, 'BORROW', {}) if hooks is not None else {}
    # a hook's EFFECT / OTHER patterns keep their drawing even when the detector read a body in them (props such as
    # Ryu's sandbag or Dudley's car): the core's pose is dropped here, the completion pass then keeps the drawing.
    # Silhouettes (a body inside an effect) are left to the completion pass, which splits the body out.
    h_eff, h_oth, h_sil = (_hook_list(hooks, n) for n in ('EFFECT', 'OTHER', 'SILHOUETTE'))
    keep_drawing = lambda p: (h_eff(p) or h_oth(p)) and not (h_sil(p) or _in_ranges(p, SILHOUETTES.get(bank_name, ())))
    for p in range(bank.count):
        if frames is not None and p not in frames: continue
        if keep_drawing(p): continue
        if p in core:
            J = fill_limbs(copy_J(core[p]), G)
        elif isinstance(borrow, dict) and borrow.get(p) in core:
            # the core couldn't pose p but the hook borrows another frame's pose for it: hand it that pose
            # (moved into p's sprite coordinates) so fix_pose gets called
            q = borrow[p]; d = np.array(pose[str(q)]['origin'], float) - np.array(pose[str(p)]['origin'], float)
            J = fill_limbs(copy_J(core[q]), G)
            for k, v in J.items():
                if isinstance(v, dict): J[k] = {kk: (None if vv is None else vv + d) for kk, vv in v.items()}
                elif isinstance(v, np.ndarray) and v.shape == (2,): J[k] = v + d
            if J.get('ground') is not None: J['ground'] = J['ground'] + d[1]
        else:
            continue
        if hooks is not None and hasattr(hooks, 'fix_pose'):
            J = hooks.fix_pose(p, J, pose[str(p)], ctx)
            if J is None: continue
        out[p] = J
    if cfg.get('complete', True):
        # nothing may show the original fighter: body patterns the detector, the core or a hook left out get a
        # borrowed mascot pose (or a puppet part); only effects and other characters keep their drawing
        classify_and_complete(bank_name, bank, pose, G, cfg, hooks, out, core, frames)
    return out


def setup(bank_name, ext=None):
    """Load everything needed to render one fighter: (bank, pose, parts, G, cfg, hooks)."""
    ext = ext or MOD / 'build' / 'extracted'
    cfg, hooks = fighter(bank_name)
    pose = json.loads((MOD / 'build' / 'skin' / bank_name / 'pose.json').read_text())
    bank = SpriteBank.open(ext / f'{bank_name}.bin')
    parts = load_puppet(cfg['mascot'], getattr(hooks, 'PIVOTS', None))
    G = global_scale(pose, cfg, parts) * cfg.get('scale', 1.0)
    # mesh skinning (sf3mod/mesh.py) when the mascot has layer art; cfg renderer='puppet' keeps the cut-out
    import mesh
    parts.mesh = mesh.attach(parts, cfg, pose, cfg['mascot'])
    cfg['renderer'] = 'mesh' if parts.mesh is not None else 'puppet'
    return bank, pose, parts, G, cfg, hooks


def render(J, parts, G, cfg, size):
    """One frame's puppet as an RGBA array on the (size + 2M) canvas (re-coloured like the original for after-images,
    silhouettes, x-rays and burns)."""
    W, H = size
    a = shrink(compose(J, parts, G, cfg, (W, H)), (W + 2 * M, H + 2 * M))
    return apply_treatment(a, J, (W, H)) if J.get('_treat') or J.get('_piece') or J.get('_keep') is not None else a


def redraw_frames(bank_name, pose, pats):
    """Hand-redrawn frames (sf3mod/redraw/sheets.py -> build/redraw/<bank>/final/frames/NNNN.png, RGBA at the original
    canvas size) replace the mesh render of the same pattern, on the same (size + 2M) canvas."""
    d = MOD / 'build' / 'redraw' / bank_name / 'final' / 'frames'
    out = {}
    if not d.exists(): return out
    for p in pats:
        f = d / f'{p:04d}.png'
        if not f.exists(): continue
        W, H = pose[str(p)]['size']
        im = Image.open(f).convert('RGBA')
        mg = (im.width - W) // 2  # frames may carry a margin for tails and capes past the original canvas
        if im.size != (W + 2 * mg, H + 2 * mg) or not 0 <= mg <= M: continue
        a = np.zeros((H + 2 * M, W + 2 * M, 4), np.uint8)
        a[M - mg:M + H + mg, M - mg:M + W + mg] = np.asarray(im)
        a[:, :, 3] = np.where(a[:, :, 3] >= 110, 255, 0); a[a[:, :, 3] == 0] = 0
        out[p] = a
    return out


def build(ext, out, only=None, frames=None, optimal=False, preview=True):
    from common import decode_color, encode_color
    for bank_name in SKINS:
        if only and bank_name not in only: continue
        work = MOD / 'build' / 'skin' / bank_name
        bank, pose, parts, G, cfg, hooks = setup(bank_name, ext)
        todo = [p for p in range(bank.count) if frames is None or p in frames]
        Js = prepare_poses(bank_name, bank, pose, G, cfg, hooks, set(todo))
        rendered = {p: render(J, parts, G, cfg, tuple(pose[str(p)]['size'])) for p, J in Js.items()}
        # body pieces keep the mesh part over their original drawing; re-coloured frames get the same treatment
        # a redrawn body piece (a lone arm) replaces the whole piece; head pieces have no redraw and keep the mesh part
        redrawn = redraw_frames(bank_name, pose, [p for p in todo if p in rendered and Js[p].get('_keep') is None])
        for p, a in redrawn.items():
            if Js[p].get('_treat') and Js[p].get('_piece') is None:
                redrawn[p] = apply_treatment(a, Js[p], tuple(pose[str(p)]['size']))
        rendered.update(redrawn)
        # loose pieces of the original fighter the game draws as separate objects (busts, arms, blurs) are not
        # posed; a redrawn copy listed in the hook's REDRAW_EXTRA replaces them as they are
        extra = [p for p in (getattr(hooks, 'REDRAW_EXTRA', ()) or ()) if p in todo and p not in rendered]
        more = redraw_frames(bank_name, pose, extra) if extra else {}
        rendered.update(more); redrawn.update(more)
        if redrawn: print(f'{bank_name}: {len(redrawn)} frames from the per-frame redraw')
        failed = [p for p in todo if p not in rendered]
        heads = [np.asarray(parts['head'].im.resize((48, int(48 * parts['head'].im.height / parts['head'].im.width)), Image.LANCZOS))] * 25
        pal = make_palette([rendered[p] for p in list(rendered)[::max(1, len(rendered) // 120)]] + heads)
        pal = decode_color(encode_color(pal)); pal[0] = 0
        new_cells = {}; layouts = {}
        prev = work / 'preview'; prev.mkdir(exist_ok=True)
        for p, a in rendered.items():
            ox, oy = pose[str(p)]['origin']
            idx = snap(a, pal)
            layouts[p] = grid_tiles(idx, ox - M, oy - M)
            if preview and (frames is not None or p % 8 == 0):
                Image.fromarray(pal[idx], 'RGBA').crop(Image.fromarray(pal[idx], 'RGBA').getbbox() or (0, 0, 1, 1)).save(prev / f'{p:04d}.png')
        # a hook's HIDE patterns (the original fighter's separate hair sprites, e.g. Ibuki's ponytail) get one
        # transparent tile: the mascot has no such part, and the redraw already draws its own scarf/tail
        for p in (getattr(hooks, 'HIDE', ()) or ()) if hooks is not None else ():
            if p < bank.count and p not in layouts and (frames is None or p in frames):
                ox, oy = pose[str(p)]['origin'] if str(p) in pose else (0, 0)
                layouts[p] = [(ox, oy, np.zeros((32, 32), np.uint8))]
        lut = recolor_lut(bank_name, ext, pal)
        for p in range(bank.count):
            if p not in layouts and (frames is None or p in frames):
                new_cells[p] = remap_cells(bank.tiles(p), lut)
        print(f'{bank_name}: {len(rendered)} frames skinned, {len(failed)} colour-mapped: {failed[:40]}')
        data = rebuild_bank(bank, new_cells, optimal, layouts)
        (out / f'{bank_name}.bin').write_bytes(data)
        orig_pal = (ext / f'{bank_name}pl.bin').read_bytes()
        (out / f'{bank_name}pl.bin').write_bytes(palette_bytes(orig_pal, pal))
        if frames is None: (work / 'failed.json').write_text(json.dumps(failed))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('bank'); ap.add_argument('--frames'); ap.add_argument('--optimal', action='store_true')
    a = ap.parse_args()
    fr = [int(x) for x in a.frames.split(',')] if a.frames else None
    # only a full-bank run writes the build input out/; a --frames try-out goes to a scratch dir (try/) so it never
    # overwrites the full bank that build_mod.py packs
    outdir = MOD / 'build' / 'skin' / a.bank / ('try' if fr else 'out'); outdir.mkdir(parents=True, exist_ok=True)
    build(MOD / 'build' / 'extracted', outdir, only=[a.bank], frames=fr, optimal=a.optimal)
    print('wrote', outdir)
