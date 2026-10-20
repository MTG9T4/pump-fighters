"""2D mesh skinning for the mascot skins (Spine / DragonBones style), replacing the rigid cut-out puppet.

Per mascot, assets/mesh/<m>/ holds two continuous layer drawings in one bind pose (sf3mod/mesh_art.py makes them):
  body.png  neck stub to feet, no arms / head: torso, pelvis and both legs as ONE surface
  arm.png   one arm, shoulder cap to fist (used for both arms; the far one is shaded)
  bind.json their bind joints (python sf3mod/mesh.py bind <m>: RTMPose on a composite + geometric correction;
            hand edits survive in bind.json's "fix" block)
Each layer is triangulated on a grid inside its alpha. Every vertex gets smooth bone weights from geodesic distances
(inside the drawing, so one leg never pulls the other) to the bones: body = spine, near/far thigh, shin, foot;
arm = upper arm, forearm, hand. Per frame the bones are laid on the frame's joints J (skin.py's pose pipeline),
and every vertex is moved by a blended transform: its bones' rotations are blended as angles about the joint they
share (no 'candy wrapper' collapse at bent knees and elbows), their stretch / squash (SF3 foreshortening, PEPE's
stretchy limbs) blended linearly. Triangles are drawn with a per-pixel inverse-affine remap at the 4x render scale,
in depth groups: far arm, far leg, torso, near leg, near upper arm, head (rigid, as before), near forearm + fist.
The head, tail and cape stay rigid cut-out parts (assets/puppet/<m>/).

    python sf3mod/mesh.py bind swordcat [--show]     # find + save the bind joints, draw assets/mesh/<m>/_bind.png
    python sf3mod/mesh.py weights swordcat           # weight / mesh debug image assets/mesh/<m>/_weights.png
skin.compose() calls compose_mesh() when cfg['renderer'] == 'mesh' (the default for mascots with layer art).
"""
import heapq, json, math, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

MOD = Path(__file__).resolve().parent
MESH = MOD / 'assets' / 'mesh'

NECK_BONE = True   # the neck stub as its own bone (guide 9.7); bind.json "neck_bone": false restores the old rig
AXIAL_LEGS = True  # leg weights as axial ramps (guide 9.2); bind.json "axial_legs": false restores the falloff
BODY_BONES = ('spine', 'n_thigh', 'n_shin', 'n_foot', 'f_thigh', 'f_shin', 'f_foot')
ARM_BONES = ('upper', 'fore', 'hand')


def available(mascot):
    d = MESH / mascot
    return all((d / f).exists() for f in ('body.png', 'arm.png', 'bind.json'))


# ---------------------------------------------------------------------------------------------------- art loading

def key_alpha(path):
    """Flat black background -> alpha (flood fill from the borders, so dark outlines inside the shape stay)."""
    import skin
    a = np.array(Image.open(path).convert('RGB'))
    bg = skin.border_background(a.max(axis=2) < 14)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
    # drop specks (stray pixels, signature dots): keep blobs over 0.5% of the biggest
    n, lab, st, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), connectivity=8)
    if n > 2:
        big = st[1:, cv2.CC_STAT_AREA].max()
        keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 0.005 * big
        alpha = np.where(keep[lab], alpha, 0).astype(np.uint8)
    return np.dstack([a, alpha])


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def _seg_proj(p, a, b):
    """Closest points on segment a-b to points p (n,2) and their distances."""
    d = b - a; L2 = float(d @ d) or 1e-9
    t = np.clip((p - a) @ d / L2, 0, 1)
    c = a + t[:, None] * d
    return c, np.linalg.norm(p - c, axis=1)


class Layer:
    """One layer drawing as a weighted triangle mesh.
    bones: [(name, a0, b0)] in art px (bind pose); groups: {bone name: depth group}; sp: grid spacing (art px)."""

    def __init__(self, path, bones, groups, sp=12, shade=None, max_inf=3, wtrim=None, chains=None, erase=(), axial=None, neck=None):
        rgba = key_alpha(path)
        for poly in erase:  # hand-marked bits of the drawing to drop (bind.json "erase": {"body": [[[x, y], ...]]})
            m = np.zeros(rgba.shape[:2], np.uint8); cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
            rgba[m > 0] = 0
        self.size = rgba.shape[1], rgba.shape[0]
        self.names = [b[0] for b in bones]
        self.A0 = np.array([b[1] for b in bones], float); self.B0 = np.array([b[2] for b in bones], float)
        al = rgba[:, :, 3] > 0
        self.alpha = al
        # weights are measured to bone segments trimmed at their root by wtrim[name] (fraction of the length): the
        # pelvis then belongs to the spine and a thigh only starts pulling below it; chains = {name: id}: a vertex
        # follows one chain at most besides chain 0 (a vertex of one leg never blends with the other leg's bones)
        self.wtrim = wtrim or {}; self.chains = chains or {}
        # axial = {chain id: (root bone, [bones root->tip])}: that chain's weights are ramps along the bone axis
        # (Spine's recipe) instead of the distance falloff: 100% one bone mid-segment, a short smoothstep band at
        # each joint sized by the limb width there (thick thighs no longer pull shin weight far up the thigh)
        self.axial = axial or {}
        self.neck = neck  # (neck bone, root bone): the stub above the root bone's tip follows the neck (_neck_weights)
        self._mesh(al, sp)
        self._weights(al, sp, max_inf)
        g = np.array([groups[n] for n in self.names])
        # a triangle's depth group: the group of the bone with most weight over its three corners
        tw = np.zeros((len(self.tris), len(self.names)))
        for k in range(self.idx.shape[1]):
            for c in range(3):
                np.add.at(tw, (np.arange(len(self.tris)), self.idx[self.tris[:, c], k]), self.w[self.tris[:, c], k])
        self.tgroup = g[tw.argmax(1)]
        # baked shading (far leg darker, fading smoothly into the pelvis): per-vertex factor, spread over the texture
        if shade:
            f = np.ones(len(self.V))
            for k in range(self.idx.shape[1]):
                f -= self.w[:, k] * np.array([1 - shade.get(n, 1.0) for n in self.names])[self.idx[:, k]]
            grid = np.ones(self.gshape, np.float32); grid[self.gi, self.gj] = f[:len(self.gi)]
            fmap = cv2.resize(grid, (self.gshape[1] * sp, self.gshape[0] * sp), interpolation=cv2.INTER_LINEAR)
            full = np.ones(al.shape, np.float32)
            y0, x0 = self.gorigin[1], self.gorigin[0]
            # grid vertex (i, j) sits at (x0 + j sp, y0 + i sp); resize puts sample centres at half cells: shift by sp/2
            fmap = np.pad(fmap, ((sp // 2, 0), (sp // 2, 0)), mode='edge')
            hh, ww = min(full.shape[0] - y0, fmap.shape[0]), min(full.shape[1] - x0, fmap.shape[1])
            full[y0:y0 + hh, x0:x0 + ww] = fmap[:hh, :ww]
            rgba = rgba.astype(np.float32); rgba[:, :, :3] *= full[:, :, None]
            rgba = rgba.clip(0, 255).astype(np.uint8)
        self.rgba = rgba
        self._tex = {}

    # -- mesh: a regular grid over the drawing, cells kept where the drawing is, never across a gap
    def _mesh(self, al, sp):
        ys, xs = np.nonzero(al)
        x0, y0 = max(int(xs.min()) - sp, 0), max(int(ys.min()) - sp, 0)
        x1, y1 = int(xs.max()) + sp, int(ys.max()) + sp
        nj, ni = (x1 - x0) // sp + 1, (y1 - y0) // sp + 1
        self.gorigin = (x0, y0); self.gshape = (ni, nj)
        H, W = al.shape
        near = cv2.dilate(al.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(sp * 0.8) + 1,) * 2)) > 0
        gx = x0 + np.arange(nj) * sp; gy = y0 + np.arange(ni) * sp
        inside = np.zeros((ni, nj), bool)
        for i, y in enumerate(gy):
            if y >= H: continue
            ok = gx < W
            inside[i, ok] = near[y, gx[ok]]
        vid = -np.ones((ni, nj), int)
        self.gi, self.gj = np.nonzero(inside)
        vid[self.gi, self.gj] = np.arange(len(self.gi))
        self.V = np.stack([x0 + self.gj * sp, y0 + self.gi * sp], 1).astype(float)
        # a cell is used if it holds any drawing (integral image) ...
        ii = cv2.integral(al.astype(np.uint8))
        def cell_any(i, j):
            ya, xa = min(y0 + i * sp, H), min(x0 + j * sp, W); yb, xb = min(ya + sp + 1, H), min(xa + sp + 1, W)
            return ii[yb, xb] - ii[ya, xb] - ii[yb, xa] + ii[ya, xa] > 0
        self._gap_cache = {}
        tris = []
        for i in range(ni - 1):
            for j in range(nj - 1):
                a, b, c, d = vid[i, j], vid[i, j + 1], vid[i + 1, j], vid[i + 1, j + 1]
                if min(a, b, c, d) < 0 or not cell_any(i, j): continue
                for t in ((a, b, d), (a, d, c)):
                    if not any(self._gap(t[k], t[(k + 1) % 3]) for k in range(3)): tris.append(t)
        self.tris = np.array(tris, int)
        self.vid = vid

    def _gap(self, u, v):
        """Does the straight edge u-v leave the drawing and come back (crossing a gap between two legs)?"""
        key = (min(u, v), max(u, v))
        if key in self._gap_cache: return self._gap_cache[key]
        p, q = self.V[u], self.V[v]
        n = int(max(4, np.linalg.norm(q - p) / 2))
        t = np.linspace(0, 1, n)
        pts = (p[None] + (q - p)[None] * t[:, None]).round().astype(int)
        H, W = self.alpha.shape
        pts[:, 0] = pts[:, 0].clip(0, W - 1); pts[:, 1] = pts[:, 1].clip(0, H - 1)
        s = self.alpha[pts[:, 1], pts[:, 0]].astype(int)
        on = np.nonzero(s)[0]
        res = bool(len(on) >= 2 and (s[on[0]:on[-1] + 1] == 0).sum() >= 2)
        self._gap_cache[key] = res
        return res

    # -- weights: geodesic distance (through the drawing) to each bone, inverse-power falloff, smoothed
    def _weights(self, al, sp, max_inf):
        nv, nb = len(self.V), len(self.names)
        ni, nj = self.gshape
        nbrs = [[] for _ in range(nv)]
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for v in range(nv):
                i, j = self.gi[v] + di, self.gj[v] + dj
                if 0 <= i < ni and 0 <= j < nj and self.vid[i, j] >= 0:
                    u = self.vid[i, j]
                    if not self._gap(u, v):
                        c = sp * math.hypot(di, dj); nbrs[v].append((u, c)); nbrs[u].append((v, c))
        self.nbrs = nbrs
        D = np.full((nv, nb), np.inf)
        for b in range(nb):
            a_w = self.A0[b] + (self.B0[b] - self.A0[b]) * self.wtrim.get(self.names[b], 0.0)
            c, d0 = _seg_proj(self.V, a_w, self.B0[b])
            src = np.nonzero(d0 <= sp * 0.8)[0]
            if not len(src): src = [int(d0.argmin())]
            dist = np.full(nv, np.inf); h = []
            for s in src: dist[s] = d0[s]; h.append((d0[s], int(s)))
            heapq.heapify(h)
            while h:
                d, v = heapq.heappop(h)
                if d > dist[v]: continue
                for u, c_ in nbrs[v]:
                    nd = d + c_
                    if nd < dist[u]: dist[u] = nd; heapq.heappush(h, (nd, u))
            # vertices cut off from the bone (separate specks) fall back to the straight distance, far
            dist[~np.isfinite(dist)] = d0[~np.isfinite(dist)] * 3
            D[:, b] = dist
        self.D = D
        W = 1.0 / (D + sp * 1.5) ** 4
        W /= W.sum(1, keepdims=True)
        if self.neck:  # the neck takes its weights from _neck_weights only: fold its falloff back into the root
            ix = {n: i for i, n in enumerate(self.names)}
            W[:, ix[self.neck[1]]] += W[:, ix[self.neck[0]]]; W[:, ix[self.neck[0]]] = 0
        for _ in range(3):  # smooth across the graph
            W2 = W.copy()
            for v in range(nv):
                if nbrs[v]: W2[v] = 0.5 * W[v] + 0.5 * np.mean([W[u] for u, _ in nbrs[v]], 0)
            W = W2
        self.vchain = np.zeros(nv, int)
        if self.chains:
            W = self._split_chains(W)
        if self.axial:
            W = self._axial_weights(W, al)
        if self.neck:
            W = self._neck_weights(W, al)
        idx = np.argsort(-W, 1)[:, :max_inf]
        w = np.take_along_axis(W, idx, 1)
        w[w < 0.02] = 0
        w /= w.sum(1, keepdims=True)
        self.idx, self.w = idx, w
        self.C = np.stack([_seg_proj(self.V, self.A0[b], self.B0[b])[0] for b in range(nb)], 1)  # (nv, nb, 2)

    def _split_chains(self, W):
        """Cut the mesh between chains (the two legs): every triangle belongs to one chain, and a corner shared with
        the other leg's triangles is duplicated with weights restricted to that triangle's chain (+ the spine).
        Where the legs touch in the bind drawing (inner thighs) they then come apart cleanly instead of stretching a
        fan of texture across the crotch; up in the pelvis both copies are mostly spine and stay together."""
        ch = np.array([self.chains.get(n, 0) for n in self.names])
        ids = sorted(set(ch) - {0})
        if len(ids) < 2: return W
        tot = np.stack([W[:, ch == c].sum(1) for c in ids], 1)
        def restrict(row, c):
            r = np.where((ch == 0) | (ch == c), row, 0.0); return r / max(r.sum(), 1e-9)
        vch = np.array(ids)[tot.argmax(1)]
        Wn = np.array([restrict(W[v], vch[v]) for v in range(len(W))])
        self.vchain = vch.copy()
        extraV, extraW, dup = [], [], {}
        n0 = len(W)
        tris = self.tris.copy()
        tch = np.array([ids[int(tot[t].sum(0).argmax())] for t in tris])
        # no islands: a triangle whose edge-neighbours mostly belong to the other leg joins them (a lone sliver of
        # one leg inside the other would tear off as a loose speck)
        edges = {}
        for t, tri in enumerate(tris):
            for k in range(3):
                e = (min(tri[k], tri[(k + 1) % 3]), max(tri[k], tri[(k + 1) % 3])); edges.setdefault(e, []).append(t)
        nbr = [[] for _ in tris]
        for ts_ in edges.values():
            if len(ts_) == 2: nbr[ts_[0]].append(ts_[1]); nbr[ts_[1]].append(ts_[0])
        for _ in range(4):
            changed = False
            for t in range(len(tris)):
                if not nbr[t]: continue
                lab = [tch[u] for u in nbr[t]]
                other = [c for c in lab if c != tch[t]]
                if len(other) * 2 > len(lab):
                    tch[t] = max(set(other), key=other.count); changed = True
            if not changed: break
        for t in range(len(tris)):
            tc = tch[t]
            for k in range(3):
                v = tris[t, k]
                if vch[v] == tc: continue
                if (v, tc) not in dup:
                    dup[(v, tc)] = n0 + len(extraV); extraV.append(self.V[v]); extraW.append(restrict(W[v], tc))
                tris[t, k] = dup[(v, tc)]
        self.tris = tris
        if extraV:
            self.V = np.concatenate([self.V, np.array(extraV)])
            Wn = np.concatenate([Wn, np.array(extraW)])
            ext = np.zeros(len(extraV), int)
            for (v, tc), i in dup.items(): ext[i - n0] = tc
            self.vchain = np.concatenate([self.vchain, ext])
        return Wn

    # joint blend half-widths (x the limb width at the joint) and centre offsets down the child bone (guide 3.2)
    AXIAL_H = (0.5, 0.35, 0.25)    # hip, knee, ankle
    AXIAL_OFF = (0.15, 0.0, 0.0)
    axial_keep_root = True

    def _neck_weights(self, W, al):
        """The neck stub (above the shoulder line, within the stub's width) moves from the root bone to the neck:
        a smoothstep ramp across the shoulder line (half-width 0.4 x the neck width) times a sideways fade, so the
        shoulder tops beside the neck stay on the chest."""
        ix = {n: i for i, n in enumerate(self.names)}
        nb, rb = ix[self.neck[0]], ix[self.neck[1]]
        base, tip = self.A0[nb], self.B0[nb]
        L = float(np.linalg.norm(tip - base)) or 1.0
        up = (tip - base) / L; side = np.array([-up[1], up[0]])
        wn = min(_seg_width(al, base, tip, 0.2, 0.7), 1.2 * L)
        ss = lambda x: (lambda c: c * c * (3 - 2 * c))(np.clip(x, 0.0, 1.0))
        t = (self.V - base) @ up; lat = np.abs((self.V - base) @ side)
        h = 0.4 * wn
        r = ss((t + h) / (2 * h)) * (1 - ss((lat - 0.5 * wn) / (0.35 * wn)))
        W = W.copy()
        move = W[:, rb] * r
        W[:, rb] -= move; W[:, nb] += move
        self.neck_info = dict(width=round(wn, 1), length=round(L, 1))
        return W

    def _axial_weights(self, W, al):
        """Re-weight every vertex of an axial chain with nested smoothstep ramps across planes at its joints:
        root|b1 at b1's root (normal along b1), b1|b2 at their joint (normal = the bisector of both directions),
        b2|b3 likewise along b2. Vertices keep their chain (geodesic split); weights no longer depend on how far a
        vertex sits from the bone axis."""
        ix = {n: i for i, n in enumerate(self.names)}
        ss = lambda x: (lambda c: c * c * (3 - 2 * c))(np.clip(x, 0.0, 1.0))
        unit = lambda v: v / (np.linalg.norm(v) or 1.0)
        W = W.copy()
        self.axial_info = {}
        for cid, (root, chain) in self.axial.items():
            sel = np.nonzero(self.vchain == cid)[0]
            if not len(sel) or len(chain) != 3: continue
            P = self.V[sel]
            b = [ix[n] for n in chain]
            A0, B0 = self.A0, self.B0
            u = [unit(B0[k] - A0[k]) for k in b]
            Ls = [float(np.linalg.norm(B0[k] - A0[k])) for k in b]
            w_hip = _seg_width(al, A0[b[0]], B0[b[0]], 0.3, 0.8)
            w_kne = 0.5 * (_seg_width(al, A0[b[0]], B0[b[0]], 0.6, 0.95) + _seg_width(al, A0[b[1]], B0[b[1]], 0.05, 0.4))
            w_ank = _seg_width(al, A0[b[1]], B0[b[1]], 0.5, 0.9)
            # touching legs in the bind read as one wide run: never wider than 0.6 of the segment
            ws = [min(w_hip, 0.6 * Ls[0]), min(w_kne, 0.6 * min(Ls[0], Ls[1])), min(w_ank, 0.6 * Ls[1])]
            planes = [(A0[b[0]], u[0]), (A0[b[1]], unit(u[0] + u[1])), (A0[b[2]], u[1])]
            r = []
            for (j, n), wj, h, off in zip(planes, ws, self.AXIAL_H, self.AXIAL_OFF):
                t = (P - j) @ n
                r.append(ss((t - off * wj + h * wj) / (2 * h * wj)))
            rows = np.zeros((len(sel), W.shape[1]))
            # the pelvis keeps its old share (the trimmed-thigh falloff holds the briefs / belt on the pelvis in
            # wide stances; the hip ramp let the thighs drag it); the leg's share is split by the knee/ankle ramps
            ws_ = W[sel, ix[root]] if self.axial_keep_root else 1 - r[0]
            leg = 1 - ws_
            rows[:, ix[root]] = ws_
            rows[:, b[0]] = leg * (1 - r[1])
            rows[:, b[1]] = leg * r[1] * (1 - r[2])
            rows[:, b[2]] = leg * r[1] * r[2]
            W[sel] = rows
            self.axial_info[cid] = dict(widths=[round(x, 1) for x in ws])
        return W

    # -- textures, prescaled to about the render scale (premultiplied RGBA), mirrored for fighters facing left
    def tex(self, scale, mirror, shade=1.0):
        key = (round(scale, 3), mirror, shade)
        if key not in self._tex:
            a = self.rgba.astype(np.float32)
            a[:, :, :3] *= a[:, :, 3:4] / 255.0 * shade
            if mirror: a = a[:, ::-1]
            s = min(1.0, scale)
            if s < 1.0: a = cv2.resize(a, (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
            self._tex[key] = (np.ascontiguousarray(a.clip(0, 255).astype(np.uint8)), a.shape[1] / self.size[0])
        return self._tex[key]

    def mirror_x(self, p):
        p = np.array(p, float).copy(); p[..., 0] = self.size[0] - 1 - p[..., 0]; return p

    # -- deformation
    def deform(self, bones, mirror):
        """bones: {name: (a0, a, phi, P)} with T(p) = a + R(phi) P (p - a0) in bind (art px, mirrored if `mirror`)
        -> deformed vertex positions (canvas px)."""
        V = self.mirror_x(self.V) if mirror else self.V
        C = self.mirror_x(self.C) if mirror else self.C
        A0 = np.array([bones[n][0] for n in self.names]); A = np.array([bones[n][1] for n in self.names])
        PHI = np.array([bones[n][2] for n in self.names]); P = np.array([bones[n][3] for n in self.names])
        idx, w = self.idx, self.w
        n = len(V)
        c = C[np.arange(n)[:, None], idx]                                   # (n,K,2) projection on each bone
        Rk = _rot(PHI[idx])                                                 # (n,K,2,2)
        Tc = A[idx] + np.einsum('nkij,nkjl,nkl->nki', Rk, P[idx], c - A0[idx])
        anchor = (w[..., None] * Tc).sum(1); pivot = (w[..., None] * c).sum(1)
        dom = PHI[idx[:, 0]]
        d = (PHI[idx] - dom[:, None] + np.pi) % (2 * np.pi) - np.pi
        th = dom + (w * d).sum(1)
        Pb = (w[..., None, None] * P[idx]).sum(1)
        return anchor + np.einsum('nij,njk,nk->ni', _rot(th), Pb, V - pivot)


def bone_xf(a0, b0, a, b, across, min_along=0.5):
    """Bone transform taking bind segment a0-b0 onto a-b: stretched along the bone to its length, `across` wide."""
    a0, b0, a, b = (np.asarray(x, float) for x in (a0, b0, a, b))
    t0 = math.atan2(*(b0 - a0)[::-1]); t = math.atan2(*(b - a)[::-1])
    L0 = np.linalg.norm(b0 - a0) or 1.0
    along = max(np.linalg.norm(b - a) / L0, across * min_along)
    R0 = _rot(np.array(t0))
    P = R0 @ np.diag([along, across]) @ R0.T
    return a0, a, t - t0, P


KNEE_STATS = None  # debug: a list collects (flexion deg, wrong-side offset / leg length, action)


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _knee_rule(h, kne, ank, face, max_flex=None, noise=0.12, lean=0.03):
    """Knee bend direction and range (guide 4.3 / 9.3), canvas px. Flexion > 0 = the knee points the way the
    fighter faces (true for every leg pose: the shin folds behind the thigh). A knee on the wrong side of the
    hip-ankle line by less than `noise` x leg length is detector noise (knock-knees, micro back-bent knees): it goes
    onto the line plus a small forward lean. Bigger wrong-way bends are left alone (label swaps, odd frames).
    Flexion past max_flex (off by default: SF3 tucks and kneels fold past 150 deg, and unfolding them pushed the
    feet out from under the body) turns the shin back to max_flex about the knee."""
    h, kne, ank = (np.asarray(x, float) for x in (h, kne, ank))
    t, s = kne - h, ank - kne
    Lt, Ls = np.linalg.norm(t), np.linalg.norm(s)
    if Lt < 1e-6 or Ls < 1e-6: return kne, ank
    line = ank - h; Ll = np.linalg.norm(line)
    flex_sign = face * _cross(t, s)
    flex = math.degrees(math.acos(float(np.clip(t @ s / (Lt * Ls), -1, 1))))
    act = ''
    if Ll > 1e-6 and flex_sign < 0:
        u = line / Ll
        off = abs(_cross(u, t)) / (Lt + Ls)
        if off < noise:
            n = np.array([-u[1], u[0]])
            if face * _cross(n, u) < 0: n = -n  # n puts the knee in front: face * cross(knee - h, ank - knee) > 0
            kne = h + u * float(t @ u) + n * lean * (Lt + Ls)
            act = 'straight'
    elif max_flex is not None and flex_sign > 0 and flex > max_flex:
        rot = math.radians(flex - max_flex) * (1 if _cross(t, s) < 0 else -1)
        c_, s_ = math.cos(rot), math.sin(rot)
        ank = kne + np.array([[c_, -s_], [s_, c_]]) @ s
        act = 'clamp'
    if KNEE_STATS is not None:
        KNEE_STATS.append((round(flex * (1 if flex_sign >= 0 else -1), 1),
                           round(abs(_cross(line / (Ll or 1), t)) / (Lt + Ls), 3), act))
    return kne, ank


def fixed_xf(a0, a, phi, s):
    return np.asarray(a0, float), np.asarray(a, float), phi, np.eye(2) * s


def raster(layer, Vd, mask, scale, mirror, shape, shade=1.0):
    """Draw the triangles `mask` of a deformed layer. Returns (premultiplied RGBA crop, x0, y0) or None."""
    tris = layer.tris[mask]
    if not len(tris): return None
    tex, ts = layer.tex(scale, mirror, shade)
    V = layer.mirror_x(layer.V) if mirror else layer.V
    dst = Vd[tris]; src = V[tris] * ts
    H, W = shape
    lo = np.floor(dst.reshape(-1, 2).min(0)).astype(int) - 1; hi = np.ceil(dst.reshape(-1, 2).max(0)).astype(int) + 2
    x0, y0 = max(lo[0], 0), max(lo[1], 0); x1, y1 = min(hi[0], W), min(hi[1], H)
    if x1 <= x0 or y1 <= y0: return None
    D = np.concatenate([dst, np.ones((len(tris), 3, 1))], 2)
    det = np.linalg.det(D)
    ok = np.abs(det) > 1e-3
    X = np.zeros((len(tris), 3, 2)); X[ok] = np.linalg.solve(D[ok], src[ok])
    ids = np.zeros((y1 - y0, x1 - x0), np.int32)
    pts = ((dst - [x0, y0]) * 16).round().astype(np.int32)
    for k in np.nonzero(ok)[0]:
        cv2.fillConvexPoly(ids, pts[k], int(k) + 1, cv2.LINE_8, 4)
    yy, xx = np.nonzero(ids)
    if not len(yy): return None
    t = ids[yy, xx] - 1
    gx, gy = (xx + x0).astype(np.float32), (yy + y0).astype(np.float32)
    mx = np.full(ids.shape, -10, np.float32); my = np.full(ids.shape, -10, np.float32)
    mx[yy, xx] = X[t, 0, 0] * gx + X[t, 1, 0] * gy + X[t, 2, 0]
    my[yy, xx] = X[t, 0, 1] * gx + X[t, 1, 1] * gy + X[t, 2, 1]
    out = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, x0, y0


def over(cv, buf):
    """Composite a premultiplied crop onto the PIL canvas."""
    if buf is None: return
    a, x0, y0 = buf
    al = a[:, :, 3:4].astype(np.float32)
    rgb = np.where(al > 0, a[:, :, :3].astype(np.float32) * 255.0 / np.maximum(al, 1), 0)
    im = Image.fromarray(np.dstack([rgb.clip(0, 255), al]).astype(np.uint8), 'RGBA')
    cv.alpha_composite(im, (int(x0), int(y0)))


# ---------------------------------------------------------------------------------------------------- model

class Model:
    """A mascot's mesh layers + bind skeleton. k = old puppet torso length / mesh body torso length, so the mesh
    scale is G * k (G stays the puppet's scale: hooks and loose pieces use it)."""

    def __init__(self, mascot, torso_len=None):
        d = MESH / mascot
        bj = json.loads((d / 'bind.json').read_text())
        self.bind = bj
        b = {k: np.array(v, float) for k, v in bj['body'].items() if not isinstance(v, dict) and k != 'size'}
        n, f = ({k: np.array(v, float) for k, v in bj['body'][s].items()} for s in ('near', 'far'))
        self.b, self.n, self.f = b, n, f
        bones = [('spine', b['sho'], b['hip'])]
        # the neck stub gets its own bone (shoulder line -> top of the stub): it keeps its length when the torso is
        # squashed and turns part way toward the head, so hunched poses don't sink the head into the chest (9.7)
        self.neck_bone = bool(bj.get('neck_bone', NECK_BONE))
        if self.neck_bone:
            self.neck0 = _neck_top(key_alpha(d / 'body.png')[:, :, 3] > 0, b['sho'])
            bones.append(('neck', b['sho'], self.neck0))
        for p, L in (('n', n), ('f', f)):
            bones += [(f'{p}_thigh', L['hip'], L['kne']), (f'{p}_shin', L['kne'], L['ank']), (f'{p}_foot', L['ank'], L['toe'])]
        groups = {'spine': 1, 'neck': 1, 'n_thigh': 2, 'n_shin': 2, 'n_foot': 2, 'f_thigh': 0, 'f_shin': 0, 'f_foot': 0}
        sp_b = int(bj.get('sp', 12))
        self.body = Layer(d / 'body.png', bones, groups, sp=sp_b,
                          shade={'f_thigh': 0.8, 'f_shin': 0.8, 'f_foot': 0.8},
                          wtrim={'n_thigh': 0.2, 'f_thigh': 0.2},
                          chains={'n_thigh': 1, 'n_shin': 1, 'n_foot': 1, 'f_thigh': 2, 'f_shin': 2, 'f_foot': 2},
                          erase=bj.get('erase', {}).get('body', ()),
                          axial={1: ('spine', ['n_thigh', 'n_shin', 'n_foot']), 2: ('spine', ['f_thigh', 'f_shin', 'f_foot'])}
                          if bj.get('axial_legs', AXIAL_LEGS) else None,
                          neck=('neck', 'spine') if self.neck_bone else None)
        a = {k: np.array(v, float) for k, v in bj['arm'].items() if k != 'size'}
        self.a = a
        self.arm = Layer(d / 'arm.png', [('upper', a['sho'], a['elb']), ('fore', a['elb'], a['wri']),
                                         ('hand', a['wri'], a['end'])], {'upper': 0, 'fore': 1, 'hand': 1}, sp=sp_b,
                         erase=bj.get('erase', {}).get('arm', ()))
        # a mascot with two different arms (PILL: green near half, white far half) has arm_far.png + bind 'arm_far'
        self.arm_far, self.af = self.arm, a
        if (d / 'arm_far.png').exists() and 'arm_far' in bj:
            af = {k: np.array(v, float) for k, v in bj['arm_far'].items() if k != 'size'}
            self.af = af
            self.arm_far = Layer(d / 'arm_far.png', [('upper', af['sho'], af['elb']), ('fore', af['elb'], af['wri']),
                                                     ('hand', af['wri'], af['end'])], {'upper': 0, 'fore': 1, 'hand': 1},
                                 sp=sp_b, erase=bj.get('erase', {}).get('arm_far', ()))
        self.tlen0 = float(np.linalg.norm(b['hip'] - b['sho']))
        self.k = (torso_len / self.tlen0) if torso_len else 1.0
        self.arm_len0 = float(np.linalg.norm(a['wri'] - a['sho']))
        self.arm_width = float(self.arm.alpha.sum(1).max())
        self.sole = {s: float(L['sole']) for s, L in (('n', n), ('f', f))}
        # torso half-width at the shoulder row (art px), for keeping shoulders on the torso
        row = int(np.clip(b['sho'][1] + 0.08 * self.tlen0, 0, self.body.alpha.shape[0] - 1))
        xs = np.nonzero(self.body.alpha[row])[0]
        self.sho_half = float((xs.max() - xs.min()) / 2) if len(xs) else self.tlen0 * 0.3
        # where a tail hangs: on the back of the pelvis (bind faces right, so the back is toward -x), 40% of the way
        # from the hip centre to the drawing's back edge, so the tail's root always sits on the body
        row = int(np.clip(b['hip'][1], 0, self.body.alpha.shape[0] - 1))
        xs = np.nonzero(self.body.alpha[row])[0]
        back = float(xs.min()) if len(xs) else b['hip'][0] - 0.2 * self.tlen0
        self.tail0 = np.array([b['hip'][0] + 0.4 * (back - b['hip'][0]), b['hip'][1]])
        # top of the neck stub (the head sits on it)
        if not self.neck_bone: self.neck0 = _neck_top(self.body.alpha, b['sho'])
        # widths of the drawing (art px) for matching the cut-out puppet's tuned widths (attach())
        bal, aal = self.body.alpha, self.arm.alpha
        self.w = dict(chest=_seg_width(bal, b['sho'], b['hip'], 0.25, 0.6),
                      thigh=np.mean([_seg_width(bal, L['hip'], L['kne'], 0.3, 0.8) for L in (n, f)]),
                      shin=np.mean([_seg_width(bal, L['kne'], L['ank'], 0.2, 0.8) for L in (n, f)]),
                      upper=_seg_width(aal, a['sho'], a['elb'], 0.3, 0.8), fore=_seg_width(aal, a['elb'], a['wri'], 0.2, 0.8),
                      fist=_seg_width(aal, a['wri'], a['end'], 0.0, 1.0, stat=np.max),
                      foot=np.mean([_foot_len(bal, L) for L in (n, f)]))
        self.fit = {k: 1.0 for k in self.w}  # cut-out width / mesh width per part (attach())


def _neck_top(al, sho):
    """Top of the neck stub (art px): the column centre 3 rows below the drawing's top."""
    ys, xs = np.nonzero(al)
    top = int(ys.min()); cols = np.nonzero(al[top + 3])[0]
    return np.array([float(cols.mean()) if len(cols) else float(sho[0]), float(top + 3)])


def _seg_width(al, a, b, lo, hi, stat=np.median):
    """Width of the drawing across the segment a-b (horizontal run through each sample, times the cosine of the
    segment's tilt), over the fraction lo..hi of its length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; L = np.linalg.norm(d) or 1.0
    cos = abs(d[1]) / L if abs(d[1]) > 1e-6 else 1.0
    out = []
    for t in np.linspace(lo, hi, 15):
        x, y = (a + d * t).round().astype(int)
        if not (0 <= y < al.shape[0] and 0 <= x < al.shape[1]): continue
        r = _runs(al[y], 2)
        r = [q for q in r if q[0] <= x <= q[1]] or (sorted(r, key=lambda q: min(abs(q[0] - x), abs(q[1] - x)))[:1])
        if r: out.append((r[0][1] - r[0][0] + 1) * cos)
    return float(stat(out)) if out else 1.0


def _foot_len(al, L):
    """Heel-to-toe length of a foot (art px): the run at the sole rows nearest the toe point."""
    y = int(min(L['sole'], al.shape[0] - 1) - 0.25 * max(1.0, L['sole'] - L['ank'][1]))
    r = _runs(al[y], 3)
    if not r: return 1.0
    x = L['toe'][0]
    q = min(r, key=lambda t: 0 if t[0] <= x <= t[1] else min(abs(t[0] - x), abs(t[1] - x)))
    return float(q[1] - q[0] + 1)


def _part_width(part, lo=0.25, hi=0.75, stat=np.median):
    """The cut-out part's width between its pivots (part px)."""
    al = np.asarray(part.im)[:, :, 3] > 0
    ys = np.linspace(part.a[1] + (part.b[1] - part.a[1]) * lo, part.a[1] + (part.b[1] - part.a[1]) * hi, 15)
    w = [al[int(np.clip(y, 0, al.shape[0] - 1))].sum() for y in ys]
    return float(stat(w)) if w else 1.0


def attach(parts, cfg, pose, mascot):
    """skin.setup(): add the mesh model to the puppet parts (parts.mesh), or None."""
    if cfg.get('renderer', 'mesh') != 'mesh' or not available(mascot): return None
    t = parts['torso']
    m = Model(mascot, float(np.linalg.norm(t.b - t.a)))
    # Widths follow the cut-out puppet's, which each fighter's hook tuned to its SF3 body (CONFIG limb_w, foot,
    # fist): fit[part] * G = the across-scale that makes the mesh part as wide as the cut-out part was drawn.
    lw = cfg.get('limb_w', 0.85)
    old = dict(chest=(_part_width(t), 1.0), thigh=(_part_width(parts['thigh']), lw),
               shin=(_part_width(parts['shin']), lw), upper=(_part_width(parts['upperarm']), lw),
               fore=(_part_width(parts['forearm']), lw),
               fist=(_part_width(parts['fist'], 0.0, 1.0, np.max), cfg.get('fist', 0.7)),
               foot=(float((np.asarray(parts['foot'].im)[:, :, 3] > 0).any(0).sum()), cfg.get('foot', 0.85)))
    for k, (w, sc) in old.items():
        m.fit[k] = w * sc / max(m.w[k], 1.0)
    # parts of one drawing keep their drawn proportions within limits (a hoof measured as a 60 px foot must not
    # blow the foot up 8x): every fit stays within 0.75-1.35x the median of its drawing's main parts
    for grp, ref in ((('chest', 'thigh', 'shin', 'foot'), ('chest', 'thigh', 'shin')), (('upper', 'fore', 'fist'), ('upper', 'fore'))):
        med = float(np.median([m.fit[k] for k in ref]))
        for k in grp:
            m.fit[k] = float(np.clip(m.fit[k], 0.75 * med, 1.35 * med))
    return m


# ---------------------------------------------------------------------------------------------------- compose

def compose_mesh(J, parts, G, cfg, size, R, Mg, paste_affine, paste_head):
    """Draw one frame with the mesh layers onto a (size + 2M) * R canvas (same contract as skin.compose)."""
    m = parts.mesh
    W, H = size
    cv = Image.new('RGBA', ((W + 2 * Mg) * R, (H + 2 * Mg) * R))
    shape = (cv.height, cv.width)
    f = J['face']; mir = f < 0
    S = lambda p: (np.asarray(p, float) + Mg) * R
    J = dict(J); J['l'] = dict(J['l']); J['r'] = dict(J['r'])
    if J.get('near') in ('l', 'r'): near = J['near']
    elif cfg.get('near_rule', 'physical') == 'legacy': near = 'l' if f > 0 else 'r'
    else: near = 'r' if f > 0 else 'l'
    far = 'r' if near == 'l' else 'l'
    fit = m.fit
    tw = G * fit['chest'] * cfg.get('mesh_torso_w', 1.0)
    lw_t = G * fit['thigh'] * cfg.get('mesh_leg_w', 1.0); lw_s = G * fit['shin'] * cfg.get('mesh_leg_w', 1.0)
    fw = G * fit['foot'] * cfg.get('mesh_foot', 1.0)
    aw_u = G * fit['upper'] * cfg.get('mesh_arm_w', 1.0); aw_f = G * fit['fore'] * cfg.get('mesh_arm_w', 1.0)
    hand_s = G * fit['fist'] * cfg.get('mesh_fist', 1.0)
    aw = aw_u
    ts = max(tw, lw_t, lw_s, aw_u, aw_f, hand_s, fw) * 1.4  # texture prescale: a little over the render scale
    bb = (lambda p: m.body.mirror_x(p)) if mir else (lambda p: np.asarray(p, float))

    sho, hip = S(J['sho']), S(J['hip'])
    tor = hip - sho; tang = math.atan2(tor[0], tor[1]); tlen = np.linalg.norm(tor) or 1.0
    ex = np.array([math.cos(tang), -math.sin(tang)]); ey = np.array([math.sin(tang), math.cos(tang)])
    # shoulders and hips stay on the body (same limits as the cut-out: group D's connect())
    for side in 'lr':
        for key, c, lo_, hi_ in (('sho', sho, -0.08, 0.3), ('hip', hip, -0.15, 0.15)):
            if J[side].get(key) is None: continue
            q = S(J[side][key]); v = (q - c) @ ey; vv = float(np.clip(v, lo_ * tlen, hi_ * tlen))
            if vv != v: J[side][key] = (q + ey * (vv - v)) / R - Mg
        q = S(J[side]['sho']); d = (q - sho) @ ex
        lim = max(m.sho_half * tw - 0.35 * m.w['upper'] * aw / 2, 0.0)
        dd = float(np.clip(d, -lim, lim))
        if dd != d: J[side]['sho'] = (q + ex * (dd - d)) / R - Mg

    # cape and tail hang behind everything (rigid cut-out parts, as before)
    P = {k: (v if f > 0 else v.flipped()) for k, v in parts.items() if k in ('cape', 'tail', 'head')}
    cs, tsc = G * cfg.get('cape', 0.75), G * cfg.get('tail', 0.75)
    if 'cape' in P: paste_affine(cv, P['cape'], sho, tang * 0.6 - f * 0.25, cs, cs, 0.9)
    if J.get('tail_scale') is not None: tsc = G * J['tail_scale']
    t_ang = J['tail_angle'] if J.get('tail_angle') is not None else tang - f * 0.2

    # ---- body bones
    bones = {'spine': bone_xf(bb(m.b['sho']), bb(m.b['hip']), sho, hip, tw, min_along=0.6)}
    sp_a0, sp_a, sp_phi, sp_P = bones['spine']
    if m.neck_bone:
        # the neck keeps its drawn length (never squashed with a foreshortened torso) and turns part way from the
        # chest's direction toward the head (at most 35 deg), so a hunched fighter's head sits ahead of the shoulders
        # instead of sinking into the chest
        n_a0, n_b0 = bb(m.b['sho']), bb(m.neck0)
        d0 = _rot(np.array(sp_phi)) @ (n_b0 - n_a0)                     # stub direction carried by the chest
        hd = S(J['head']) - sho
        turn = 0.0
        if np.linalg.norm(hd) > 1e-6 and np.linalg.norm(d0) > 1e-6:
            turn = math.atan2(_cross(d0, hd), float(d0 @ hd)) * cfg.get('mesh_neck_mix', 0.5)
            turn = float(np.clip(turn, -math.radians(35), math.radians(35)))
        tip = sho + _rot(np.array(turn)) @ d0 / (np.linalg.norm(d0) or 1.0) * np.linalg.norm(n_b0 - n_a0) * tw
        bones['neck'] = bone_xf(n_a0, n_b0, sho, tip, tw)
    if 'tail' in P:  # rooted on the back of the deformed pelvis
        root = sp_a + _rot(np.array(sp_phi)) @ sp_P @ (bb(m.tail0) - sp_a0)
        paste_affine(cv, P['tail'], root, t_ang, tsc, tsc)
    ground = J.get('ground') if cfg.get('plant', True) and J.get('plant', True) else None
    follow = cfg.get('mesh_hip_follow', 0.35)
    for tag, side, L0 in (('n', near, m.n), ('f', far, m.f)):
        L = J[side]
        # the thigh starts at the hip joint carried by the pelvis, pulled part way to the detected hip
        rig = hip + _rot(np.array(sp_phi)) @ sp_P @ (bb(L0['hip']) - bb(m.b['hip']))
        h = rig * (1 - follow) + S(L['hip']) * follow
        kne, ank = S(L['kne']), S(L['ank'])
        if cfg.get('mesh_knee_rule', True):
            kne, ank = _knee_rule(h, kne, ank, f)
        bones[f'{tag}_thigh'] = bone_xf(bb(L0['hip']), bb(L0['kne']), h, kne, lw_t)
        sh = bone_xf(bb(L0['kne']), bb(L0['ank']), kne, ank, lw_s)
        phi_f = sh[2]
        if ground is not None:
            # a foot as low as a standing one stands flat on the floor line; lifting off it turns with the shin
            ah = J.get('ankle_h', 10.0)
            ank_s = np.asarray(L['ank'], float)
            up = ground - ank_s[1] - ah
            wgt = float(np.clip((up - 3) / max(6.0, ah), 0, 1))
            phi_f = sh[2] * wgt
            sole = (m.sole[tag] - L0['ank'][1]) * fw / R
            ank = ank + np.array([0.0, (ground - sole - ank_s[1]) * (1 - wgt)]) * R
            sh = bone_xf(bb(L0['kne']), bb(L0['ank']), kne, ank, lw_s)
        bones[f'{tag}_shin'] = sh
        bones[f'{tag}_foot'] = fixed_xf(bb(L0['ank']), ank, phi_f, fw)
    Vb = m.body.deform(bones, mir)

    def arm_bones(side, lay, A):
        L = J[side]
        mx = (lambda p: lay.mirror_x(p)) if mir else (lambda p: np.asarray(p, float))
        s_, e_, w_ = S(L['sho']), S(L['elb']), S(L['wri'])
        up = bone_xf(mx(A['sho']), mx(A['elb']), s_, e_, aw_u)
        fo = bone_xf(mx(A['elb']), mx(A['wri']), e_, w_, aw_f)
        return {'upper': up, 'fore': fo, 'hand': fixed_xf(mx(A['wri']), w_, fo[2], hand_s)}

    Vfa = m.arm_far.deform(arm_bones(far, m.arm_far, m.af), mir)
    Vna = m.arm.deform(arm_bones(near, m.arm, m.a), mir)
    far_shade = cfg.get('mesh_far_arm_shade', 0.74 if m.arm_far is m.arm else 0.88)
    over(cv, raster(m.arm_far, Vfa, np.ones(len(m.arm_far.tris), bool), ts, mir, shape, far_shade))
    for g in (0, 1, 2):
        over(cv, raster(m.body, Vb, m.body.tgroup == g, ts, mir, shape))
    over(cv, raster(m.arm, Vna, m.arm.tgroup == 0, ts, mir, shape))
    # head: rigid, tilted along shoulder -> detected head (damped as in paste_head), seated on the neck tip below
    # (CONFIG head_sink only applies to the cut-out renderer: here the head always sits on the neck stub)
    nk = S(J['head']) - sho; hs = G * cfg['head']
    hp = P['head']
    if 'neck' in bones:
        na0, na, nphi, nP = bones['neck']; neck = na + _rot(np.array(nphi)) @ nP @ (bb(m.neck0) - na0)
    else: neck = sp_a + _rot(np.array(sp_phi)) @ sp_P @ (bb(m.neck0) - sp_a0)
    ang = math.atan2(-nk[0], -nk[1]) if np.linalg.norm(nk) else 0.0
    ang *= float(np.clip((abs(ang) - 0.9) / 0.6, 0, 1))  # same damping as paste_head
    c_, s_ = math.cos(ang), math.sin(ang)
    Mh = np.array([[c_ * hs, s_ * hs], [-s_ * hs, c_ * hs]])
    # The head always sits ON the neck: its neck end (pivot b) goes exactly on the deformed neck stub. The detector's
    # head point drifts down into the chest in hunched poses (Alex's lunges), which buried the head; it now only
    # sets the tilt.
    head = neck - Mh @ (_head_seat(hp) - hp.a)
    over(cv, raster(m.arm, Vna, m.arm.tgroup == 1, ts, mir, shape))
    # head last: a guard's near forearm/fist must never cover the mascot's face
    paste_head(cv, hp, head, nk, hs)
    return drop_specks(cv, R)


def _head_seat(hp):
    """The head's point that goes on the neck tip: its neck pivot b, or, when b falls in empty space (a head drawn
    with a gap under the chin, e.g. PIPPIN's mane), the drawn pixel nearest b, one px into the art. Seating an
    empty pivot on the neck left the head floating a few px off the body."""
    key = (id(hp.im), hp.b.tobytes())
    if getattr(hp, '_seat_key', None) == key: return hp._seat
    al = np.asarray(hp.im)[:, :, 3] >= 110
    bx, by = (int(round(v)) for v in hp.b)
    seat = hp.b
    if not (0 <= by < al.shape[0] and 0 <= bx < al.shape[1] and al[by, bx]):
        ys, xs = np.nonzero(al)
        if len(ys):
            i = int(np.argmin((xs - hp.b[0]) ** 2 + (ys - hp.b[1]) ** 2))
            q = np.array([xs[i], ys[i]], float); d = hp.a - q
            seat = q + d / (np.linalg.norm(d) or 1.0)
    hp._seat_key, hp._seat = key, seat
    return seat


def drop_specks(cv, R):
    """Remove tiny loose fragments (a belt tail or a sliver of outline torn off where the mesh folds at a deep bend):
    pieces under ~20 sprite px and under 3% of the figure. The mascot is one connected body."""
    a = np.array(cv)
    al = (a[:, :, 3] >= 110).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(al, connectivity=8)
    if n <= 2: return cv
    area = st[1:, cv2.CC_STAT_AREA]
    small = np.zeros(n, bool)
    small[1:] = (area < 20 * R * R) & (area < 0.03 * area.max())
    if not small.any(): return cv
    kill = small[lab]
    # also the faint fringe around them
    kill = cv2.dilate(kill.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    kill &= ~((al > 0) & ~small[lab])
    a[kill] = 0
    return Image.fromarray(a, 'RGBA')


# ---------------------------------------------------------------------------------------------------- bind joints

def _runs(row, min_len=6):
    xs = np.nonzero(row)[0]
    if not len(xs): return []
    br = np.nonzero(np.diff(xs) > 1)[0]
    st = np.r_[xs[0], xs[br + 1]]; en = np.r_[xs[br], xs[-1]]
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a + 1 >= min_len]


def _arm_geometry(al):
    ys, xs = np.nonzero(al)
    y0, y1 = int(ys.min()), int(ys.max())
    L = y1 - y0
    cx = lambda y: float(np.mean(np.nonzero(al[int(y)])[0])) if al[int(y)].any() else float(xs.mean())
    width = lambda y: int(al[int(y)].sum())
    top_w = max(width(y) for y in range(y0, y0 + max(2, L // 6)))
    sho_y = y0 + 0.42 * top_w
    # wrist: the narrowest row above the fist
    rng = range(int(y0 + 0.62 * L), int(y1 - 0.1 * L))
    wri_y = min(rng, key=width) if len(rng) else y0 + 0.78 * L
    elb_y = sho_y + 0.52 * (wri_y - sho_y)
    end_y = y1 - 0.15 * (y1 - wri_y)
    pt = lambda y: [round(cx(y), 1), round(float(y), 1)]
    return dict(size=[al.shape[1], al.shape[0]], sho=pt(sho_y), elb=pt(elb_y), wri=pt(wri_y), end=pt(end_y))


def _body_geometry(al):
    """Legs below the crotch, centre lines, joints at standard fractions (the RTMPose read refines them)."""
    ys, xs = np.nonzero(al)
    top, bot = int(ys.min()), int(ys.max())
    # crotch: the first row (from the bottom up) where two leg runs merge into one
    crotch = None
    for y in range(bot - 5, top, -1):
        r = _runs(al[y], 12)
        if len(r) < 2 and y < bot - 0.25 * (bot - top):
            crotch = y + 1; break
    if crotch is None: crotch = int(top + 0.55 * (bot - top))
    legs = {'left': [], 'right': []}
    for y in range(crotch, bot + 1):
        r = _runs(al[y], 6)
        if len(r) >= 2:
            legs['left'].append((y, r[0])); legs['right'].append((y, r[-1]))
    out = {}
    width_at = lambda y: (max(a for a in range(0, 1)) if False else None)
    for k, rows in legs.items():
        if not rows: continue
        ysl = np.array([y for y, _ in rows]); cxl = np.array([(a + b) / 2 for _, (a, b) in rows])
        wl = np.array([b - a for _, (a, b) in rows])
        sole = int(ysl.max())
        # the foot: bottom rows where the run widens toward the toes; ankle above it
        leg_len = sole - crotch
        upper = wl[: max(3, len(wl) // 2)]
        shin_w = float(np.median(wl[len(wl) // 2: int(len(wl) * 0.8)])) if len(wl) > 10 else float(np.median(wl))
        foot_top = sole
        for y, wv in zip(ysl[::-1], wl[::-1]):
            if wv <= shin_w * 1.15: foot_top = y; break
        ank_y = foot_top - 0.15 * shin_w
        hip_y = crotch - 0.35 * float(np.median(upper))
        kne_y = hip_y + 0.5 * (ank_y - hip_y)
        cx = lambda y: float(np.interp(y, ysl, cxl))
        # toe: the far end of the foot in the facing direction (+x)
        c0 = cx(ank_y); fxmax = c0
        for y in range(int(foot_top), sole + 1):
            r = _runs(al[y], 3)
            if r:
                a, b = min(r, key=lambda t: 0 if t[0] <= c0 <= t[1] else min(abs(t[0] - c0), abs(t[1] - c0)))
                fxmax = max(fxmax, b)
        toe = [float(fxmax) - 0.15 * (fxmax - c0), float(sole) - 0.35 * (sole - foot_top)]
        hip_x = cx(crotch + 2)
        out[k] = dict(hip=[round(hip_x, 1), round(hip_y, 1)], kne=[round(cx(kne_y), 1), round(kne_y, 1)],
                      ank=[round(cx(ank_y), 1), round(ank_y, 1)], toe=[round(toe[0], 1), round(toe[1], 1)], sole=sole)
    # shoulders: the first row below the neck stub as wide as 80% of the chest
    # shoulders: the first row as wide as 85% of the upper chest (the top 30% of the neck-to-crotch height)
    widths = np.array([al[y].sum() for y in range(top, top + max(3, int(0.3 * (crotch - top))))])
    sho_y = top + int(np.argmax(widths >= 0.85 * widths.max()))
    row = np.nonzero(al[sho_y])[0]
    hip_c = np.mean([out[k]['hip'] for k in out], 0) if out else [float(xs.mean()), float(crotch)]
    sho = [round(float((row.min() + row.max()) / 2), 1), float(sho_y)]
    return dict(size=[al.shape[1], al.shape[0]], sho=sho, hip=[round(float(hip_c[0]), 1), round(float(hip_c[1]), 1)],
                legs=out, crotch=crotch)


def _rtmpose_read(body_rgba, arm_rgba, geo_b, geo_a, head_path):
    """RTMPose on a composite (body + both arms at the shoulders + the puppet head) -> COCO keypoints in body px."""
    sys.path.insert(0, str(MOD))
    import pose as posemod
    Hh, Wh = body_rgba.shape[:2]
    pad = 300
    can = Image.new('RGBA', (Wh + 2 * pad, Hh + 2 * pad), (255, 255, 255, 255))
    off = np.array([pad, pad])
    sx, sy = geo_b['sho']; row = np.nonzero(body_rgba[int(sy), :, 3])[0]
    arm = Image.fromarray(arm_rgba, 'RGBA')
    sc = 0.95 * (geo_b['hip'][1] - geo_b['sho'][1]) * 1.1 / max(1.0, geo_a['wri'][1] - geo_a['sho'][1])
    arm = arm.resize((int(arm.width * sc), int(arm.height * sc)), Image.LANCZOS)
    for x in (row.min() + 0.1 * (row.max() - row.min()), row.max() - 0.1 * (row.max() - row.min())):
        can.alpha_composite(arm, tuple(int(v) for v in (off + [x, sy] - np.array(geo_a['sho']) * sc)))
    can.alpha_composite(Image.fromarray(body_rgba, 'RGBA'), tuple(off))
    if head_path is not None and Path(head_path).exists():
        import skin
        hd = skin.key_part(head_path)
        hs = 1.25 * (row.max() - row.min()) / hd.width
        hd = hd.resize((int(hd.width * hs), int(hd.height * hs)), Image.LANCZOS)
        can.alpha_composite(hd, (int(off[0] + sx - hd.width / 2), int(off[1] + sy - hd.height * 0.92)))
    rgb = np.array(can.convert('RGB'))
    s = 640 / max(rgb.shape[:2])
    small = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    body = posemod.make_body()
    kp, scr = body(small[:, :, ::-1].copy())
    if not len(scr): return None, can
    i = int(np.argmax(scr.mean(1)))
    return (kp[i] / s - off, scr[i]), can


def bind(mascot, show=True, use_pose=True):
    d = MESH / mascot
    body = key_alpha(d / 'body.png'); arm = key_alpha(d / 'arm.png')
    gb = _body_geometry(body[:, :, 3] > 0); ga = _arm_geometry(arm[:, :, 3] > 0)
    gaf = _arm_geometry(key_alpha(d / 'arm_far.png')[:, :, 3] > 0) if (d / 'arm_far.png').exists() else None
    legs = gb['legs']
    notes = []
    if use_pose and 'left' in legs and 'right' in legs:
        res, can = _rtmpose_read(body, arm, gb, ga, MOD / 'assets' / 'puppet' / mascot / 'head.png')
        if res is not None:
            kp, sc = res
            # COCO hips 11/12, knees 13/14, ankles 15/16: label by x (image left / right leg)
            for name, (i, j) in dict(hip=(11, 12), kne=(13, 14), ank=(15, 16)).items():
                pts = [(kp[i], sc[i]), (kp[j], sc[j])]
                pts.sort(key=lambda t: t[0][0])
                for leg, (p, s) in zip(('left', 'right'), pts):
                    if s < 0.45: continue
                    g = legs[leg][name]
                    if name == 'hip':  # height from the read, x on the leg's centre line
                        y = float(np.clip(p[1], gb['crotch'] - 0.6 * (legs[leg]['kne'][1] - gb['crotch']), gb['crotch']))
                        legs[leg][name] = [g[0], round(y, 1)]
                    else:
                        lo = legs[leg]['hip'][1] + 20; hi = legs[leg]['sole'] - 10
                        y = float(np.clip(p[1], lo, hi))
                        al = body[int(y), :, 3] > 0
                        r = _runs(al, 6)
                        r = r[0] if leg == 'left' else r[-1]
                        legs[leg][name] = [round((r[0] + r[1]) / 2, 1), round(y, 1)]
                    notes.append(f'{leg}.{name} rtmpose {s:.2f}')
            if sc[5] > 0.45 and sc[6] > 0.45:
                y = float(np.clip((kp[5][1] + kp[6][1]) / 2, gb['sho'][1] - 40, gb['sho'][1] + 60))
                gb['sho'][1] = round(y, 1); notes.append('sho rtmpose')
            can.convert('RGB').resize((can.width // 3, can.height // 3)).save(d / '_rtmpose_in.png')
    for leg in legs.values():  # knees between hip and ankle
        if not (leg['hip'][1] < leg['kne'][1] < leg['ank'][1]):
            leg['kne'] = [leg['kne'][0], round((leg['hip'][1] + leg['ank'][1]) / 2, 1)]
    hipc = np.mean([legs[k]['hip'] for k in legs], 0)
    gb['hip'] = [round(float(hipc[0]), 1), round(float(hipc[1]), 1)]
    old = json.loads((d / 'bind.json').read_text()) if (d / 'bind.json').exists() else {}
    near = old.get('near_leg', 'left')
    out = dict(near_leg=near, sp=old.get('sp', 12), notes=notes, erase=old.get('erase', {}),
               body=dict(size=gb['size'], sho=gb['sho'], hip=gb['hip'],
                         near=legs[near], far=legs['right' if near == 'left' else 'left']),
               arm=ga, fix=old.get('fix', {}))
    if gaf is not None: out['arm_far'] = gaf
    # hand corrections: "fix": {"body.near.kne": [x, y], "arm.wri": [x, y], ...}
    for k, v in out['fix'].items():
        node = out
        ks = k.split('.')
        for kk in ks[:-1]: node = node[kk]
        node[ks[-1]] = v
    (d / 'bind.json').write_text(json.dumps(out, indent=1))
    if show: draw_bind(mascot)
    return out


def draw_bind(mascot):
    d = MESH / mascot
    bj = json.loads((d / 'bind.json').read_text())
    ims = []
    for name, key in (('body', 'body'), ('arm', 'arm'), ('arm_far', 'arm_far')):
        if key not in bj: continue
        im = Image.open(d / f'{name}.png').convert('RGB'); a = np.array(im)
        J = bj[key]
        def p(v): return tuple(int(round(x)) for x in v)
        col = {'near': (255, 210, 0), 'far': (0, 230, 255)}
        if key == 'body':
            cv2.line(a, p(J['sho']), p(J['hip']), (255, 255, 255), 5)
            for s in ('near', 'far'):
                L = J[s]
                for u, v in (('hip', 'kne'), ('kne', 'ank'), ('ank', 'toe')):
                    cv2.line(a, p(L[u]), p(L[v]), col[s], 5)
                for u in ('hip', 'kne', 'ank', 'toe'): cv2.circle(a, p(L[u]), 12, col[s], 3)
                cv2.line(a, (int(L['toe'][0]) - 60, int(L['sole'])), (int(L['toe'][0]) + 30, int(L['sole'])), (255, 0, 0), 3)
            for u in ('sho', 'hip'): cv2.circle(a, p(J[u]), 14, (255, 255, 255), 3)
        else:
            for u, v in (('sho', 'elb'), ('elb', 'wri'), ('wri', 'end')):
                cv2.line(a, p(J[u]), p(J[v]), (255, 210, 0), 5)
            for u in ('sho', 'elb', 'wri', 'end'): cv2.circle(a, p(J[u]), 12, (255, 210, 0), 3)
        ims.append(Image.fromarray(a))
    h = 768
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    sh = Image.new('RGB', (sum(i.width for i in ims), h))
    x = 0
    for i in ims: sh.paste(i, (x, 0)); x += i.width
    sh.save(d / '_bind.png')
    print(d / '_bind.png')


def draw_weights(mascot):
    """Debug: each vertex coloured by its bone mix, mesh edges, over the drawing."""
    m = Model(mascot)
    pal = np.array([(255, 255, 255), (255, 60, 60), (255, 160, 0), (255, 255, 0), (60, 120, 255), (0, 220, 255),
                    (160, 255, 160)], float)
    ims = []
    for L in (m.body, m.arm):
        a = (L.rgba[:, :, :3] * 0.35).astype(np.uint8).copy()
        col = (pal[L.idx[:, :]] * L.w[..., None]).sum(1)
        for t in L.tris:
            pts = L.V[t].astype(np.int32)
            cv2.polylines(a, [pts], True, (70, 70, 70), 1)
        for v, c in zip(L.V.astype(int), col):
            cv2.circle(a, tuple(v), 4, tuple(int(x) for x in c), -1)
        ims.append(Image.fromarray(a))
    h = 900
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    sh = Image.new('RGB', (sum(i.width for i in ims), h)); x = 0
    for i in ims: sh.paste(i, (x, 0)); x += i.width
    sh.save(MESH / mascot / '_weights.png'); print(MESH / mascot / '_weights.png')


def bind_sheet(ms):
    """All mascots' _bind.png in one contact sheet: assets/mesh/_binds.png."""
    ims = [(m, Image.open(MESH / m / '_bind.png')) for m in ms if (MESH / m / '_bind.png').exists()]
    if not ims: return
    h = 384; cols = 4
    ims = [(m, i.resize((int(i.width * h / i.height), h))) for m, i in ims]
    w = max(i.width for _, i in ims)
    sh = Image.new('RGB', (w * cols, (h + 20) * ((len(ims) + cols - 1) // cols)))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sh)
    for k, (m, i) in enumerate(ims):
        x, y = (k % cols) * w, (k // cols) * (h + 20)
        sh.paste(i, (x, y + 20)); d.text((x + 4, y + 4), m, fill='yellow')
    sh.save(MESH / '_binds.png'); print(MESH / '_binds.png')


if __name__ == '__main__':
    cmd, ms = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith('--')]
    if ms == ['all']: ms = sorted(p.name for p in MESH.iterdir() if p.is_dir() and not p.name.startswith('_'))
    if cmd == 'sheet': bind_sheet(ms); ms = []
    for mm in ms:
        if cmd == 'bind': bind(mm, use_pose='--nopose' not in sys.argv)
        elif cmd == 'show': draw_bind(mm)
        elif cmd == 'weights': draw_weights(mm)
