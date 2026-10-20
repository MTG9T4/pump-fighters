"""Compare encoded cell bytes for skin variants on a sample of one fighter's frames.
    python sf3mod/scripts/size_test.py pl18 [step]
"""
import json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from multiprocessing import Pool

MOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOD)); sys.path.insert(0, str(MOD / 'tools'))
import skin


def enc(cell):
    from codecs_sf3 import p6_encode_greedy
    from spritebank import morton
    raw = np.empty(cell.size, np.uint8); raw[morton(32).reshape(-1)] = cell.reshape(-1)
    return len(p6_encode_greedy(raw.tobytes())) + 1


def flatten(part):
    """Edge-preserving flatten: mean-shift the RGB, keep alpha."""
    a = np.asarray(part.im).copy()
    rgb = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(a[:, :, :3]), 12, 28)
    a[:, :, :3] = rgb
    p = skin.Part.__new__(skin.Part); p.__dict__.update(part.__dict__); p.im = Image.fromarray(a, 'RGBA')
    return p


def mode3(idx):
    """Replace isolated pixels with their most common 3x3 neighbour (outline slots untouched)."""
    out = idx.copy()
    pad = np.pad(idx, 1)
    H, W = idx.shape
    stack = np.stack([pad[y:y + H, x:x + W] for y in range(3) for x in range(3)])
    same = (stack == idx[None]).sum(0)
    lone = (same <= 2) & (idx > 0) & ~np.isin(idx, [1, 16, 32, 48])
    if lone.any():
        ys, xs = np.nonzero(lone)
        for y, x in zip(ys, xs):
            v = stack[:, y, x]; v = v[v > 0]
            if len(v): out[y, x] = np.bincount(v).argmax()
    return out


def run(bank, step, flat, mode):
    cfg = skin.SKINS[bank]
    pose = json.loads((MOD / 'build' / 'skin' / bank / 'pose.json').read_text())
    parts = skin.load_puppet(cfg['mascot'])
    if flat: parts = {k: flatten(v) for k, v in parts.items()}
    G = skin.global_scale(pose, cfg, parts)
    frames = [p for p in sorted(pose, key=int)[::step] if skin.pose_frame(pose[p]) is not None]
    imgs = []
    for p in frames:
        J = skin.fill_limbs(skin.pose_frame(pose[p]), G); W, H = pose[p]['size']
        imgs.append(skin.shrink(skin.compose(J, parts, G, cfg, (W, H)), (W + 2 * skin.M, H + 2 * skin.M)))
    pal = skin.make_palette(imgs[::3])
    cells = []
    for a in imgs:
        idx = skin.snap(a, pal)
        if mode: idx = mode3(idx)
        cells += [c for _, _, c in skin.grid_tiles(idx, 0, 0)]
    with Pool() as pool:
        total = sum(pool.map(enc, cells, chunksize=32))
    if flat or mode:
        a = imgs[len(imgs) // 2]; idx = skin.snap(a, pal); idx = mode3(idx) if mode else idx
        Image.fromarray(pal[idx], 'RGBA').resize((idx.shape[1] * 3, idx.shape[0] * 3), Image.NEAREST).save(
            MOD / 'build' / 'skin' / f'size_{bank}_{int(flat)}{int(mode)}.png')
    return len(frames), len(cells), total


if __name__ == '__main__':
    bank = sys.argv[1]; step = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for flat, mode in ((0, 0), (1, 0), (0, 1), (1, 1)):
        n, c, t = run(bank, step, flat, mode)
        print(f'flat={flat} mode={mode}: {n} frames, {c} cells, {t:,} bytes ({t / c:.0f} B/cell)', flush=True)
