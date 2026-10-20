"""Shared, lossless colour and output helpers (Python 3.12, Pillow, numpy)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'game' / 'SF33RD.AFS'


def output_path(path, *inputs):
    p = Path(path).resolve()
    protected = [ORIGINAL, *(Path(i) for i in inputs)]
    if any(p == i.resolve() or (p.exists() and i.exists() and p.samefile(i)) for i in protected):
        raise ValueError(f'refusing to overwrite original/input: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def decode_color(raw, fmt=0x1555, little=True):
    fields = {0x1555: ((10, 31), (5, 31), (0, 31), (15, 1)),
              0x565: ((11, 31), (5, 63), (0, 31), (0, 0)),
              0x4444: ((8, 15), (4, 15), (0, 15), (12, 15)),
              0x8888: ((16, 255), (8, 255), (0, 255), (24, 255))}
    if fmt not in fields:
        raise ValueError(f'unsupported colour format 0x{fmt:04x}')
    a = np.frombuffer(raw, dtype=('<' if little else '>') + ('u4' if fmt == 0x8888 else 'u2')).astype(np.uint32)
    return np.stack([((a >> s) & m) * 255 // m if m else np.full(a.shape, 255)
                     for s, m in fields[fmt]], axis=-1).astype(np.uint8)


def encode_color(rgba, fmt=0x1555, little=True):
    fields = {0x1555: ((10, 31), (5, 31), (0, 31), (15, 1)),
              0x565: ((11, 31), (5, 63), (0, 31), (0, 0)),
              0x4444: ((8, 15), (4, 15), (0, 15), (12, 15)),
              0x8888: ((16, 255), (8, 255), (0, 255), (24, 255))}
    if fmt not in fields:
        raise ValueError(f'unsupported colour format 0x{fmt:04x}')
    a = np.asarray(rgba, dtype=np.uint32).reshape(-1, 4)
    v = np.zeros(len(a), dtype=np.uint32)
    for ch, (s, m) in enumerate(fields[fmt]):
        if m:
            v |= ((a[:, ch] * m + 127) // 255) << s
    return v.astype(('<' if little else '>') + ('u4' if fmt == 0x8888 else 'u2')).tobytes()


def palette_indices(rgba, palette, exact=False):
    """RGBA distance including alpha; bounded memory, deterministic duplicate handling."""
    pixels = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
    unique, inverse = np.unique(pixels, axis=0, return_inverse=True)
    pal = np.asarray(palette, dtype=np.int32)
    result = np.empty(len(unique), dtype=np.uint8)
    for start in range(0, len(unique), 1024):
        u = unique[start:start + 1024].astype(np.int32)
        dist = ((u[:, None] - pal[None]) ** 2).sum(axis=2)
        best = dist.argmin(axis=1)
        if exact and np.any(dist[np.arange(len(u)), best] != 0):
            raise ValueError('PNG contains colours outside the tile palette; quantize it first')
        result[start:start + len(u)] = best
    return result[inverse].reshape(np.asarray(rgba).shape[:-1])
