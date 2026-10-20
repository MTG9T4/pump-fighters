"""MOO DENG on Hugo (pl06).

Hugo's bank also holds frames that are not Hugo's body: Poison (his manager, drawn in blue: intro, taunts, win
poses) and loose arm pieces for his grabs (Meat Squasher, Gigas Breaker, Moonsault Press). Posing a whole hippo
on those put a second MOO DENG on screen, so they keep their own drawing (colour-mapped). Hugo's own frames track
well with the shared core; the puppet stays at the shared scale (he is the biggest fighter; never larger).
"""
import numpy as np
from pathlib import Path
from PIL import Image

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('skins_groupD_lib', str(Path(__file__).resolve().parent / 'pl18.py'))
D = _ilu.module_from_spec(_spec); _spec.loader.exec_module(D)  # group D shared helpers (connect, IK, blobs)


MOD = Path(__file__).resolve().parent.parent

# keep the shared size (scale 1.0): MOO DENG already fills Hugo's silhouette and memory is tight
CONFIG = dict(scale=1.0)

_kind = {}


def kind(p):
    """'poison' (blue figure), 'part' (loose arm / no trousers in a small frame) or 'body'."""
    if p in _kind: return _kind[p]
    f = MOD / 'build' / 'skin' / 'pl06' / 'frames' / f'{p:04d}.png'
    k = 'body'
    if f.exists():
        a = np.asarray(Image.open(f).convert('RGBA')).astype(int)
        m = a[..., 3] > 0
        if m.sum() >= 20:
            px = a[m]
            blue = (px[:, 2] > px[:, 0] + 25).mean()
            pink = ((px[:, 0] > 150) & (px[:, 2] > 90) & (px[:, 1] < 120)).mean()  # Hugo's trousers
            if blue > 0.35: k = 'poison'
            elif pink < 0.02 and a.shape[0] < 80: k = 'part'
    _kind[p] = k
    return k


def fix_pose(p, J, rec, ctx):
    if kind(p) != 'body': return None
    return D.connect(sane_ground(p, J, 'pl06'))


def sane_ground(p, J, bank_name):
    """Drop the core's floor line when it sits well above the bottom of the original drawing: frames with an
    unusual anchor (e.g. pl18 1004, origin (64, 10)) would otherwise have their feet yanked up onto it."""
    g = J.get('ground')
    if g is None: return J
    f = Path(__file__).resolve().parent.parent / 'build' / 'skin' / bank_name / 'frames' / f'{p:04d}.png'
    if not f.exists(): return J
    a = np.asarray(Image.open(f).convert('RGBA'))[..., 3] > 0
    ys = np.nonzero(a.any(1))[0]
    if len(ys) and g < ys.max() - 6: J['ground'] = None
    return J
