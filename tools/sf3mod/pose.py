"""Read a 2D skeleton out of every sprite frame of one fighter bank.

Renders each pattern, pads + upscales it, runs RTMPose (rtmlib, COCO-17) and stores the keypoints in sprite
pixel coordinates (relative to the pattern's render canvas) in build/skin/<bank>/pose.json.
Usage: python sf3mod/pose.py pl02 [pl11 ...]   (GPU via onnxruntime-gpu; resumes from a partial pose.json)
"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD / 'tools'))
from spritebank import SpriteBank, load_palettes

PAD, UP = 24, 4


def unrot(pts, k, shape):
    """Map points found in np.rot90(img, k) back to img coordinates."""
    H, W = shape[:2]; x, y = pts[:, 0], pts[:, 1]
    if k == 1: return np.stack([W - 1 - y, x], 1)
    if k == 2: return np.stack([W - 1 - x, H - 1 - y], 1)
    if k == 3: return np.stack([y, H - 1 - x], 1)
    return pts.copy()


def make_body():
    try:
        import torch  # its lib dir carries the CUDA/cuDNN DLLs onnxruntime-gpu needs
        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))
    except Exception:
        pass
    import onnxruntime
    from rtmlib import Body
    dev = 'cuda' if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else 'cpu'
    print('pose device', dev, flush=True)
    return Body(mode='performance', backend='onnxruntime', device=dev)


def main(bank_name, body=None):
    out = MOD / 'build' / 'skin' / bank_name
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    ext = MOD / 'build' / 'extracted'
    bank = SpriteBank.open(ext / f'{bank_name}.bin')
    pal = load_palettes(ext / f'{bank_name}.bin', ext / f'{bank_name}pl.bin')
    body = body or make_body()
    part = out / 'pose.partial.json'
    res = json.loads(part.read_text()) if part.exists() else {}
    for p in range(bank.count):
        if str(p) in res: continue
        try:
            im, meta = bank.render(p, pal, metadata=True)
        except ValueError as e:  # effect pieces on palettes the character loader doesn't map: not a body frame
            res[str(p)] = dict(origin=[0, 0], size=[1, 1], skip=str(e)); continue
        im.save(out / 'frames' / f'{p:04d}.png')
        rec = dict(origin=meta['origin'], size=list(im.size))
        if im.width >= 24 and im.height >= 40:
            bg = Image.new('RGBA', (im.width + 2 * PAD, im.height + 2 * PAD), (255, 255, 255, 255))
            bg.alpha_composite(im, (PAD, PAD))
            a = np.array(bg.convert('RGB').resize((bg.width * UP, bg.height * UP), Image.NEAREST))[:, :, ::-1].copy()
            best = None
            # upright first; flips, knockdowns and rolls read better from a rotated copy
            for k in (0, 1, 2, 3):
                if best is not None and best[0] >= 0.55: break
                r = np.ascontiguousarray(np.rot90(a, k))
                kp, sc = body(r)
                if not len(sc): continue
                i = int(np.argmax(sc.mean(1))); m = float(sc[i].mean())
                if best is None or m > best[0] + 0.03:
                    best = (m, unrot(kp[i], k, a.shape), sc[i], k)
            if best:
                rec['kp'] = (best[1] / UP - PAD).round(2).tolist()
                rec['score'] = best[2].round(3).tolist(); rec['rot'] = best[3]
        res[str(p)] = rec
        if p % 50 == 0:
            print(bank_name, p, '/', bank.count, flush=True)
            part.write_text(json.dumps(res))
    (out / 'pose.json').write_text(json.dumps(res))
    part.unlink(missing_ok=True)
    print('pose ->', out / 'pose.json')


if __name__ == '__main__':
    b = make_body()
    for name in sys.argv[1:]:
        main(name, b)
