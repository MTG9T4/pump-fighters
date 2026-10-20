"""Mascot recolours for the 10 SF3 bodies (plNNpl.bin palettes).

Each palette row is split into its colour ramps; every ramp is classified by its average colour
(skin / white / dark / red / blue / yellow / green / purple / brown), and the mascot's rule for that
class re-hues the ramp while keeping each colour's brightness, so shading survives.
Applied to the 13 selectable rows; index 0 (transparent) and 1 (outline) are left alone.
"""
import colorsys, json, sys
from pathlib import Path
import numpy as np

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD / 'tools'))

# target: (hue degrees or None to keep, saturation or None to keep, value scale, contrast)
RULES = {
    'pl02': {  # Ryu -> SWORDCAT: orange fur, polished steel gi, blue cape accents, gold gloves
        'skin': (26, 0.88, 1.1, 1.0), 'brown': (26, 0.88, 1.15, 1.0), 'white': (212, 0.14, 1.0, 1.25),
        'red': (222, 0.78, 1.0, 1.0), 'blue': (45, 0.85, 1.1, 1.0)},
    'pl11': {  # Ken -> INU: shiba tan fur, cream highlights, red gi stays
        'skin': (31, 0.62, 1.08, 1.0), 'yellow': (33, 0.55, 1.0, 1.0)},
    'pl04': {  # Dudley -> JEAN PHIL: pale skin, grey houndstooth suit, red tie stays
        'skin': (22, 0.32, 1.35, 1.0), 'brown': (22, 0.32, 1.35, 1.0), 'white': (0, 0.0, 0.85, 1.35),
        'blue': (220, 0.06, 0.6, 1.3), 'green': (220, 0.06, 0.55, 1.3),
        9: '#d8b048'},  # index 9 is pure black hair/brows/mustache: set it blond outright
    'pl13': {  # Urien -> ELON (parody): pale skin, black outfit
        'skin': (22, 0.30, 1.30, 1.0), 'brown': (22, 0.30, 1.30, 1.0), 'white': (230, 0.08, 0.28, 1.2),
        'purple': (230, 0.10, 0.35, 1.2)},
    'pl05': {  # Necro -> PEPE: his lavender skin -> frog green, magenta outfit -> blue shirt, orange -> brown
        'blue': (108, 0.66, 0.9, 1.1), 'white': (108, 0.55, 0.95, 1.05), 'purple': (222, 0.72, 0.9, 1.0),
        'brown': (28, 0.55, 0.7, 1.0)},
    'pl16': {  # Chun-Li -> CLAUDIA: chrome silver outfit, black accents
        'white': (215, 0.07, 1.0, 1.45), 'blue': (230, 0.10, 0.30, 1.2), 'yellow': (30, 0.60, 0.95, 1.0)},
    'pl03': {  # Yun -> WIF: pink beanie, grey hoodie, shiba tan skin
        'blue': (330, 0.62, 1.15, 1.0), 'white': (0, 0.0, 0.78, 1.15), 'skin': (30, 0.58, 1.1, 1.0),
        'yellow': (0, 0.0, 0.7, 1.1)},
    'pl17': {  # Makoto -> POPCAT: white fur, yellow sash/shorts
        'skin': (30, 0.08, 1.3, 1.0), 'red': (50, 0.85, 1.1, 1.0), 'brown': (30, 0.10, 1.3, 1.0)},
    'pl18': {  # Q -> PNUT: squirrel-brown coat and hat, cream accents
        'skin': (24, 0.68, 0.72, 1.0), 'white': (24, 0.66, 0.72, 1.05), 'brown': (24, 0.68, 0.72, 1.0),
        'blue': (24, 0.62, 0.55, 1.0), 'purple': (24, 0.62, 0.55, 1.0)},
    'pl06': {  # Hugo -> MOO DENG: grey-pink hippo skin, blue sumo belt
        'skin': (350, 0.16, 0.9, 1.0), 'red': (225, 0.75, 0.95, 1.0), 'purple': (225, 0.75, 0.95, 1.0),
        'yellow': (350, 0.18, 0.85, 1.0), 'dark': (0, 0.0, 1.6, 1.0)},
}


def to_hsv(c):
    return colorsys.rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255)


def ramps(cols):
    out = []; start = 2
    for i in range(3, 64):
        h0, s0, v0 = to_hsv(cols[i - 1]); h1, s1, v1 = to_hsv(cols[i])
        dh = min(abs(h1 - h0), 1 - abs(h1 - h0))
        if v1 > v0 + 0.06 or (dh > 0.08 and s0 > 0.15 and s1 > 0.15) or abs(s1 - s0) > 0.35:
            out.append((start, i - 1)); start = i
    out.append((start, 63))
    return out


def classify(cols):
    hs = [to_hsv(c) for c in cols]
    h = np.angle(np.mean([np.exp(2j * np.pi * x[0]) for x in hs])) / (2 * np.pi) % 1 * 360
    s = float(np.mean([x[1] for x in hs])); v = float(np.max([x[2] for x in hs]))
    if v < 0.22: return 'dark'
    if s < 0.16: return 'white'
    if 12 <= h < 45 and s < 0.75 and v > 0.45: return 'skin'
    if 12 <= h < 45: return 'brown'
    if h < 12 or h >= 335: return 'red'
    if 45 <= h < 70: return 'yellow'
    if 70 <= h < 170: return 'green'
    if 170 <= h < 265: return 'blue'
    return 'purple'


def recolor_row(row, rules):
    out = [c[:] for c in row]
    for i, hx in rules.items():  # int keys: fixed colour for one palette index
        if isinstance(i, int):
            out[i] = [int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16), row[i][3]]
    for a, b in ramps(row):
        cols = row[a:b + 1]
        if all(c[3] == 0 for c in cols): continue
        cls = classify(cols)
        if cls not in rules or any(isinstance(i, int) and a <= i <= b for i in rules): continue
        th, ts, vs, contrast = rules[cls]
        vmean = np.mean([to_hsv(c)[2] for c in cols])
        for i in range(a, b + 1):
            h, s, v = to_hsv(row[i])
            v2 = min(1.0, max(0.0, (vmean + (v - vmean) * contrast) * vs))
            h2 = h if th is None else th / 360
            s2 = s if ts is None else min(1.0, ts * (0.6 + 0.4 * (s / max(0.01, max(to_hsv(c)[1] for c in cols)))) if ts > 0.05 else ts)
            r, g, bb = colorsys.hsv_to_rgb(h2, s2, v2)
            out[i] = [int(r * 255), int(g * 255), int(bb * 255), row[i][3]]
    return out


def hexrow(colors):
    return ['#%02x%02x%02x%02x' % tuple(c) for c in colors]


def build(ext, out):
    """Write recoloured plNNpl.bin files into the overrides dir."""
    for key, rules in RULES.items():
        src = json.loads((MOD / 'build' / 'palettes' / f'{key}pl.json').read_text())
        for row in src['rows']:
            if not row.get('selectable'): continue
            cols = [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), int(h[7:9], 16)] for h in row['colors']]
            row['colors'] = hexrow(recolor_row(cols, rules))
        j = MOD / 'build' / f'{key}pl_mod.json'
        j.write_text(json.dumps(src))
        import subprocess
        subprocess.run([sys.executable, str(MOD / 'tools' / 'pal.py'), 'import', str(j), str(ext / f'{key}pl.bin'),
                        str(out / f'{key}pl.bin')], check=True)
        subprocess.run([sys.executable, str(MOD / 'tools' / 'pal.py'), 'preview', str(ext / f'{key}.bin'),
                        str(out / f'{key}pl.bin'), '0', str(MOD / 'build' / f'{key}_mod_preview.png')], check=True)
        print('recolored', key)
