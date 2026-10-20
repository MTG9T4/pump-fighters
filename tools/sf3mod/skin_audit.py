"""Audit: does any pattern of a fighter still show the original SF3 body?

    python sf3mod/skin_audit.py pl18            # one bank
    python sf3mod/skin_audit.py all             # all 19 skinned banks
    python sf3mod/skin_audit.py all --before    # what the pipeline did without the completion pass
    python sf3mod/skin_audit.py pl06 --sheets   # also contact sheets of the effect / other / completed patterns

Runs the same pose pipeline as the build (core + the fighter's hook + completion) and classifies every pattern:
posed (mascot from the detector / core / hook), body (mascot with a borrowed pose), piece (mascot head / arm over a
loose body piece), effect (no body pixels, keeps its drawing), other (another character, keeps its drawing),
none (empty / tiny). Anything classified body or piece that ends up without a mascot is STILL ORIGINAL BODY.
Also renders every skinned frame and counts the puppet's connected pieces (specks < 6 px ignored): more than one
is a SEPARATED frame (hips off the legs, an arm off the torso), listed in audit_separated.png.
Writes build/skin/<bank>/review/audit_original.png (every pattern still showing the original body) and, with
--sheets, audit_effect.png, audit_other.png and audit_completed.png (original | mascot pairs). Exits 1 when any
original body or separated frame remains.
"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD)); sys.path.insert(0, str(MOD / 'tools'))
import skin  # noqa


def sheet(items, path, W=1800, title=''):
    """items: [(label, [PIL images])] -> one contact sheet."""
    cells = []
    for label, ims in items:
        w = sum(i.width for i in ims) + 2 * (len(ims) - 1); h = max(i.height for i in ims)
        c = Image.new('RGBA', (max(w, 34) + 4, h + 14), (60, 60, 70, 255)); x = 2
        for i in ims: c.alpha_composite(i, (x, 12)); x += i.width + 2
        ImageDraw.Draw(c).text((2, 0), label, fill='yellow'); cells.append(c)
    if not cells:
        if path.exists(): path.unlink()
        return
    x = y = rowh = 0; pos = []
    for c in cells:
        if x + c.width > W: x = 0; y += rowh + 2; rowh = 0
        pos.append((x, y + 16)); x += c.width + 2; rowh = max(rowh, c.height)
    out = Image.new('RGBA', (W, y + rowh + 16), (20, 20, 24, 255))
    ImageDraw.Draw(out).text((4, 2), title, fill='white')
    for c, p in zip(cells, pos): out.alpha_composite(c, p)
    path.parent.mkdir(parents=True, exist_ok=True); out.save(path)


def frame(bank, p):
    f = MOD / 'build' / 'skin' / bank / 'frames' / f'{p:04d}.png'
    return Image.open(f).convert('RGBA') if f.exists() else Image.new('RGBA', (8, 8))


def audit(bank_name, before=False, sheets=False, check_sep=True):
    bank, pose, parts, G, cfg, hooks = skin.setup(bank_name)
    run = dict(cfg, complete=not before)
    out = skin.prepare_poses(bank_name, bank, pose, G, run, hooks, None)
    if before:  # classify what the completion pass would have had to fix, without changing anything
        core = skin._last_core[bank_name]
        classes = skin.classify_and_complete(bank_name, bank, pose, G, cfg, hooks, dict(out), core)
        still = [p for p, c in classes.items() if c in ('body', 'piece')]
    else:
        classes = skin._audit.get(bank_name, {})
        still = [p for p, c in classes.items() if c in ('body', 'piece') and p not in out]
    cnt = {k: sum(1 for c in classes.values() if c == k) for k in ('posed', 'body', 'piece', 'effect', 'other', 'none')}
    print(f"{bank_name} ({skin.SKINS[bank_name]['mascot']}): {bank.count} patterns | mascot {cnt['posed']} posed + "
          f"{0 if before else cnt['body']} borrowed + {0 if before else cnt['piece']} pieces | effect {cnt['effect']} | "
          f"other character {cnt['other']} | empty {cnt['none']} | STILL ORIGINAL BODY {len(still)}", flush=True)
    # body connectivity: every skinned frame's puppet must be one piece (specks under 6 px ignored)
    separated = []
    if check_sep and not before:
        import cv2
        R0 = skin.R
        try:
            skin.R = 1  # compose straight at sprite size (16x faster than the 4x render + shrink; same shapes)
            for p, J in sorted(out.items()):
                W, H = pose[str(p)]['size']
                rd = skin.redraw_frames(bank_name, pose, [p]) if J.get('_keep') is None else {}
                al = (rd[p][:, :, 3] > 0) if p in rd else np.asarray(skin.compose(J, parts, G / R0, cfg, (W, H)))[:, :, 3] >= 110
                n, lab, st, _ = cv2.connectedComponentsWithStats(al.astype(np.uint8), connectivity=8)
                if sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 6) > 1: separated.append(p)
        finally:
            skin.R = R0
        print(f'{bank_name}: SEPARATED {len(separated)} of {len(out)} skinned frames' +
              (f' e.g. {separated[:20]}' if separated else ''), flush=True)
    rev = MOD / 'build' / 'skin' / bank_name / 'review'
    if check_sep and not before:
        items = []
        for p in separated[:300]:
            W, H = pose[str(p)]['size']
            sk = Image.fromarray(skin.render(out[p], parts, G, cfg, (W, H)), 'RGBA')
            bg = Image.new('RGBA', sk.size, (40, 40, 48, 255)); bg.alpha_composite(sk)
            items.append((str(p), [frame(bank_name, p), bg]))
        sheet(items, rev / 'audit_separated.png', title=f'{bank_name}: separated puppets ({len(separated)}; original | mascot)')
    sheet([(str(p), [frame(bank_name, p)]) for p in sorted(still)], rev / 'audit_original.png',
          title=f'{bank_name}: patterns still showing the original body ({len(still)})')
    if sheets:
        for cls in ('effect', 'other'):
            ps = sorted(p for p, c in classes.items() if c == cls)
            sheet([(str(p), [frame(bank_name, p)]) for p in ps], rev / f'audit_{cls}.png', title=f'{bank_name}: {cls} ({len(ps)})')
        if not before:
            items = []
            for p in sorted(p for p, c in classes.items() if c in ('body', 'piece')):
                o = frame(bank_name, p); W, H = pose[str(p)]['size']
                sk = Image.fromarray(skin.render(out[p], parts, G, cfg, (W, H)), 'RGBA').crop((skin.M, skin.M, skin.M + W, skin.M + H))
                bg = Image.new('RGBA', sk.size, (40, 40, 48, 255)); bg.alpha_composite(sk)
                items.append((f"{p} {classes[p][0]}{out[p].get('_borrowed', '')}", [o, bg]))
            sheet(items, rev / 'audit_completed.png', title=f'{bank_name}: completed (original | mascot)')
    return len(still), len(separated), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank'); ap.add_argument('--before', action='store_true'); ap.add_argument('--sheets', action='store_true')
    ap.add_argument('--no-separated', action='store_true', help='skip the connectivity check (faster)')
    a = ap.parse_args()
    banks = list(skin.SKINS) if a.bank == 'all' else a.bank.split(',')
    total = sep = 0
    for b in banks:
        n, k, _ = audit(b, a.before, a.sheets, not a.no_separated); total += n; sep += k
    print(f'TOTAL still showing the original body: {total}; separated puppets: {sep}')
    sys.exit(1 if total or sep else 0)


if __name__ == '__main__':
    main()
