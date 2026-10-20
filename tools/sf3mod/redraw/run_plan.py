"""Redraw the sheets in build/redraw/<bank>/plan.json with Codex, a few at a time, then cut each into frames.

  python sf3mod/redraw/run_plan.py <bank> [<bank>...] [--jobs 4] [--only idle,s003] [--skip idle] [--redo]

Fighters, mascots and reference art come from sf3mod/redraw/cast.json. The 'idle' sheet is drawn first from the
concept art and portrait and becomes the style reference (the character in sprite form) for that fighter's other
sheets. Resumable: a sheet whose codex.png exists is not sent again (unless --redo); every new sheet is posted.
Logs go to build/redraw/<bank>/sheets/<name>/codex.log.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'sf3mod' / 'build' / 'redraw'
SHEETS = ROOT / 'sf3mod' / 'redraw' / 'sheets.py'
CAST = json.loads((ROOT / 'sf3mod' / 'redraw' / 'cast.json').read_text())
STYLE_EXAMPLE = 'sf3mod/build/redraw/pl01/idle_codex.png'  # GIGA: the finished sprite look

HEAD = """Use your image generation tool. Do not write code (only copy the file). This is a fan mod of Street Fighter III:
3rd Strike where each fighter is redrawn as a pump.fun mascot. {NAME}: {desc}
"""
RULES = """Redraw the SAME sheet with {NAME} replacing {fighter} in every cell: identical grid, identical cell positions, and in
each cell the character in EXACTLY the same pose as that cell's frame: same position inside the cell, same size, same
facing direction, same stance, same knee bends, same arm and hand positions, same torso lean, same head position and
tilt. If the fighter is lying down, flipping, upside down, curled up or seen from the back, draw {NAME} in exactly that
orientation. Replace {fighter}'s face, hair, body and clothes completely with {NAME}'s design ({fighter} must not be
recognisable anywhere); an animal mascot fights upright in the same human pose. Every body part stays connected and
anatomically correct: head on a neck, arms attached at the shoulders, legs at the hips, hands at the ends of the arms.
{note}Draw only the fighter: no hair flames, no hit sparks, no effects. Keep the 3rd Strike hand-drawn sprite look: bold
dark outline, cel shading with a few tones, crisp edges, no blur, no background detail, no text. Flat grey background
(#808080) in every cell, keep the thin grid lines, keep empty cells empty. {NAME} must look identical from cell to
cell so the frames animate smoothly.
Save as {out} (original bytes, same aspect ratio as the source sheet). Finish with the saved path.
"""
IDLE = HEAD + """
Attached: (1) {src}: a 4x3 grid sheet of animation frames of the fighter {fighter} ({fighter_desc}), his idle stance
loop. Each frame sits in its own cell on flat grey (#808080), separated by thin darker grid lines. (2) {concept} and
(3) {portrait}: {NAME}'s design references; match the character exactly (face, colours, outfit, proportions of the
head and features). (4) {style_example}: a DIFFERENT character already finished in the target sprite style; copy only
its art style (outline weight, cel shading, level of detail, sprite size), not its character.

""" + RULES
SHEET = HEAD + """
Attached: (1) {src}: a 4x3 grid sheet of animation frames of the fighter {fighter} ({fighter_desc}) from his move set
(stances, attacks, being hit, knockdowns, throws, jumps). Each frame sits in its own cell on flat grey (#808080),
separated by thin darker grid lines. (2) {style}: {NAME} already redrawn in this exact sprite style; match this
character exactly (face, colours, outfit, outline weight and shading). (3) {concept} and (4) {portrait}: {NAME}'s
design references.

""" + RULES
PIECES = HEAD + """
Attached: (1) {src}: a 4x3 grid sheet. Each cell shows ONLY one body part of the fighter {fighter} ({fighter_desc}),
for example an arm with its hand or a leg, as a separate sprite piece; the rest of the body is drawn by another sprite,
so the part is cut off on purpose. (2) {style}: {NAME} already redrawn in this exact sprite style. (3) {concept}:
{NAME}'s design reference.

Redraw the SAME sheet: in each cell replace {fighter}'s body part with the same body part of {NAME}, nothing else: same
position in the cell, same size, same angle, same bend, same hand shape and direction, cut off exactly where the
original is cut. Use {NAME}'s skin, fur or sleeves for that part. Do not add a body, head or anything else. Keep the
3rd Strike hand-drawn sprite look: bold dark outline, cel shading with a few tones, crisp edges, no blur, no text.
Flat grey background (#808080) in every cell, keep the thin grid lines, keep the empty cells empty.
Save as {out} (original bytes, same aspect ratio as the source sheet). Finish with the saved path.
"""


def style_ref(bank):
    c = CAST[bank]
    return c.get('style') or f'sf3mod/build/redraw/{bank}/sheets/idle/codex.png'


def job(bank, name):
    c = CAST[bank]
    sd = OUT / bank / 'sheets' / name
    rel = lambda f: f.relative_to(ROOT).as_posix()
    kw = dict(c, src=rel(sd / 'src.png'), out=rel(sd / 'codex.png'), style=style_ref(bank), style_example=STYLE_EXAMPLE,
              note=(c['note'] + ' ') if c.get('note') else '')
    if name == 'idle':
        prompt, refs = IDLE.format(**kw), [c['concept'], c['portrait'], STYLE_EXAMPLE]
    elif name.startswith('pieces'):
        prompt, refs = PIECES.format(**kw), [kw['style'], c['concept']]
    else:
        prompt, refs = SHEET.format(**kw), [kw['style'], c['concept'], c['portrait']]
    cmd = ['codex', 'exec', '--skip-git-repo-check', '-s', 'workspace-write', '-i', rel(sd / 'src.png')]
    for r in refs: cmd += ['-i', r]
    return cmd + ['-'], prompt


def main():
    args = sys.argv[1:]
    flag = lambda k: args[args.index(k) + 1] if k in args else None
    banks = [a for a in args if a.startswith('pl')]
    jobs = int(flag('--jobs') or 4)
    only = set(flag('--only').split(',')) if flag('--only') else None
    skip = set(flag('--skip').split(',')) if flag('--skip') else set()
    redo = '--redo' in args
    todo = []
    for bank in banks:
        plan = json.loads((OUT / bank / 'plan.json').read_text())
        for n in plan:
            if (only and n not in only) or n in skip: continue
            if n != 'idle' and not (ROOT / style_ref(bank)).exists():
                print(f'{bank} {n}: no style reference yet ({style_ref(bank)}); skipped', flush=True); continue
            sd = OUT / bank / 'sheets' / n
            if not (sd / 'layout.json').exists() or redo:
                subprocess.run([sys.executable, str(SHEETS), 'make', bank, n, *map(str, plan[n])], cwd=ROOT, check=True,
                               stdout=subprocess.DEVNULL)
            todo.append((bank, n))
    queue = [t for t in todo if redo or not (OUT / t[0] / 'sheets' / t[1] / 'codex.png').exists()]
    print(f'{len(todo)} sheets, {len(queue)} to draw', flush=True)
    running, t0 = {}, time.time()
    while queue or running:
        for t, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[t]
                ok = (OUT / t[0] / 'sheets' / t[1] / 'codex.png').exists()
                print(f'{t[0]} {t[1]} codex {"ok" if ok else "FAILED"} ({time.time() - t0:.0f}s)', flush=True)
        while queue and len(running) < jobs:
            bank, n = queue.pop(0)
            sd = OUT / bank / 'sheets' / n
            if redo and (sd / 'codex.png').exists(): (sd / 'codex.png').unlink()
            cmd, prompt = job(bank, n)
            log = open(sd / 'codex.log', 'w')
            p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True,
                                 shell=sys.platform == 'win32')
            p.stdin.write(prompt); p.stdin.close()
            running[(bank, n)] = (p, log)
        time.sleep(3)
    for bank, n in todo:
        sd = OUT / bank / 'sheets' / n
        if (sd / 'codex.png').exists() and (redo or not (sd / 'stats.json').exists()
                                             or (sd / 'stats.json').stat().st_mtime < (sd / 'codex.png').stat().st_mtime):
            r = subprocess.run([sys.executable, str(SHEETS), 'post', bank, n], cwd=ROOT, capture_output=True, text=True)
            st = json.loads((sd / 'stats.json').read_text()) if (sd / 'stats.json').exists() else {}
            low = {k: v['iou_1x'] for k, v in st.items() if v['iou_1x'] < 0.65}
            print(f'{bank} {n} posted, min iou {min((v["iou_1x"] for v in st.values()), default=0):.2f}'
                  + (f' LOW {low}' if low else '') + ('' if r.returncode == 0 else ' POST FAILED'), flush=True)


if __name__ == '__main__':
    main()
