"""Layer art for the mesh renderer (sf3mod/mesh.py): one continuous BODY drawing (neck to boots, no arms, no head) and
one continuous ARM drawing (shoulder to fist) per mascot, made by Codex image generation in a fixed bind pose.

    python sf3mod/mesh_art.py guides                 # writes assets/mesh/_guide/{body,arm}.png (pose + framing guides)
    python sf3mod/mesh_art.py prompt swordcat        # prints the Codex prompt for one mascot
    python sf3mod/mesh_art.py run swordcat pnut ...  # runs one Codex job per mascot in parallel (logs in assets/mesh/<m>/)

Inputs per mascot: art/concept/<m>.png (character sheet), the existing puppet parts (style reference, assembled into
assets/mesh/<m>/_style.png) and the two guides. Outputs: assets/mesh/<m>/body.png and arm.png on flat black.
The bind joints are then found by `python sf3mod/mesh.py bind <m>` (assets/mesh/<m>/bind.json).
"""
import subprocess, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

MOD = Path(__file__).resolve().parent
ROOT = MOD.parent
OUT = MOD / 'assets' / 'mesh'
GUIDE = OUT / '_guide'

LOOKS = {
    'swordcat': 'orange tabby kitten, big green eyes, pink cherry blossom behind the left ear, silver plate armour with gold '
                'trim and gold fleur-de-lis, brown leather belt with gold buckle, armoured tasset skirt plates over orange fur '
                'thighs, silver knee cops, greaves and sabatons, silver gauntlets',
    'pnut': 'grey-brown squirrel cowboy, brown cowboy hat, red bandana, brown leather vest over a cream shirt, brown belt '
            'with a silver buckle, blue jeans, brown cowboy boots, tan leather gloves',
    'claudia': 'AI popstar girl, black blunt bob haircut with a copper streak, headset microphone, silver chrome jacket, '
               'black crop top, black shorts and tights, silver platform boots',
    'elon': 'a generic tech-billionaire PARODY caricature (exaggerated cartoon, not a real likeness): dark swept hair, black '
            'bomber jacket, black t-shirt, dark jeans, black sneakers. No logos, no text',
    'inu': 'shiba inu puppy fighter: tan/orange shiba fur with cream chest and belly, red karate gi with sleeves torn at the '
           'shoulders and a black belt, bare paws',
    'jeanphil': 'jean phil: long-faced smug man, grey houndstooth blazer, red tie, white shirt, dark trousers, brown leather shoes',
    'moodeng': 'moo deng: chubby baby pygmy hippo, glossy grey-pink skin, blue sumo belt (mawashi), stubby legs',
    'pepe': 'pepe the frog fighter: green frog man, blue t-shirt, brown trousers, bare green frog feet',
    'popcat': 'popcat: white cat, yellow boxing shorts, white hand wraps, bare white cat feet',
    'wif': 'dogwifhat: shiba inu dog in a grey hoodie, dark jogger trousers, sneakers',
    'giga': 'GIGA (gigachad meme): hyper-muscular man, shirtless, black wrestling trunks, white wrist tape, black boots',
    'bonk': 'BONK: orange shiba inu dog, long dark navy trench coat with the collar up, red scarf, black gloves, white '
            'trousers, black boots',
    'fwog': 'FWOG: small round cute green frog, no clothes, cream belly, stubby limbs, webbed hands and feet',
    'pippin': 'PIPPIN: friendly white unicorn, white tribal capoeira outfit with gold bangles, white hooves',
    'ponke': 'PONKE: cheeky brown monkey, red bandana around the neck, yellow tank top, green cargo shorts, bare monkey feet',
    'slerf': 'SLERF: sleepy brown sloth with long hooked claws, old hermit look: tattered tan loincloth, prayer beads, a '
             'rope tied around one arm',
    'chillguy': 'CHILLGUY: brown-and-beige cartoon dog-like guy, grey knit sweater with sleeves pushed up, rolled-up blue '
                'jeans, red sneakers',
    'goat': 'GOAT (Goatseus Maximus): white goat warrior, dark purple martial-arts gi with torn sleeves, thick rope belt, '
            'prayer beads around the neck, bare hooves',
    'mew': 'MEW: grey-and-white cat, black ninja outfit with a red scarf, wrapped forearms and shins, ninja tabi shoes',
    'pill': 'PILL, the final boss: the pump.fun pill mascot as a tall hyper-muscular god-like humanoid whose whole body is '
            'split exactly down the vertical middle like the pump.fun capsule: the character RIGHT half bright pump.fun '
            'green, the LEFT half glossy white; glossy polished-plastic capsule skin; a short gold-trimmed white loincloth / '
            'sash at the hips; bare muscular legs and bare feet',
}
# mascots whose two arms differ: image C, assets/mesh/<m>/arm_far.png (the far arm)
ARM_FAR = {
    'pill': 'the same complete arm as image B, same size and pose, but WHITE: glossy white capsule skin with cool grey '
            'shading (the arm of the white half), gold bracer',
}
# what goes on the arm layer, when it isn't obvious from the look
ARM_NOTES = {
    'swordcat': 'the layered silver shoulder pauldron with gold rivets at the top, orange fur upper arm with a brown strap, '
                'silver vambrace, armoured gauntlet fist',
    'pnut': 'cream shirt sleeve with the rounded shoulder at the top, rolled cream sleeve at the elbow, grey-brown fur '
            'forearm, tan leather glove fist',
    'pill': 'bare massive GREEN muscular arm (glossy green capsule skin), a gold shoulder clasp at the top, gold bracer at '
            'the wrist, big green fist',
}


# what the body layer must do, when the outfit fights the A-stance (long coats, skirts) or the shape is unusual
BODY_NOTES = {
    'bonk': 'The trench coat stops at the hips and is split at the sides, so each leg (white trousers, black boots) is '
            'free; the coat tails hang over the top of each thigh separately, with a clear gap between the legs.',
    'moodeng': 'Chubby round belly; short stubby hippo legs, still apart with a gap between them.',
    'fwog': 'Round cream belly; short stubby frog legs, still apart with a gap between them, webbed feet.',
    'slerf': 'The loincloth hangs only at the front and back, short, so the legs are free with a gap between them.',
    'goat': 'The gi trousers are loose but each leg is separate with a gap between them; bare hooves.',
    'pippin': 'Each leg is separate with a gap between them; white hooves for feet.',
    'popcat': 'Yellow boxing shorts; bare white cat legs and feet.',
    'inu': 'Full red karate gi: the red gi jacket closed with the black belt (knot and belt tails at the front), and long '
           'loose RED gi TROUSERS down to the ankles on both legs (like a classic karate fighter), the legs apart with a '
           'clear gap between them; bare shiba paws for feet below the trouser hems.',
    'pill': 'The body is split exactly down the vertical middle: the near half (the leg on the image LEFT and the left part '
            'of the chest as seen) bright green, the far half (the leg on the image RIGHT) glossy white. No cape in this '
            'layer (the cape is separate).',
    'giga': 'Absolutely NO arm or hand anywhere in this image, not even folded behind the back or across the belly: '
            'just the bare torso with the pectorals and abs fully visible, the shoulders cut off flush at the sides.',
}


def guides():
    """Grey mannequins showing the bind pose and framing (1024x1536)."""
    GUIDE.mkdir(parents=True, exist_ok=True)
    col = (110, 110, 118)

    def cap(im, a, b, ra, rb):
        a, b = np.array(a, float), np.array(b, float)
        for t in np.linspace(0, 1, 60):
            p = a + (b - a) * t; r = ra + (rb - ra) * t
            cv2.circle(im, tuple(int(v) for v in p), int(r), col, -1, cv2.LINE_AA)

    im = np.zeros((1536, 1024, 3), np.uint8)
    cap(im, (512, 215), (512, 275), 48, 52)                     # neck stub
    cv2.fillPoly(im, [np.array([[372, 280], [652, 280], [620, 640], [404, 640]])], col, cv2.LINE_AA)  # chest
    cv2.fillPoly(im, [np.array([[404, 630], [620, 630], [650, 780], [374, 780]])], col, cv2.LINE_AA)  # hips
    for hx, kx, ax in ((436, 372, 326), (588, 652, 698)):
        cap(im, (hx, 760), (kx, 1040), 82, 64)                  # thigh
        cap(im, (kx, 1040), (ax, 1320), 60, 46)                 # shin
        cap(im, (ax - 10, 1345), (ax + 105, 1372), 46, 34)      # foot, toes to the right
    Image.fromarray(im).save(GUIDE / 'body.png')
    im = np.zeros((1536, 1024, 3), np.uint8)
    cap(im, (512, 250), (512, 760), 92, 66)                     # shoulder cap + upper arm
    cap(im, (512, 760), (512, 1150), 64, 50)                    # forearm
    cap(im, (512, 1240), (512, 1300), 92, 88)                   # fist
    Image.fromarray(im).save(GUIDE / 'arm.png')
    print('guides ->', GUIDE)


def style_sheet(m):
    """The existing cut-out parts of the mascot side by side: the style reference."""
    d = MOD / 'assets' / 'puppet' / m
    fs = [d / f'{n}.png' for n in ('torso', 'pelvis', 'thigh', 'shin', 'foot', 'upperarm', 'forearm', 'fist') if (d / f'{n}.png').exists()]
    sh = Image.new('RGB', (384 * len(fs), 384))
    for i, f in enumerate(fs): sh.paste(Image.open(f).convert('RGB').resize((384, 384), Image.LANCZOS), (i * 384, 0))
    p = OUT / m / '_style.png'; p.parent.mkdir(parents=True, exist_ok=True); sh.save(p)
    return p


def prompt(m):
    rel = lambda p: p.relative_to(ROOT).as_posix()
    d = OUT / m
    arm = ARM_NOTES.get(m, "the character's sleeve or shoulder at the top, the full arm, the hand clenched into a fist")
    body_note = (BODY_NOTES[m] + ' ') if m in BODY_NOTES else ''
    n_img = 3 if m in ARM_FAR else 2
    far = (f"C) {rel(d / 'arm_far.png')} - the FAR ARM layer: {ARM_FAR[m]}. Same framing as B.\n") if m in ARM_FAR else ''
    return f"""Use your image generation tool. Do not write code (only save / copy the generated files).
This is a private Street Fighter III mod, "PUMP COMBAT". Character: {m.upper()} - {LOOKS[m]}.

Attached images, in order:
1) art/concept/{m}.png - the character sheet (identity, colours, outfit). Keep the character exactly on-model.
2) {rel(d / '_style.png')} - the character's existing cut-out puppet parts. MATCH THIS STYLE EXACTLY: bold readable
   shapes, thick dark outline (#14101c), clean cel shading with 3 tones per material, light from the upper left, no
   painterly texture, no fur noise, no glow, no cast shadows. Same palette and line weight.
3) sf3mod/assets/mesh/_guide/body.png - grey mannequin: the POSE and FRAMING of image A below.
4) sf3mod/assets/mesh/_guide/arm.png - grey mannequin: the POSE and FRAMING of image B below.
The mannequins only show pose and placement: do not copy their grey colour; use the character's own proportions.

Purpose: layers for 2D mesh skinning (like Spine / DragonBones): a script deforms each drawing with bones onto every
animation frame of a fighting-game character (the whole body ends up ~110 px tall), so each layer must be ONE
continuous drawing with no gaps or seams between its body parts. View: 3/4 side view of a fighter whose body faces
the viewer's RIGHT (like a fighting-game character on the left side of the screen), slightly turned toward camera.

Make exactly {n_img} images, portrait 2:3 (1024x1536), each on a pure flat black #000000 background, nothing else in the
image (no text, no ground, no shadow), and save them with original bytes, no resizing:

A) {rel(d / 'body.png')} - the BODY layer: the whole character from the neck down WITHOUT ARMS and WITHOUT HEAD, as one
   continuous drawing: a short neck stub / collar at the top centre, chest, belly, waist, hips, both legs and both feet
   (pelvis and legs joined naturally like a real body, no separate plates, no seams at the hips or knees). Pose exactly
   like the guide: standing upright, legs straight and apart in a wide A-stance, a clear black gap between the legs
   from the crotch down, both feet flat on one line with the toes pointing RIGHT. No arms at all: the sides of
   the chest end flush at the armpits, with NO shoulder caps, arm stubs or sleeves sticking out (the arm layer is
   drawn separately and covers the shoulder). No tail, no cape, no weapon, no hands. {body_note}Fill about 85% of
   the image height, centred.
B) {rel(d / 'arm.png')} - the ARM layer: one complete arm as one continuous drawing from shoulder to fist: {arm}.
   Shoulder at the TOP, elbow in the middle, wrist, and the fist at the BOTTOM with the knuckles pointing down.
   Hanging straight down, the arm's long axis exactly vertical and centred, elbow straight. Seen from the outside of
   the arm. Same scale as the body in image A (arm length from shoulder to wrist about the length from the neck stub
   to the crotch). Fill about 80% of the image height.
{far}
Finish with the saved paths."""


def body_only(text):
    """The prompt cut down to image A (redo a body layer, keep the arm)."""
    a = text.index('B) sf3mod/'); b = text.index('Finish with')
    return text[:a].replace('Make exactly 2 images', 'Make exactly 1 image (image A)').replace('Make exactly 3 images', 'Make exactly 1 image (image A)') + 'Finish with the saved path.'


def run(ms, only=None):
    jobs = []
    for m in ms:
        d = OUT / m; d.mkdir(parents=True, exist_ok=True)
        st = style_sheet(m)
        (d / '_prompt.txt').write_text(body_only(prompt(m)) if only == 'body' else prompt(m), encoding='utf-8')
        imgs = [ROOT / 'art' / 'concept' / f'{m}.png', st, GUIDE / 'body.png', GUIDE / 'arm.png']
        cmd = ['codex', 'exec', '--skip-git-repo-check', '-s', 'workspace-write']
        for i in imgs: cmd += ['-i', str(i)]
        cmd += ['-']
        log = open(d / '_codex.log', 'w', encoding='utf-8')
        jobs.append((m, subprocess.Popen(cmd, cwd=ROOT, stdin=open(d / '_prompt.txt', 'rb'), stdout=log,
                                         stderr=subprocess.STDOUT, shell=sys.platform == 'win32')))
    for m, p in jobs:
        p.wait()
        ok = all((OUT / m / f).exists() for f in ('body.png', 'arm.png'))
        print(m, 'exit', p.returncode, 'OK' if ok else 'MISSING OUTPUT', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'guides': guides()
    elif cmd == 'prompt': print(prompt(sys.argv[2]))
    elif cmd == 'run':
        args = [a for a in sys.argv[2:] if not a.startswith('--')]
        run(args, 'body' if '--body-only' in sys.argv else None)
