# Modding pipeline

This folder is the pipeline that made the PUMP FIGHTERS patch. You don't need it to play. It's here so you
can see how the mod was made, change a mascot, or build your own.

It needs **your own game files** and, for the sprite skins, **an NVIDIA GPU** for pose reading. None of the
game data it reads or writes is in this repo, and none of it should ever be committed (see `.gitignore`).

Run every command from this `tools/` folder. The scripts treat it as the project root.

## Layout

```
tools/
  game/SF33RD.AFS        <- put your original here (PS2 US v1.0; never modified, git-ignored)
  art/concept/           mascot character sheets (our art)
  sf3mod/
    tools/               format toolkit: afs.py, ppg.py, spritebank.py, pal.py, codecs; FORMAT_NOTES.md
    build_mod.py         builds every override file and repacks the archive
    skin.py, skins/      full-body skins: mascot art posed onto every frame of a fighter
    pose.py, anims.py    pose reading (RTMPose) and animation-script parsing per fighter
    mesh.py, mesh_art.py mesh skinning (body + arm layers, bone weights)
    hud.py, names.py     HUD name plates, face icons, select-screen names
    opening.py           redrawn opening / attract mode
    recolor.py           fighter palettes
    assets/              our art: gen/ (logo, portraits), puppet/ (cut-out parts), mesh/ (layer art,
                         bind joints, guides), opening/ (redrawn opening frames)
    build/               all output (git-ignored)
```

## Requirements

- Python 3.12 with `pillow`, `numpy`, `opencv-python`
- for `pose.py` / `mesh.py bind`: `rtmlib`, `onnxruntime-gpu` (and `torch` for its CUDA DLLs on Windows).
  It runs on CPU too, but very slowly.
- for `hud.py` and `verify.py`: source checkouts of 3SX at `tools/mod/3sx`
  (https://github.com/crowded-street/3sx) and 3s-psp at `tools/mod/3s-psp`
  (https://github.com/demmis98/3s-psp); they read tables from the decompiled source.
- Node 18+ for the patch generator.
- `mesh_art.py run` and the original art were made with an AI image tool (OpenAI Codex CLI); you only need
  it to make new art.

## Build the mod

```bash
# 1. unpack the archive (about 640 MB of output)
python sf3mod/tools/afs.py extract game/SF33RD.AFS sf3mod/build/extracted

# 2. read every fighter's poses (GPU), then build the skins (slow: hours for all 20)
python sf3mod/pose.py pl02 pl11 ...        # bank ids: see CAST in build_mod.py / SKINS in skin.py
python sf3mod/skin_all.py --jobs 2
python sf3mod/skin_audit.py all            # checks no frame still shows the original fighter

# 3. build everything and repack -> sf3mod/build/SF33RD.AFS
python sf3mod/build_mod.py --only logo,portraits,names,palettes,hud,skins,opening

# 4. make the patch (from the repo root) -> patch/pumpfighters.json + pumpfighters-N.deltas
cd .. && node patcher/tools/make-patch.mjs --audit
```

`make-patch.mjs --audit` also checks that no literal bytes in the deltas appear verbatim in the original
archive, so the patch carries changes, not game data.

For a quick test on PSP / PPSSPP, `python sf3mod/pack_phone.py` lays out `sf3mod/build/phone/PSP/GAME/...`
once you've put the 3s-psp `EBOOT.PBP` (from its own release) at `tools/psp/EBOOT.PBP`.

## Changing a mascot

1. Draw or generate a character sheet in `art/concept/<name>.png`.
2. Make the portrait (`sf3mod/assets/gen/portrait_<name>.png`) and the puppet parts
   (`sf3mod/assets/puppet/<name>/`: head, torso, pelvis, upperarm, forearm, fist, thigh, shin, foot, optional
   tail / cape), each on a flat black background.
3. For smooth joints, make the mesh layers with `python sf3mod/mesh_art.py prompt <name>` (body.png +
   arm.png) and find the bind joints with `python sf3mod/mesh.py bind <name>`.
4. Point a fighter at it in `CAST` / `NAMES` (`build_mod.py`) and `SKINS` (`skin.py`); per-fighter fixes go
   in `sf3mod/skins/plNN.py`.
5. Rebuild that fighter (`python sf3mod/skin.py plNN --optimal`) and the archive.

The file formats are documented in `sf3mod/tools/FORMAT_NOTES.md` and `sf3mod/tools/README.md`.
