# PUMP COMBAT asset toolkit

Python 3.12 + Pillow + numpy. Run from the project root. Every CLI and subcommand
supports `--help`. Put outputs in git-ignored `sf3mod/build/`. Tools refuse to
overwrite input files or `game/SF33RD.AFS`; they never commit.

```powershell
python sf3mod/tools/afs.py list game/SF33RD.AFS
python sf3mod/tools/afs.py extract game/SF33RD.AFS sf3mod/build/extracted
python sf3mod/tools/afs.py repack game/SF33RD.AFS sf3mod/build/overrides sf3mod/build/mod.afs

python sf3mod/tools/ppg.py topng sf3mod/build/extracted/TitleTM.ppg sf3mod/build/title
python sf3mod/tools/ppg.py frompng sf3mod/build/extracted/TitleTM.ppg 0 sf3mod/build/title/tex_000.png sf3mod/build/TitleTM.ppg

python sf3mod/tools/spritebank.py render sf3mod/build/extracted/ef02_usa.bin auto 1181 sf3mod/build/ryu_portrait.png
python sf3mod/tools/spritebank.py dump sf3mod/build/extracted/ef02_usa.bin --patterns 1180-1202 sf3mod/build/portraits
python sf3mod/tools/spritebank.py replace sf3mod/build/extracted/ef02_usa.bin 1181 sf3mod/build/ryu_portrait.png sf3mod/build/ef02_usa.bin

python sf3mod/tools/pal.py export sf3mod/build/extracted/pl02pl.bin sf3mod/build/ryu_palette.png
python sf3mod/tools/pal.py import sf3mod/build/ryu_palette.json sf3mod/build/extracted/pl02pl.bin sf3mod/build/pl02pl.bin
python sf3mod/tools/pal.py preview sf3mod/build/extracted/pl02.bin sf3mod/build/extracted/pl02pl.bin 0 sf3mod/build/ryu_preview.png

python sf3mod/tools/verify.py --real
```

Create the overrides directory before repacking. Only matching original AFS
filenames replace entries; an empty directory produces a byte-identical archive.
Use a separate empty directory for a no-op test. Generic AFS repacking does not
check asset-specific budgets.

PPG texture indexes and sprite pattern numbers are zero-based. PPG palette rows
are selected externally by drawing code: `--palette` / `--palette-source` choose
the export/edit interpretation. Files without embedded palettes default to a
sibling `default.bin` when present. Sprite palettes default to `default.bin` or
the matching `plNNpl.bin`; `--row` selects character colour (0..12 selectable).
Renamed sprite banks need `--kind` with their original filename.

Keep exported sprite canvas dimensions; edits must use the exact palette of
each tile. Shared cells are cloned. Budget, overlap and palette conflicts are
rejected. ef02 replacements stay within **1,128,236 bytes**. Palette JSON edits
use `#RRGGBBAA`, rounded on import to ARGB1555.

See [FORMAT_NOTES.md](FORMAT_NOTES.md) for the **20-character portrait/name map**,
verified dimensions, palette layout differences, fixed offsets and source
references. `verify.py --real` creates the 60 mapped sprites plus six alternates
in `sf3mod/build/ef02/`, all character palette swatches/previews, PPG exports and
repack examples (about 2 GB scratch). Results: `sf3mod/build/verification.json`.
Offline verification passes; emulator/PSP runtime behaviour is untested.
