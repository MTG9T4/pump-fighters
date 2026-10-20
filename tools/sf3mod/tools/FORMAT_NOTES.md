# Format details, constraints and pattern map

Put completed replacement files in `sf3mod/build/overrides/` under their original
AFS names before repacking. Unmatched files (including `index.json`) are ignored.
Empty overrides, or overrides identical to originals, produce an exact archive
copy, including padding. Changed-size repacks preserve every other entry's
payload, names/dates and trailer; offsets, affected size fields and sector padding
are rebuilt. Generic AFS repacking does not validate asset-specific size budgets.

PPG indexes are zero-based `pTEX` indexes. `topng` also writes `textures.json`.
It handles 4/8/16/32bpp, raw/LZ77/zlib, endian flags, nibble order, and transform
tables. `frompng` keeps dimensions, pixel format, transform data and compression
mode; other chunks are copied exactly. Direct formats include ARGB1555,
RGB565, ARGB4444 and ARGB8888. Paletted inputs are quantized by RGBA distance to
the existing palette. Compression 1 uses valid literal blocks and can grow;
compression 2 uses zlib. An unchanged PNG retains the original bytes exactly.

**PPG palettes are chosen by drawing code, not bound to textures in the file.**
`--palette N` chooses a zero-based row among compatible `pPAL` sets (16 colours
for 4bpp, 256 for 8bpp, falling back to 64). Default is 0. For files without
palettes, such as `SelectNew.ppg`, both commands accept `--palette-source PATH`
(another PPG, or raw LE ARGB1555). A sibling `default.bin` is used automatically
when available. Raw rows are grouped by the texture's 16/256-colour capacity;
default.bin group 0 corresponds to ColorRAM 32..35 for an 8-bit texture.
These external/default-row exports are palette interpretations; the game can
draw the same atlas with other rows. Choose the appropriate row/source for edits.

Sprite patterns use **bank-local numbers**. Global CG = local + bank base (27104
for ef02). `render` accepts a palette file or `auto`; `dump` and `replace` accept
`--palette-source`. Automatic sources are sibling `default.bin` or `plNNpl.bin`,
with a read-only fallback to the original AFS. Use `--kind ef02_usa.bin` (or the
original character bank name) for renamed banks. `--row` selects character colour
0..27 (selectable 0..12); `--half 1` selects Gill's second set. Non-Gill part rows
use selected colour + auxiliary rows 16..21 exactly as the loader does; Gill maps
the second body set separately. `--base` overrides the inferred ColorRAM base: 0x90 for select portraits/names,
plus `Victory_Color_Data` for the mapped win-name patterns.

Sprite PNGs use tightly bounded, unscaled tile canvases; render JSON records the
original origin. Delta positions, Morton ordering, tile cropping, flip bits,
per-part palettes and foreground ordering are applied. Replacement PNGs must
retain that canvas and use each tile's exact RGBA palette. Portraits often use
**several** 64-colour rows, rather than one palette for the whole image.
Only changed visible pixels are encoded; hidden pixels remain intact. Shared
cells are cloned, preventing changes to other patterns. Replacements that exceed
the fixed texture budget, alias a transform list, introduce out-of-tile pixels,
or have unresolved overlap/palette conflicts fail without writing the output.
Cells are rebuilt with the ported `compress_p6`; `to_tex`, character `to_chd`,
bank length and character animation/hit data retain their original positions.

Palette JSON uses editable `#RRGGBBAA` colours with explicit half/row numbers.
Import rounds to ARGB1555 and preserves unlisted rows. Export includes every
stored row. Preview follows the original `nmca[0]` idle script, including Yun's
loop setup and Remy's subroutine, to select three distinct standing frames.
It is a static sprite preview; it does not emulate gameplay effects or timing.

## Verified original formats and budgets

| Asset | Real archive findings |
|---|---|
| AFS | 642,492,416 bytes; 1,535 named entries; 2,048-byte sector alignment; 48-byte name rows |
| `TitleTM.ppg` (78) | 38,496 bytes; one 512x256 16bpp ARGB1555 texture, BE pixel words, zlib |
| `scrscrn.ppg` (10) | 50,348 bytes; 7 textures, mixed 4/8bpp; 4 pPAL chunks |
| `SelectNew.ppg` (11) | 53,140 bytes; 24 8bpp textures; external palettes |
| `ef02_usa.bin` (1454) | 1,128,236 bytes; 1,863 local patterns; 5,739 LZ-p6 cells; `to_tex=0x20CA4` |
| Select portraits | 216..400 pixels wide, 184..248 high; 43..93 tiles; 6-bit indices, 64-colour rows, ARGB1555 LE |
| Select big names | 72..152 pixels wide, 24..31 high; win names 40..224 by 32..34 |
| `default.bin` (9) | 61,440 bytes; 480x64 ARGB1555 LE; ColorRAM row r = file row r-32 |
| `pl00pl.bin` (Gill) | 7,168 bytes, `u16[2][28][64]` |
| Other 19 `plNNpl.bin` | 3,584 bytes each, `u16[28][64]`; no pl15; selectable rows 0..12 |

**ef02 limit:** the tools conservatively cap it at the original **1,128,236 bytes**.
PC `reservMemKeySelObj` uses that exact size; PSP reserves
`ceil(0x11372C/2048)*2048 = 1,128,448` bytes. The extra 212 bytes are sector
slack, not an approved expansion budget. Character banks also retain fixed
`to_tex`/`to_chd` offsets from `banks.json`. Enlarging an AFS entry alone does not
make fixed game allocations or offsets safe. New dimensions/layouts/palette
counts require game code changes outside this toolkit's scope.

## Portrait and big-name map

Numbers below are local to `ef02_usa.bin`. English select names are used for
Akuma. Global CG adds 27104. Win names are a separate sprite set.

| ID | Character | Portrait | Select name | Win name |
|---:|---|---:|---:|---:|
| 0 | Gill | 1179 / 1202 | 4 | 855 |
| 1 | Alex | 1180 | 5 | 856 |
| 2 | Ryu | 1181 | 6 | 857 |
| 3 | Yun | 1182 | 7 | 858 |
| 4 | Dudley | 1183 | 8 | 859 |
| 5 | Necro | 1184 | 9 | 860 |
| 6 | Hugo | 1185 | 10 | 861 |
| 7 | Ibuki | 1186 | 11 | 862 |
| 8 | Elena | 1187 | 12 | 863 |
| 9 | Oro | 1188 | 13 | 864 |
| 10 | Yang | 1189 | 14 | 865 |
| 11 | Ken | 1190 | 15 | 866 |
| 12 | Sean | 1191 | 16 | 867 |
| 13 | Urien | 1192 | 17 | 868 |
| 14 | Akuma | 1193 | 19 | 870 |
| 15 | Chun-Li (pl16) | 1194 | 20 | 871 |
| 16 | Makoto (pl17) | 1195 | 21 | 872 |
| 17 | Q (pl18) | 1196 | 22 | 873 |
| 18 | Twelve (pl19) | 1197 | 23 | 874 |
| 19 | Remy (pl20) | 1198 | 24 | 875 |

Gill uses portrait **1179 for first-player select**, substituted through animation
index 23 in `eff38.c`; the main table entry (second-player side) is **1202**.
Japanese Gouki names are 18 (select), 869 (win). Portrait variants 1199..1201
also exist. `verify.py --real` resolves the animation tables again and dumps all
60 mapped sprites plus those six alternates to `sf3mod/build/ef02/`; its
`pattern_map.json` includes dimensions, tile counts and exact ColorRAM rows.

`verify.py` tests codec instructions, endian/nibble modes, all depth/compression
combinations and real modifications. `--real` additionally extracts the archive,
checks every real PPG, writes exact-empty and changed-size AFS repacks, validates
all sprite cells, checks replacement isolation against every other ef02 pattern,
and exports/round-trips all 20 palettes with standing previews. Evidence is in
`sf3mod/build/verification.json`. This is offline format verification; emulator
and PSP runtime behaviour remain untested.

Source references (read-only): PC `port/io/afs.c`, PSP `psp/afs.c`; PC
`sf33rd/Source/Common/PPGFile.c` and `Source/Compress/Lz77/Lz77Dec.c`;
`arcade/arcade_cg.c` (`swizzle`, `compress_p6`); rendering `mtrans.c`,
`color3rd.c`, `texgroup.c`, `texgroup_data.c`; PSP `Game/texgroup.c` and
`Game/GD3rd.c`; effect `eff38.c`/`eff39.c`/`eff76.c`, `screen/sel_data.c`,
`bin2obj/char_table.c`; engine `charid.c`/`charset.c`. PC paths are relative to
`mod/3sx/src/`, PSP paths to `mod/3s-psp/src/`. `banks.json` records the 21
supported original bank layouts; runtime commands need no source tree.
