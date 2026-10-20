#!/usr/bin/env bash
# Builds the PUMP FIGHTERS browser version into web/dist (Git Bash on Windows, macOS or Linux).
#   bash web/build.sh            Release build
#   bash web/build.sh Debug      assertions + source maps
# Needs: emsdk activated (emcmake on PATH), cmake, ninja, and a 3SX clone with web/3sx-web.patch applied.
# The 3SX clone is looked for in $THREESX, default web/3sx. See web/BUILDING.md.
# No game data is copied: players bring their own SF33RD.AFS and it is patched in their browser.
set -euo pipefail

WEB="$(cd "$(dirname "$0")" && pwd)"
ROOT="$WEB/.."
CONFIG="${1:-Release}"
THREESX="${THREESX:-$WEB/3sx}"

command -v emcmake >/dev/null || { echo "emcmake not found: activate emsdk first (source <emsdk>/emsdk_env.sh)"; exit 1; }
[ -f "$THREESX/CMakeLists.txt" ] || { echo "no 3SX clone at $THREESX (set THREESX=...; see web/BUILDING.md)"; exit 1; }
[ -d "$THREESX/src/platform/web" ] || { echo "$THREESX is not patched yet: cd $THREESX && git apply $WEB/3sx-web.patch"; exit 1; }

BUILD="$WEB/build/3sx-$CONFIG"
emcmake cmake -S "$THREESX" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE="$CONFIG"
cmake --build "$BUILD"

DIST="$WEB/dist"
mkdir -p "$DIST/icons"
cp "$BUILD/pumpcombat.js" "$BUILD/pumpcombat.wasm" "$DIST/"
[ -f "$BUILD/pumpcombat.wasm.map" ] && cp "$BUILD/pumpcombat.wasm.map" "$DIST/" || rm -f "$DIST/pumpcombat.wasm.map"
cp "$WEB"/src/*.html "$WEB"/src/*.js "$WEB"/src/*.css "$WEB"/src/manifest.webmanifest "$DIST/"
cp "$WEB"/src/icons/*.png "$DIST/icons/"
# Bring-your-own-file mode: the patcher library (shared with patcher/) and the patch
# (manifest + VCDIFF deltas against the player's own file; no game data). The patch files are
# attached to the GitHub Release: download them into patch/ first (see patch/README.md).
cp "$ROOT/patcher/afs-patch.js" "$DIST/"
rm -rf "$DIST/patch"
if [ -f "$ROOT/patch/pumpfighters.json" ]; then
  mkdir -p "$DIST/patch"
  cp "$ROOT/patch/pumpfighters.json" "$ROOT"/patch/pumpfighters-*.deltas "$DIST/patch/"
else
  echo "note: patch/pumpfighters.json not found; the site will build but cannot patch until you add the release patch files"
fi
# Never upload a local copy of the game data as a static asset
printf '%s\n' '*.AFS' '*.afs' > "$DIST/.assetsignore"

ls -la "$DIST"
