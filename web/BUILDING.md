# Building the browser version

The browser game is **[3SX](https://github.com/crowded-street/3sx)** (the native 3rd Strike port,
AGPL-3.0) compiled to WebAssembly with Emscripten, plus our patch `3sx-web.patch` and the page in `src/`.
Because 3SX is AGPL-3.0, this source is published alongside every hosted build. If you host a modified
build, you must publish your changes too.

No game data is built in. The page asks each player for their own `SF33RD.AFS` and patches it in their
browser with `patcher/afs-patch.js`.

## What you need

- [emsdk](https://emscripten.org/docs/getting_started/downloads.html) (latest), activated so `emcmake` is on PATH
- CMake 3.20+ and Ninja
- git, bash (Git Bash works on Windows), Node 18+
- the patch files from the GitHub Release, in `patch/` (see `patch/README.md`)

## Steps

```bash
# 1. emsdk
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh && cd ..

# 2. 3SX at the commit our patch was made against, with our patch applied
git clone https://github.com/crowded-street/3sx.git web/3sx
cd web/3sx
git checkout 8ed7a0304382f0693c88ae0130a706541f022811
git apply ../3sx-web.patch
cd ../..

# 3. Build: web/dist gets pumpcombat.js/.wasm, the page, afs-patch.js and patch/
bash web/build.sh            # or: bash web/build.sh Debug
```

Use another 3SX checkout with `THREESX=/path/to/3sx bash web/build.sh`. Newer 3SX commits may need the patch
rebased.

## What the patch changes in 3SX

- `CMakeLists.txt`: a WebAssembly target (`pumpcombat.js` / `.wasm`) with SDL3's Emscripten backend,
  WebGL2 and Web Audio; no ISO import, ROM zips, retail AFS checksum or desktop GL loader on web.
- `src/platform/web/`: the bridge to the page (the page hands the game its archive as `Module.pcAfs`),
  WebGL2 versions of the shaders, and stubs for the arcade ROM loader.
- `src/port/io/afs.c`, `src/port/resources.c`, `src/port/sound/adx.c`: read the archive through the bridge,
  with music streamed asynchronously.
- `src/platform/app/sdl/sdl_app.c`, `input/sdl/sdl_pad.c`, `video/opengl/opengl_renderer.c`: browser main
  loop, pause while the page shows something else, touch/pad input, and WebGL2 texture formats.
- `src/sf33rd/Source/Game/effect/eff48.c`: the opening's name cards show the mascots' names.

## Running it locally

Serve `web/dist` over http(s) from localhost with any static server, for example:

```bash
npx http-server web/dist -p 8790
```

Open http://localhost:8790/ and pick your `SF33RD.AFS`. `patch.html` is the PSP / PPSSPP helper that saves the
modded file instead of starting the game.

## Hosting

`web/dist` is a plain static site: put it on any static host that serves it over HTTPS (OPFS and the
file picker need a secure context). Serve `.wasm` as `application/wasm`. No special headers are needed:
the game is single-threaded, so it does not need COOP/COEP.

### On Cloudflare Workers (optional)

`worker/` serves `web/dist` as static assets and adds cache headers (`no-cache` for the page, engine
and patch files, a day for icons).

```bash
cd web/worker
# edit wrangler.jsonc: your own Worker name, and a route if you want your own domain
npx wrangler deploy
```

Worker tests: `node --test web/worker/test/worker.test.mjs`.
