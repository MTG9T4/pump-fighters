// PUMP FIGHTERS web shell: stores the game data, shows the touch controls and starts the
// WebAssembly build of 3SX (pumpcombat.js / pumpcombat.wasm).
//
// Bring your own file: the player picks SF33RD.AFS from their own copy of the game, it is checked
// and patched on the device (afs-patch-worker.js + patch/pumpfighters.json) and kept in this
// browser's own storage (OPFS). Nothing is uploaded.

import { AfsStore, fetchPatchManifest } from './afs-store.js';
import { TouchControls } from './controls.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const patchUrl = document.querySelector('meta[name="pc-patch-url"]').content;

const app = $('app');
const loader = $('loader');
const statusEl = $('status');
const detailEl = $('detail');
const hintEl = $('hint');
const barEl = $('bar');
const barFill = $('bar-fill');
const playBtn = $('btn-play');
const retryBtn = $('btn-retry');

// ---------------------------------------------------------------- settings

const SETTINGS_KEY = 'pumpcombat.settings';
const settings = { controls: 'auto', haptics: true, scale: 1, ...readJson(SETTINGS_KEY) };

function readJson(key) {
  try {
    return JSON.parse(localStorage.getItem(key) || '{}') || {};
  } catch (_) {
    return {};
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (_) {
    // private browsing: settings are not kept
  }
}

const touchDevice = () => matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;

function applySettings() {
  const show = settings.controls === 'on' || (settings.controls === 'auto' && touchDevice());
  if (params.get('controls') === 'on' || params.get('controls') === 'off') {
    app.classList.toggle('no-controls', params.get('controls') === 'off');
  } else {
    app.classList.toggle('no-controls', !show);
  }
  document.documentElement.style.setProperty('--scale', String(settings.scale));
  controls.haptics = !!settings.haptics;
  // Let SDL re-read the canvas size
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}

// ---------------------------------------------------------------- game module

let Module = null;
let started = false;

const controls = new TouchControls({
  stickZone: $('stick-zone'),
  stickBase: $('stick'),
  stickKnob: $('stick-knob'),
  buttonZone: $('button-zone'),
  holdButtons: [...document.querySelectorAll('.chip.hold')],
  onChange: (mask) => Module?._Web_SetTouchState?.(mask),
  onFirstTouch: resumeAudio,
});

function resumeAudio() {
  const ctx = Module?.SDL3?.audioContext;
  if (ctx && ctx.state !== 'running') ctx.resume().catch(() => {});
}

// iOS only unlocks audio on touchend/click, so try on every kind of user input
for (const type of ['pointerup', 'touchend', 'click', 'keydown']) {
  document.addEventListener(type, resumeAudio, true);
}

function mountSaves(mod) {
  // SDL keeps config and memory-card saves under /libsdl; keep them in IndexedDB
  try {
    mod.FS.mkdir('/libsdl');
    mod.FS.mount(mod.IDBFS, {}, '/libsdl');
    mod.addRunDependency('pc-saves');
    mod.FS.syncfs(true, (err) => {
      if (err) console.warn('[saves] load failed', err);
      mod.removeRunDependency('pc-saves');
    });

    let syncing = false;
    const flush = () => {
      if (syncing) return;
      syncing = true;
      mod.FS.syncfs(false, (err) => {
        syncing = false;
        if (err) console.warn('[saves] store failed', err);
      });
    };

    setInterval(flush, 10000);
    document.addEventListener('visibilitychange', () => document.hidden && flush());
    window.addEventListener('pagehide', flush);
  } catch (error) {
    console.warn('[saves] not available', error);
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(s);
  });
}

async function startGame(store) {
  playBtn.hidden = true;
  setStatus('Starting the game…');
  barEl.hidden = true;
  keepAwake();

  await loadScript('pumpcombat.js');

  const moduleArg = {
    canvas: $('canvas'),
    pcAfs: store,
    arguments: [],
    print: (text) => console.log(text),
    printErr: (text) => console.warn(text),
    preRun: [mountSaves],
    onGameStarted: () => {
      started = true;
      loader.classList.add('hidden');
      Module?._Web_SetTouchState?.(controls.mask);
      window.dispatchEvent(new Event('resize'));
    },
    onAbort: (what) => showError(`The game stopped: ${what}`),
  };

  try {
    Module = await window.createPumpCombat(moduleArg);
    window.pc.Module = Module;
  } catch (error) {
    showError(error);
  }
}

// ---------------------------------------------------------------- loader UI

function setStatus(text, detail = '') {
  statusEl.textContent = text;
  detailEl.textContent = detail;
}

const mb = (bytes) => (bytes / 1e6).toFixed(bytes > 1e8 ? 0 : 1);

/** Progress of loadResident (phase 'unpack'): the stored archive is read into memory. */
function onProgress({ loaded = 0, total = 0 }) {
  barEl.hidden = false;
  barFill.style.width = total ? `${((loaded / total) * 100).toFixed(1)}%` : '0%';
  setStatus('Loading…', `${mb(loaded)} / ${mb(total)} MB`);
  hintEl.textContent = '';
}

function showError(error) {
  loader.classList.remove('hidden');
  barEl.hidden = true;
  playBtn.hidden = true;

  console.error(error);
  retryBtn.hidden = false;
  setStatus('Something went wrong', String(error && error.message ? error.message : error));
}

retryBtn.addEventListener('click', () => location.reload());

async function boot() {
  applySettings();
  setStatus('Checking game data…');

  if (!window.WebAssembly || !document.createElement('canvas').getContext('webgl2')) {
    showError('This browser needs WebAssembly and WebGL2.');
    return;
  }

  const store = new AfsStore();
  window.pc = { store, Module: null };

  await bootByo(store);
}

function ready(store) {
  byoEl.hidden = true;
  loader.classList.remove('warn');
  barEl.hidden = true;
  setStatus('Ready');
  hintEl.textContent = touchDevice()
    ? 'Tip: add this page to your home screen to play full screen.'
    : 'Keys: arrows/WASD, U I O punch, J K L kick, Enter start.';

  if (params.get('autostart') === '1') {
    startGame(store).catch(showError);
    return;
  }

  playBtn.hidden = false;
  playBtn.focus();
  playBtn.addEventListener('click', () => startGame(store).catch(showError), { once: true });
}

// ---------------------------------------------------------------- bring your own file

const byoEl = $('byo');
const pickBtn = $('btn-pick');
const fileInput = $('byo-file');

async function bootByo(store) {
  const manifest = await fetchPatchManifest(patchUrl);
  const stored = AfsStore.storedMeta();

  try {
    if (await store.openPatched(manifest ? manifest.target.sha256 : null)) {
      setStatus('Game data is on this device');
      navigator.storage?.persist?.().catch(() => {});
      await store.loadResident(onProgress);
      ready(store);
      return;
    }
  } catch (error) {
    showError(error);
    return;
  }

  const outdated = stored && stored.byo && manifest && stored.target !== manifest.target.sha256;
  showPicker(store, outdated ? ['PUMP FIGHTERS was updated', 'Pick your SF33RD.AFS once more to update it.'] : null);
}

function showPicker(store, message) {
  loader.classList.remove('hidden');
  barEl.hidden = true;
  playBtn.hidden = true;
  retryBtn.hidden = true;
  byoEl.hidden = false;
  loader.classList.toggle('warn', !!message);
  setStatus(...(message || ['Bring your own game file', '']));
  hintEl.textContent = '';

  pickBtn.onclick = () => {
    fileInput.value = '';
    fileInput.click();
  };
  fileInput.onchange = () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) patchAndStart(store, file);
  };
}

const patchProgress = { step: null, download: null };

function onPatchProgress({ phase, loaded = 0, total = 0 }) {
  if (phase === 'download') patchProgress.download = { loaded, total };
  else patchProgress.step = { phase, loaded, total };

  const step = patchProgress.step || { phase: 'check', loaded: 0, total: 1 };
  const share = step.total ? step.loaded / step.total : 0;
  const frac = step.phase === 'verify' ? 0.4 * share : step.phase === 'patch' ? 0.4 + 0.6 * share : 0;
  barEl.hidden = false;
  barFill.style.width = `${(frac * 100).toFixed(1)}%`;

  const d = patchProgress.download;
  const dl = d && d.loaded < d.total ? `patch download ${mb(d.loaded)} / ${mb(d.total)} MB` : '';
  const sizes = `${mb(step.loaded)} / ${mb(step.total)} MB${dl ? ` · ${dl}` : ''}`;
  if (step.phase === 'verify') setStatus('Checking your file…', sizes);
  else if (step.phase === 'patch') setStatus('Building PUMP FIGHTERS…', sizes);
  else setStatus('Checking your file…', dl);
  hintEl.textContent = 'Keep this page open. Your file is not uploaded anywhere.';
}

async function patchAndStart(store, file) {
  byoEl.hidden = true;
  loader.classList.remove('warn');
  patchProgress.step = null;
  patchProgress.download = null;
  setStatus('Checking your file…', file.name);
  keepAwake();

  try {
    await store.patchFromFile(file, patchUrl, onPatchProgress);
    navigator.storage?.persist?.().catch(() => {});
    await store.loadResident(onProgress);
  } catch (error) {
    console.error(error);
    const fileProblem = ['wrong-file', 'wrong-version', 'damaged-file', 'not-afs'].includes(error.code);
    showPicker(store, [fileProblem ? 'That file did not work' : 'Something went wrong', String(error.message || error)]);
    return;
  }

  ready(store);
}

// ---------------------------------------------------------------- wake lock, fullscreen, gestures

let wakeLock = null;

async function keepAwake() {
  try {
    if (navigator.wakeLock && (!wakeLock || wakeLock.released)) {
      wakeLock = await navigator.wakeLock.request('screen');
    }
  } catch (_) {
    // not allowed right now (battery saver, not visible)
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    controls.reset();
  } else if (started) {
    keepAwake();
  }
});

const fsBtn = $('btn-fullscreen');
const root = document.documentElement;
const canFullscreen = !!(document.fullscreenEnabled || document.webkitFullscreenEnabled);

if (!canFullscreen) fsBtn.hidden = true;

fsBtn.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement || document.webkitFullscreenElement) {
      await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    } else {
      await (root.requestFullscreen || root.webkitRequestFullscreen).call(root, { navigationUI: 'hide' });
    }
  } catch (error) {
    console.warn('[fullscreen]', error);
  }
  keepAwake();
});

for (const type of ['fullscreenchange', 'webkitfullscreenchange']) {
  document.addEventListener(type, () => setTimeout(() => window.dispatchEvent(new Event('resize')), 50));
}

// No scrolling, pinch zoom or double-tap zoom while playing (the settings sheet may scroll)
document.addEventListener(
  'touchmove',
  (e) => {
    if (!e.target.closest || !e.target.closest('#settings')) e.preventDefault();
  },
  { passive: false },
);
document.addEventListener('gesturestart', (e) => e.preventDefault());
document.addEventListener('dblclick', (e) => e.preventDefault());

// ---------------------------------------------------------------- settings sheet

const sheet = $('settings');
const optControls = $('opt-controls');
const optHaptics = $('opt-haptics');
const optSize = $('opt-size');

$('btn-settings').addEventListener('click', async () => {
  controls.reset();
  optControls.value = settings.controls;
  optHaptics.checked = !!settings.haptics;
  optSize.value = String(Math.round(settings.scale * 100));

  const total = window.pc?.store?.total || 0;
  $('data-info').textContent = total ? `Game data stored on this device: ${mb(total)} MB` : '';
  sheet.hidden = false;
});

$('btn-close').addEventListener('click', () => {
  sheet.hidden = true;
});

sheet.addEventListener('click', (e) => {
  if (e.target === sheet) sheet.hidden = true;
});

optControls.addEventListener('change', () => {
  settings.controls = optControls.value;
  saveSettings();
  applySettings();
});

optHaptics.addEventListener('change', () => {
  settings.haptics = optHaptics.checked;
  saveSettings();
  applySettings();
});

optSize.addEventListener('input', () => {
  settings.scale = Number(optSize.value) / 100;
  saveSettings();
  applySettings();
});

$('btn-redownload').addEventListener('click', async () => {
  if (!confirm('Delete the stored game data? You can make it again from your SF33RD.AFS.')) return;
  await AfsStore.clear();
  location.reload();
});

boot();
