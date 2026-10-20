// PSP / PPSSPP helper: patch the player's own SF33RD.AFS in the browser and save the modded file.
// - showSaveFilePicker (Chrome, Edge): the patcher worker streams straight into the chosen file.
// - elsewhere: the result goes to this browser's own storage (OPFS, the same copy the browser game
//   uses) and is saved from there with a download link; the file is read from disk, not from memory.

import { AfsStore, fetchPatchManifest, runPatchWorker } from './afs-store.js';

const $ = (id) => document.getElementById(id);
const patchUrl = document.querySelector('meta[name="pc-patch-url"]').content;
const pickBtn = $('btn-pick');
const fileInput = $('file');
const picked = $('picked');
const saveBtn = $('btn-save');
const saveHow = $('save-how');
const bar = $('bar');
const barFill = $('bar-fill');
const statusEl = $('status');
const detailEl = $('detail');
const downloadLink = $('btn-download');

const canSaveDirect = typeof window.showSaveFilePicker === 'function';
const mb = (bytes) => (bytes / 1e6).toFixed(bytes > 1e8 ? 0 : 1);
let manifest = null;
let file = null;
let busy = false;

saveHow.textContent = canSaveDirect
  ? 'You choose where to save it; the file is written as it is made.'
  : 'The file is made in this browser first (about 660 MB of storage), then you download it.';
saveBtn.textContent = canSaveDirect ? 'SAVE MODDED FILE' : 'MAKE MODDED FILE';

function setStatus(text, detail = '') {
  statusEl.textContent = text;
  detailEl.textContent = detail;
}

const progress = { step: null, download: null };
function onProgress({ phase, loaded = 0, total = 0 }) {
  if (phase === 'download') progress.download = { loaded, total };
  else progress.step = { phase, loaded, total };
  const step = progress.step || { phase: 'check', loaded: 0, total: 1 };
  const share = step.total ? step.loaded / step.total : 0;
  const frac = step.phase === 'verify' ? 0.4 * share : step.phase === 'patch' || step.phase === 'save' ? 0.4 + 0.6 * share : 0;
  bar.hidden = false;
  barFill.style.width = `${(frac * 100).toFixed(1)}%`;
  const d = progress.download;
  const dl = d && d.loaded < d.total ? ` · patch download ${mb(d.loaded)} / ${mb(d.total)} MB` : '';
  const label = { verify: 'Checking your file…', patch: 'Making the modded file…', save: 'Saving…' }[step.phase] || 'Checking your file…';
  setStatus(label, step.phase === 'check' ? dl.slice(3) : `${mb(step.loaded)} / ${mb(step.total)} MB${dl}`);
}

async function storedPatched() {
  const meta = AfsStore.storedMeta();
  if (!meta || !meta.byo || !meta.complete || !manifest || meta.target !== manifest.target.sha256) return null;
  const stored = await AfsStore.storedFile();
  return stored && stored.size === meta.size ? stored : null;
}

pickBtn.addEventListener('click', () => {
  if (busy) return;
  fileInput.value = '';
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  const chosen = fileInput.files && fileInput.files[0];
  if (!chosen) return;
  downloadLink.hidden = true;
  if (manifest && chosen.size !== manifest.source.size) {
    file = null;
    saveBtn.disabled = true;
    picked.textContent = `${chosen.name} is ${chosen.size.toLocaleString('en-US')} bytes, not the PS2 US v1.0 SF33RD.AFS (${manifest.source.size.toLocaleString('en-US')} bytes). Pick that file.`;
    picked.classList.add('bad');
    return;
  }
  file = chosen;
  picked.classList.remove('bad');
  picked.textContent = `${chosen.name} · ${mb(chosen.size)} MB`;
  saveBtn.disabled = false;
  setStatus('');
});

saveBtn.addEventListener('click', async () => {
  if (busy) return;
  progress.step = null;
  progress.download = null;

  try {
    if (canSaveDirect) {
      let handle;
      try {
        // Must be the first await: the picker needs the tap
        handle = await window.showSaveFilePicker({
          suggestedName: 'SF33RD.AFS',
          types: [{ description: 'AFS archive', accept: { 'application/octet-stream': ['.AFS'] } }],
        });
      } catch (error) {
        if (error.name === 'AbortError') return;
        throw error;
      }
      busy = true;
      saveBtn.disabled = pickBtn.disabled = true;
      if (file) {
        await runPatchWorker({ file, manifestUrl: patchUrl, target: handle, onProgress });
      } else {
        await copyStored(await storedPatched(), handle);
      }
      bar.hidden = true;
      setStatus(`Saved ${handle.name}`, 'Now put it in PSP/GAME/<folder>/resources/ next to the 3s-psp EBOOT (step 3).');
    } else {
      busy = true;
      saveBtn.disabled = pickBtn.disabled = true;
      if (file) {
        const store = new AfsStore();
        await store.patchFromFile(file, patchUrl, onProgress);
      }
      const stored = await storedPatched();
      if (!stored) throw new Error('The modded file was not stored. Try again.');
      offerDownload(stored);
    }
  } catch (error) {
    console.error(error);
    bar.hidden = true;
    setStatus('That did not work', String(error && error.message ? error.message : error));
    statusEl.classList.add('bad');
    busy = false;
    saveBtn.disabled = !file;
    pickBtn.disabled = false;
    return;
  }
  statusEl.classList.remove('bad');
  busy = false;
  pickBtn.disabled = false;
  saveBtn.disabled = !file;
});

/** Streams the stored (already patched) archive into a file the player chose. */
async function copyStored(stored, handle) {
  if (!stored) throw new Error('Pick your SF33RD.AFS first.');
  const writable = await handle.createWritable();
  let done = 0;
  try {
    await stored.stream().pipeThrough(
      new TransformStream({
        transform(chunk, controller) {
          done += chunk.byteLength;
          onProgress({ phase: 'save', loaded: done, total: stored.size });
          controller.enqueue(chunk);
        },
      }),
    ).pipeTo(writable);
  } catch (error) {
    await writable.abort().catch(() => {});
    throw error;
  }
}

function offerDownload(stored) {
  bar.hidden = true;
  if (downloadLink.href) URL.revokeObjectURL(downloadLink.href);
  downloadLink.href = URL.createObjectURL(stored);
  downloadLink.hidden = false;
  setStatus('The modded file is ready', 'Tap DOWNLOAD and keep the name SF33RD.AFS. It also stays here for the browser game.');
}

(async () => {
  manifest = await fetchPatchManifest(patchUrl);
  if (!manifest) {
    setStatus('The patch could not be loaded', 'Check your connection and reload the page.');
    pickBtn.disabled = true;
    return;
  }
  const stored = await storedPatched();
  if (stored) {
    picked.textContent = 'This browser already has the modded file (made earlier): you can save it without picking again.';
    saveBtn.disabled = false;
    if (!canSaveDirect) offerDownload(stored);
  }
})();
