// Turns the player's own SF33RD.AFS into the PUMP FIGHTERS archive, entirely on this device.
// Module worker: OPFS synchronous access handles (the one OPFS write API Safari, Chrome and Firefox
// all have) only exist in workers, and the patching work stays off the page's main thread.
// The player's file is only read (File.slice); nothing is uploaded. The patch itself (manifest +
// delta parts) is fetched from this site.
//
// in:  { file: File, manifestUrl, target: 'opfs' | FileSystemFileHandle, opfsName }
// out: { type: 'progress', phase: 'download'|'check'|'verify'|'patch', loaded, total }
//    | { type: 'done', size, target }        target = sha256 of the patched archive (manifest)
//    | { type: 'error', code, message }

import { PatchError, applyPatch, blobSource, checkManifest, createPartReader } from './afs-patch.js';

const MB = 1024 * 1024;

self.onmessage = async (event) => {
  const { file, manifestUrl, target, opfsName } = event.data;
  let finish = async () => {};
  let abort = async () => {};
  let lastPost = 0;
  const progress = (msg, force) => {
    const now = performance.now();
    if (force || now - lastPost > 100) {
      lastPost = now;
      self.postMessage({ type: 'progress', ...msg });
    }
  };

  try {
    const response = await fetch(manifestUrl, { cache: 'no-cache' });
    if (!response.ok) throw new PatchError('bad-patch', `Could not load the patch (HTTP ${response.status}).`);
    const manifest = checkManifest(await response.json());

    // Patch parts: downloaded as needed (the next one ahead), with progress
    const patchTotal = manifest.parts.reduce((n, p) => n + p.length, 0);
    let patchLoaded = 0;
    const reader = createPartReader(manifest, async (part) => {
      const res = await fetch(new URL(part.file, manifestUrl), { cache: 'no-cache' });
      if (!res.ok) throw new PatchError('bad-patch', `Could not download the patch (HTTP ${res.status}).`);
      const bytes = new Uint8Array(part.length);
      let at = 0;
      const body = res.body.getReader();
      for (;;) {
        const { done, value } = await body.read();
        if (done) break;
        if (at + value.length > bytes.length) throw new PatchError('bad-patch', `Patch file ${part.file} is larger than expected.`);
        bytes.set(value, at);
        at += value.length;
        patchLoaded += value.length;
        progress({ phase: 'download', loaded: patchLoaded, total: patchTotal });
      }
      progress({ phase: 'download', loaded: patchLoaded, total: patchTotal }, true);
      return at === bytes.length ? bytes : bytes.subarray(0, at);
    });
    reader.prefetch(0);

    // Where the patched archive goes
    let sink;
    if (target === 'opfs') {
      const root = await navigator.storage.getDirectory();
      const handle = await (await root.getFileHandle(opfsName, { create: true })).createSyncAccessHandle();
      handle.truncate(0);
      let pos = 0;
      let flushedAt = 0;
      sink = {
        write(chunk) {
          let done = 0;
          while (done < chunk.length) {
            const n = handle.write(chunk.subarray(done), { at: pos + done });
            if (!(n > 0)) throw new PatchError('io', 'Could not write the game data on this device.');
            done += n;
          }
          pos += chunk.length;
          if (pos - flushedAt >= 64 * MB) {
            handle.flush();
            flushedAt = pos;
          }
        },
      };
      finish = async () => {
        handle.flush();
        handle.close();
      };
      abort = async () => {
        try {
          handle.truncate(0);
        } finally {
          handle.close();
        }
      };
    } else {
      const writable = await target.createWritable();
      sink = { write: (chunk) => writable.write(chunk) };
      finish = () => writable.close();
      abort = () => writable.abort();
    }

    const result = await applyPatch({
      manifest,
      readDelta: reader.read,
      source: blobSource(file),
      sink,
      onProgress: (p) => progress(p, p.loaded === p.total),
    });
    await finish();
    finish = abort = async () => {};
    self.postMessage({ type: 'done', size: result.size, target: manifest.target.sha256 });
  } catch (error) {
    await abort().catch(() => {});
    const quota = error && (error.name === 'QuotaExceededError' || /quota/i.test(String(error.message)));
    self.postMessage({
      type: 'error',
      code: quota ? 'quota' : error.code || 'io',
      message: quota
        ? 'Not enough free storage on this device: the game needs about 660 MB.'
        : String(error && error.message ? error.message : error),
    });
  }
};
