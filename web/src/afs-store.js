// The game archive (SF33RD.AFS, ~640 MB) for the web build.
//
// 1. The player picks SF33RD.AFS from their own copy of the game. afs-patch-worker.js checks it and
//    patches it into this browser's own storage (the origin-private file system, OPFS). Nothing is
//    uploaded.
// 2. The stored file is a disk-backed File. Everything except the .adx music (~80 MB: sprites,
//    stages, sound effects, the AFS header and name table) is copied into one ArrayBuffer so the
//    game can read it synchronously. Music entries are read with File.slice() when a track starts.
// 3. The game reaches this object as Module.pcAfs (see src/platform/web/web_bridge.c in 3sx-web.patch).

const OPFS_NAME = 'SF33RD.AFS';
const META_KEY = 'pumpcombat.afs.meta';

function readMeta() {
  try {
    return JSON.parse(localStorage.getItem(META_KEY) || 'null');
  } catch (_) {
    return null;
  }
}

function writeMeta(meta) {
  try {
    localStorage.setItem(META_KEY, JSON.stringify(meta));
  } catch (_) {
    // private browsing: the patched copy is not remembered
  }
}

function hasOpfs() {
  return !!(navigator.storage && navigator.storage.getDirectory && typeof Worker !== 'undefined');
}

/** Runs afs-patch-worker.js: the player's file + the site's patch -> OPFS (target 'opfs') or a FileSystemFileHandle. */
export function runPatchWorker({ file, manifestUrl, target = 'opfs', onProgress = () => {} }) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./afs-patch-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'progress') {
        onProgress(msg);
      } else if (msg.type === 'done') {
        worker.terminate();
        resolve(msg);
      } else if (msg.type === 'error') {
        worker.terminate();
        const error = new Error(msg.message);
        error.name = 'PatchError';
        error.code = msg.code;
        reject(error);
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || 'The patcher could not start in this browser.'));
    };
    worker.postMessage({ file, manifestUrl: new URL(manifestUrl, location.href).href, target, opfsName: OPFS_NAME });
  });
}

/** The patch manifest (small JSON), or null when it cannot be fetched (offline). */
export async function fetchPatchManifest(url) {
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    return response.ok ? await response.json() : null;
  } catch (_) {
    return null;
  }
}

async function opfsFile() {
  const root = await navigator.storage.getDirectory();
  const handle = await root.getFileHandle(OPFS_NAME, { create: false });
  return handle.getFile();
}

export class AfsStore {
  constructor() {
    this.file = null; // disk-backed Blob/File of the whole archive
    this.total = 0;
    this.resident = null; // Uint8Array with the resident ranges, back to back
    this.segStart = null; // sorted archive offsets of resident ranges
    this.segEnd = null;
    this.segAt = null; // where each range starts inside `resident`
    this.fetches = new Map(); // token -> { done, data, error }
    this.entryCount = 0;
    this.names = [];
  }

  // ---- 1. stored archive ------------------------------------------------------------------

  /** Deletes the stored archive (settings: "Delete stored game data"). */
  static async clear() {
    localStorage.removeItem(META_KEY);
    try {
      const root = await navigator.storage.getDirectory();
      await root.removeEntry(OPFS_NAME);
    } catch (_) {
      // nothing stored
    }
  }

  static hasOpfs() {
    return hasOpfs();
  }

  /** What is stored: null or { byo: true, target, size, complete }. */
  static storedMeta() {
    return readMeta();
  }

  /**
   * Opens a patched archive made earlier from the player's own file. With `target` (the current
   * patch's result sha256), an archive made by an older patch is not used.
   */
  async openPatched(target) {
    const meta = readMeta();
    if (!meta || !meta.byo || !meta.complete || !hasOpfs()) return false;
    if (target && meta.target !== target) return false;
    try {
      const file = await opfsFile();
      if (file.size !== meta.size) return false;
      await this._useFile(file);
      return true;
    } catch (_) {
      return false;
    }
  }

  /** Patches the player's SF33RD.AFS into OPFS and opens the result. */
  async patchFromFile(file, manifestUrl, onProgress) {
    if (!hasOpfs()) throw new Error('This browser cannot store the game. In a private window, try a normal one.');
    try {
      const { quota, usage } = await navigator.storage.estimate();
      const stored = readMeta();
      const reusable = stored && stored.size ? stored.size : 0; // the old copy is overwritten
      if (quota && quota - usage + reusable < 700e6) {
        throw new Error(`Not enough free storage on this device: the game needs about 660 MB (${Math.max(0, Math.round((quota - usage) / 1e6))} MB free for this site).`);
      }
    } catch (error) {
      if (/storage/.test(error.message)) throw error;
    }
    try {
      localStorage.removeItem(META_KEY); // the stored copy is being replaced
    } catch (_) {
      // private browsing
    }
    const result = await runPatchWorker({ file, manifestUrl, target: 'opfs', onProgress });
    writeMeta({ url: `byo:${result.target}`, byo: true, target: result.target, size: result.size, complete: true });
    return this._useFile(await opfsFile());
  }

  /** The stored archive as a disk-backed File (for saving it), or null. */
  static async storedFile() {
    try {
      return await opfsFile();
    } catch (_) {
      return null;
    }
  }

  // ---- 2. resident copy --------------------------------------------------------------------

  async _useFile(file) {
    this.file = file;
    this.total = file.size;
    return this;
  }

  async _slice(start, end) {
    let lastError;

    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return new Uint8Array(await this.file.slice(start, end).arrayBuffer());
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError;
  }

  /** Reads the header and copies every non-music entry into memory. */
  async loadResident(onProgress) {
    const head = await this._slice(0, 8);
    const view = new DataView(head.buffer);

    if (view.getUint32(0, false) !== 0x41465300) throw new Error('Not an AFS archive');

    const count = view.getUint32(4, true);
    const toc = new DataView((await this._slice(8, 8 + count * 8 + 8)).buffer);
    const entries = [];
    let firstOffset = Infinity;
    let dataEnd = 0;

    for (let i = 0; i < count; i++) {
      const offset = toc.getUint32(i * 8, true);
      const size = toc.getUint32(i * 8 + 4, true);
      entries.push({ offset, size });

      if (offset !== 0) {
        firstOffset = Math.min(firstOffset, offset);
        dataEnd = Math.max(dataEnd, offset + size);
      }
    }

    let attrOffset = toc.getUint32(count * 8, true);
    let attrSize = toc.getUint32(count * 8 + 4, true);

    if (!attrOffset || attrOffset + attrSize > this.total) {
      // Older layout: the pointer sits just before the first entry
      const alt = new DataView((await this._slice(firstOffset - 8, firstOffset)).buffer);
      attrOffset = alt.getUint32(0, true);
      attrSize = alt.getUint32(4, true);
    }

    const ranges = [[0, Math.min(firstOffset, this.total)]];
    const names = new Array(count).fill('');

    if (attrOffset && attrSize && attrOffset + attrSize <= this.total) {
      const attrs = await this._slice(attrOffset, attrOffset + attrSize);
      const decoder = new TextDecoder('latin1');

      for (let i = 0; i < count && i * 48 + 32 <= attrs.length; i++) {
        const raw = attrs.subarray(i * 48, i * 48 + 32);
        const zero = raw.indexOf(0);
        names[i] = decoder.decode(zero >= 0 ? raw.subarray(0, zero) : raw);
      }

      ranges.push([attrOffset, attrOffset + attrSize]);
    }

    let streamedBytes = 0;

    entries.forEach((entry, i) => {
      if (!entry.offset || !entry.size) return;

      if (/\.adx$/i.test(names[i])) {
        streamedBytes += entry.size;
      } else {
        ranges.push([entry.offset, entry.offset + entry.size]);
      }
    });

    // Merge touching ranges (entries are padded to 2 KB sectors, so allow small gaps)
    ranges.sort((a, b) => a[0] - b[0]);
    const merged = [];

    for (const [start, end] of ranges) {
      const last = merged[merged.length - 1];

      if (last && start <= last[1] + 4096) {
        last[1] = Math.max(last[1], end);
      } else {
        merged.push([start, end]);
      }
    }

    const residentBytes = merged.reduce((sum, [s, e]) => sum + (e - s), 0);
    this.resident = new Uint8Array(residentBytes);
    this.segStart = new Float64Array(merged.length);
    this.segEnd = new Float64Array(merged.length);
    this.segAt = new Float64Array(merged.length);

    let at = 0;
    merged.forEach(([start, end], i) => {
      this.segStart[i] = start;
      this.segEnd[i] = end;
      this.segAt[i] = at;
      at += end - start;
    });

    // Read in ~4 MB pieces, a few at a time
    const jobs = [];
    merged.forEach(([start, end], i) => {
      for (let s = start; s < end; s += 4 * 1024 * 1024) {
        jobs.push({ start: s, end: Math.min(end, s + 4 * 1024 * 1024), dest: this.segAt[i] + (s - start) });
      }
    });

    let done = 0;
    let next = 0;
    const worker = async () => {
      while (next < jobs.length) {
        const job = jobs[next++];
        this.resident.set(await this._slice(job.start, job.end), job.dest);
        done += job.end - job.start;
        onProgress({ phase: 'unpack', loaded: done, total: residentBytes });
      }
    };

    await Promise.all([worker(), worker(), worker(), worker()]);

    this.entryCount = count;
    this.names = names;
    console.info(
      `[afs] ${count} entries, ${(residentBytes / 1e6).toFixed(1)} MB in memory (${merged.length} ranges), ` +
        `${(streamedBytes / 1e6).toFixed(1)} MB music read on demand`,
    );
    return { residentBytes, streamedBytes, ranges: merged.length };
  }

  // ---- 3. game-facing API (called from wasm) ---------------------------------------------

  size() {
    return this.total;
  }

  _segmentFor(offset) {
    let lo = 0;
    let hi = this.segStart.length - 1;

    while (lo <= hi) {
      const mid = (lo + hi) >> 1;

      if (this.segStart[mid] <= offset) {
        if (offset < this.segEnd[mid]) return mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    return -1;
  }

  isResident(offset, len) {
    const seg = this._segmentFor(offset);
    return seg >= 0 && offset + len <= this.segEnd[seg];
  }

  /** Copies a resident range into wasm memory. Returns bytes copied, or -1. */
  readSync(offset, len, heap, ptr) {
    const seg = this._segmentFor(offset);
    if (seg < 0) return len === 0 ? 0 : -1;

    const count = Math.min(len, this.segEnd[seg] - offset);
    const from = this.segAt[seg] + (offset - this.segStart[seg]);
    heap.set(this.resident.subarray(from, from + count), ptr);
    return count;
  }

  fetchStart(token, offset, len) {
    const job = { done: false, data: null, error: null };
    this.fetches.set(token, job);

    this._slice(offset, offset + len).then(
      (data) => {
        job.data = data;
        job.done = true;
      },
      (error) => {
        console.error('[afs] read failed', error);
        job.error = error;
        job.done = true;
      },
    );
  }

  fetchPoll(token, heap, ptr, len) {
    const job = this.fetches.get(token);
    if (!job) return -1;
    if (!job.done) return 0;

    this.fetches.delete(token);
    if (job.error || job.data.length < len) return -1;

    heap.set(job.data.subarray(0, len), ptr);
    return 1;
  }

  fetchCancel(token) {
    this.fetches.delete(token);
  }
}
