// PUMP FIGHTERS patcher: turns the player's own SF33RD.AFS (Street Fighter III 3rd Strike, PS2 US v1.0)
// into the PUMP FIGHTERS archive. One module for the browser (File/Blob, in a worker) and Node
// (patcher/patch.mjs). No dependencies.
//
// The patch (patch/pumpfighters.json + pumpfighters-N.deltas) holds only:
//   - a JSON manifest: hashes and sizes of the source/target archive and of each changed entry;
//   - one VCDIFF delta (RFC 3284, the xdelta3 wire format) per changed entry, whose source segment
//     is that entry of the player's original archive, packed back to back into the .deltas parts.
// Every byte not produced by a delta is read from the player's file. The archive is rebuilt exactly
// like sf3mod/tools/afs.py `repack`: same header, entries in index order, 2048-byte sector padding,
// the 48-byte name table with the size field (+44) of every written entry set to its new size,
// the original trailer, and the table-of-contents pointers updated.
//
// Streaming: the output is produced front to back through `sink.write(chunk)`, never held whole in
// memory (largest buffer: one changed entry, ~4 MB). Source reads go through `source.read(off, len)`.

export const PATCH_FORMAT = 'pumpfighters-afs-patch';
export const PATCH_VERSION = 1;
export const SECTOR = 2048;

const CHUNK = 4 * 1024 * 1024;

export class PatchError extends Error {
  /**
   * @param {'bad-patch'|'not-afs'|'wrong-file'|'wrong-version'|'damaged-file'|'delta-failed'|'target-mismatch'|'io'} code
   * @param {string} message
   */
  constructor(code, message) {
    super(message);
    this.name = 'PatchError';
    this.code = code;
  }
}

export const align = (n, a = SECTOR) => Math.ceil(n / a) * a;

// ---------------------------------------------------------------------------------------------
// SHA-256 (incremental; WebCrypto has no streaming digest)

const K256 = new Int32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export class Sha256 {
  constructor() {
    this.h = new Int32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    this.w = new Int32Array(64);
    this.buf = new Uint8Array(64);
    this.bufLen = 0;
    this.bytes = 0;
  }

  /** @param {Uint8Array} data */
  update(data) {
    const len = data.length;
    let pos = 0;
    this.bytes += len;

    if (this.bufLen) {
      const take = Math.min(64 - this.bufLen, len);
      this.buf.set(data.subarray(0, take), this.bufLen);
      this.bufLen += take;
      pos = take;
      if (this.bufLen < 64) return this;
      this._blocks(this.buf, 0, 64);
      this.bufLen = 0;
    }

    const whole = pos + ((len - pos) & ~63);
    if (whole > pos) this._blocks(data, pos, whole);
    if (whole < len) {
      this.buf.set(data.subarray(whole), 0);
      this.bufLen = len - whole;
    }
    return this;
  }

  _blocks(d, p, end) {
    const w = this.w;
    const H = this.h;
    let h0 = H[0], h1 = H[1], h2 = H[2], h3 = H[3], h4 = H[4], h5 = H[5], h6 = H[6], h7 = H[7];

    for (; p < end; p += 64) {
      for (let i = 0, q = p; i < 16; i++, q += 4) w[i] = (d[q] << 24) | (d[q + 1] << 16) | (d[q + 2] << 8) | d[q + 3];
      for (let i = 16; i < 64; i++) {
        const x = w[i - 15];
        const y = w[i - 2];
        const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
        const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
      }

      let a = h0, b = h1, c = h2, dd = h3, e = h4, f = h5, g = h6, h = h7;
      for (let i = 0; i < 64; i++) {
        const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
        const t1 = (h + S1 + ((e & f) ^ (~e & g)) + K256[i] + w[i]) | 0;
        const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
        const t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) | 0;
        h = g; g = f; f = e; e = (dd + t1) | 0; dd = c; c = b; b = a; a = (t1 + t2) | 0;
      }

      h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + dd) | 0;
      h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
    }

    H[0] = h0; H[1] = h1; H[2] = h2; H[3] = h3; H[4] = h4; H[5] = h5; H[6] = h6; H[7] = h7;
  }

  /** @returns {string} lowercase hex digest (the object is finished afterwards) */
  hex() {
    const bits = this.bytes * 8;
    const tail = new Uint8Array(this.bufLen < 56 ? 64 : 128);
    tail.set(this.buf.subarray(0, this.bufLen));
    tail[this.bufLen] = 0x80;
    const dv = new DataView(tail.buffer);
    dv.setUint32(tail.length - 8, Math.floor(bits / 0x100000000));
    dv.setUint32(tail.length - 4, bits >>> 0);
    this._blocks(tail, 0, tail.length);
    let out = '';
    for (const v of this.h) out += (v >>> 0).toString(16).padStart(8, '0');
    return out;
  }
}

/** Pure-JS incremental SHA-256 (fallback where WebCrypto is missing). */
export const jsHash = () => new Sha256();

// ---------------------------------------------------------------------------------------------
// VCDIFF (RFC 3284) decoder: default code table, VCD_SOURCE / VCD_TARGET windows, the xdelta3
// VCD_ADLER32 window checksum and application header. Secondary compression and custom code
// tables are refused (our patches never use them; xdelta3 needs `-S none` to match).

const NOOP = 0, ADD = 1, RUN = 2, COPY = 3;

/** The RFC 3284 section 5.6 default instruction code table: [inst1, size1, mode1, inst2, size2, mode2] */
export function defaultCodeTable() {
  const t = [[RUN, 0, 0, NOOP, 0, 0], [ADD, 0, 0, NOOP, 0, 0]];
  for (let s = 1; s <= 17; s++) t.push([ADD, s, 0, NOOP, 0, 0]);
  for (let m = 0; m < 9; m++) {
    t.push([COPY, 0, m, NOOP, 0, 0]);
    for (let s = 4; s <= 18; s++) t.push([COPY, s, m, NOOP, 0, 0]);
  }
  for (let m = 0; m < 6; m++) for (let a = 1; a <= 4; a++) for (let c = 4; c <= 6; c++) t.push([ADD, a, 0, COPY, c, m]);
  for (let m = 6; m < 9; m++) for (let a = 1; a <= 4; a++) t.push([ADD, a, 0, COPY, 4, m]);
  for (let m = 0; m < 9; m++) t.push([COPY, 4, m, ADD, 1, 0]);
  return t;
}

const CODE_TABLE = defaultCodeTable();
const S_NEAR = 4;
const S_SAME = 3;
const MAX_WINDOW = 64 * 1024 * 1024;

function adler32(bytes) {
  let a = 1, b = 0;
  for (let i = 0; i < bytes.length; ) {
    const end = Math.min(bytes.length, i + 3800);
    for (; i < end; i++) {
      a += bytes[i];
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

/**
 * @param {Uint8Array} delta  a whole VCDIFF file
 * @param {Uint8Array} source the source the delta was made against
 * @param {{expectedSize?: number}} [opts]
 * @returns {Uint8Array}
 */
export function vcdiffDecode(delta, source, opts = {}) {
  const fail = (msg) => {
    throw new PatchError('delta-failed', `VCDIFF: ${msg}`);
  };
  let p = 0;
  const byte = () => (p < delta.length ? delta[p++] : fail('truncated'));
  const varint = () => {
    let v = 0;
    for (let i = 0; i < 9; i++) {
      const b = byte();
      v = v * 128 + (b & 0x7f);
      if (!(b & 0x80)) {
        if (v > Number.MAX_SAFE_INTEGER) fail('integer too large');
        return v;
      }
    }
    return fail('integer too long');
  };

  if (byte() !== 0xd6 || byte() !== 0xc3 || byte() !== 0xc4) fail('not a VCDIFF file');
  if (byte() !== 0) fail('unsupported VCDIFF version');
  const hdr = byte();
  if (hdr & 0x01) fail('secondary compression is not supported (encode with xdelta3 -S none)');
  if (hdr & 0x02) fail('custom code tables are not supported');
  if (hdr & ~0x07) fail('unknown header flags');
  if (hdr & 0x04) {
    const appHeader = varint(); // xdelta3 application header (file names): skipped
    p += appHeader;
  }

  let out = new Uint8Array(opts.expectedSize || 0);
  let outLen = 0;
  const ensure = (n) => {
    if (outLen + n <= out.length) return;
    const grown = new Uint8Array(Math.max(outLen + n, out.length * 2));
    grown.set(out.subarray(0, outLen));
    out = grown;
  };

  while (p < delta.length) {
    const win = byte();
    if (win & ~0x07) fail('unknown window flags');
    if ((win & 0x03) === 0x03) fail('window with both source and target segment');

    let seg = null;
    if (win & 0x03) {
      const segLen = varint();
      const segPos = varint();
      const from = win & 0x01 ? source : out.subarray(0, outLen);
      if (segPos + segLen > from.length) fail('source segment outside the source');
      seg = from.subarray(segPos, segPos + segLen);
    }
    const segLen = seg ? seg.length : 0;

    const deltaLen = varint();
    const deltaEnd = p + deltaLen;
    const targetLen = varint();
    if (targetLen > MAX_WINDOW) fail('target window too large');
    if (byte() !== 0) fail('compressed delta sections are not supported (encode with xdelta3 -S none)');
    const dataLen = varint();
    const instLen = varint();
    const addrLen = varint();
    let checksum = null;
    if (win & 0x04) checksum = ((byte() << 24) | (byte() << 16) | (byte() << 8) | byte()) >>> 0;

    let dp = p;
    const dataEnd = dp + dataLen;
    let ip = dataEnd;
    const instEnd = ip + instLen;
    let ap = instEnd;
    const addrEnd = ap + addrLen;
    if (addrEnd !== deltaEnd || addrEnd > delta.length) fail('window sections do not add up');

    ensure(targetLen);
    const t = out.subarray(outLen, outLen + targetLen);
    let tp = 0;
    const near = [0, 0, 0, 0];
    let nextSlot = 0;
    const same = new Float64Array(S_SAME * 256);

    const instVarint = () => {
      let v = 0;
      for (let i = 0; i < 9; i++) {
        if (ip >= instEnd) fail('instruction section truncated');
        const b = delta[ip++];
        v = v * 128 + (b & 0x7f);
        if (!(b & 0x80)) return v;
      }
      return fail('integer too long');
    };
    const addrVarint = () => {
      let v = 0;
      for (let i = 0; i < 9; i++) {
        if (ap >= addrEnd) fail('address section truncated');
        const b = delta[ap++];
        v = v * 128 + (b & 0x7f);
        if (!(b & 0x80)) return v;
      }
      return fail('integer too long');
    };

    const exec = (inst, size, mode) => {
      if (size === 0) size = instVarint();
      if (tp + size > targetLen) fail('instruction writes past the window');

      if (inst === ADD) {
        if (dp + size > dataEnd) fail('data section truncated');
        t.set(delta.subarray(dp, dp + size), tp);
        dp += size;
      } else if (inst === RUN) {
        if (dp >= dataEnd) fail('data section truncated');
        t.fill(delta[dp++], tp, tp + size);
      } else {
        const here = segLen + tp;
        let addr;
        if (mode === 0) addr = addrVarint();
        else if (mode === 1) addr = here - addrVarint();
        else if (mode < 2 + S_NEAR) addr = near[mode - 2] + addrVarint();
        else {
          if (ap >= addrEnd) fail('address section truncated');
          addr = same[(mode - 2 - S_NEAR) * 256 + delta[ap++]];
        }
        if (!(addr >= 0 && addr < here)) fail('bad copy address');
        near[nextSlot] = addr;
        nextSlot = (nextSlot + 1) % S_NEAR;
        same[addr % (S_SAME * 256)] = addr;

        if (addr + size <= segLen) {
          t.set(seg.subarray(addr, addr + size), tp);
        } else if (addr >= segLen && addr - segLen + size <= tp) {
          t.copyWithin(tp, addr - segLen, addr - segLen + size);
        } else {
          for (let k = 0; k < size; k++) {
            const a = addr + k;
            t[tp + k] = a < segLen ? seg[a] : t[a - segLen];
          }
        }
      }
      tp += size;
    };

    while (ip < instEnd) {
      const e = CODE_TABLE[delta[ip++]];
      if (e[0] !== NOOP) exec(e[0], e[1], e[2]);
      if (e[3] !== NOOP) exec(e[3], e[4], e[5]);
    }

    if (tp !== targetLen) fail('window is shorter than declared');
    if (dp !== dataEnd || ap !== addrEnd) fail('unused delta bytes');
    if (checksum !== null && adler32(t) !== checksum) fail('window checksum mismatch');
    outLen += targetLen;
    p = deltaEnd;
  }

  return outLen === out.length ? out : out.slice(0, outLen);
}

// ---------------------------------------------------------------------------------------------
// AFS archives

const u32 = (b, o) => (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
const putU32 = (b, o, v) => {
  b[o] = v & 0xff;
  b[o + 1] = (v >>> 8) & 0xff;
  b[o + 2] = (v >>> 16) & 0xff;
  b[o + 3] = (v >>> 24) & 0xff;
};

/**
 * Reads the table of contents, like afs.py's AFS class.
 * @param {{size: number, read: (offset: number, length: number) => Promise<Uint8Array>}} src
 */
export async function readAfs(src) {
  const total = src.size;
  if (total < 16) throw new PatchError('not-afs', 'This file is too small to be SF33RD.AFS.');
  const head = await src.read(0, 8);
  if (head[0] !== 0x41 || head[1] !== 0x46 || head[2] !== 0x53 || head[3] !== 0) {
    throw new PatchError('not-afs', 'This is not an AFS archive. Pick SF33RD.AFS from the game disc.');
  }
  const count = u32(head, 4);
  if (!(count > 0 && count <= (total - 16) / 8)) throw new PatchError('not-afs', 'Invalid AFS header.');

  const toc = await src.read(8, count * 8);
  const entries = [];
  let first = Infinity;
  let end = 0;
  for (let i = 0; i < count; i++) {
    const offset = u32(toc, i * 8);
    const size = u32(toc, i * 8 + 4);
    if (offset + size > total) throw new PatchError('not-afs', 'AFS entry outside the archive.');
    entries.push({ index: i, offset, size, name: '' });
    if (offset) first = Math.min(first, offset);
    end = Math.max(end, offset + size);
  }
  if (!Number.isFinite(first)) throw new PatchError('not-afs', 'AFS has no entries.');

  const pointerPositions = [];
  let nameOffset = 0;
  let nameSize = 0;
  for (const pos of [...new Set([8 + count * 8, first - 8])]) {
    const ptr = await src.read(pos, 8);
    const o = u32(ptr, 0);
    const s = u32(ptr, 4);
    if (o >= end && s >= count * 48 && o + s <= total) {
      pointerPositions.push(pos);
      nameOffset = o;
      nameSize = s;
    }
  }
  if (!pointerPositions.length) throw new PatchError('not-afs', 'AFS has no 48-byte name table.');

  const names = await src.read(nameOffset, nameSize);
  for (const e of entries) {
    const raw = names.subarray(e.index * 48, e.index * 48 + 32);
    const zero = raw.indexOf(0);
    e.name = String.fromCharCode(...(zero >= 0 ? raw.subarray(0, zero) : raw));
  }

  return { total, count, entries, first, pointerPositions, nameOffset, nameSize, tocBytes: toc };
}

/**
 * Where everything goes in the patched archive, exactly like afs.py repack.
 * @param {Awaited<ReturnType<typeof readAfs>>} afs
 * @param {Map<number, number>} newSizes changed entry index -> new size
 */
export function planTarget(afs, newSizes) {
  if (newSizes.size === 0) return { mode: 'copy', size: afs.total };

  // Same-size changes: copy the archive and overwrite the entries in place (padding kept)
  if ([...newSizes].every(([i, size]) => afs.entries[i].size === size)) {
    return { mode: 'in-place', size: afs.total };
  }

  let pos = afs.first;
  const placed = afs.entries.map((e) => {
    if (e.offset === 0 && !newSizes.has(e.index)) return { offset: 0, size: e.size, written: false };
    pos = align(pos);
    const size = newSizes.has(e.index) ? newSizes.get(e.index) : e.size;
    const at = { offset: pos, size, written: true };
    pos += size;
    return at;
  });
  const nameOffset = align(pos);
  const trailer = afs.total - (afs.nameOffset + afs.nameSize);
  const end = nameOffset + afs.nameSize + trailer;
  return { mode: 'repack', placed, nameOffset, trailer, size: align(end) };
}

// ---------------------------------------------------------------------------------------------
// Patch files: <name>.json (manifest) + <name>-N.deltas parts. A part is the deltas of a run of
// entries back to back (index order), small enough for any static host (< 25 MiB).

/** Validates a parsed manifest. */
export function checkManifest(manifest) {
  const bad = (msg) => {
    throw new PatchError('bad-patch', `Patch is damaged: ${msg}`);
  };
  if (!manifest || manifest.format !== PATCH_FORMAT) bad('not a PUMP FIGHTERS patch manifest');
  if (manifest.version !== PATCH_VERSION) bad(`unsupported patch version ${manifest.version}`);
  if (!Array.isArray(manifest.parts) || !Array.isArray(manifest.entries)) bad('manifest incomplete');
  if (manifest.hashChunk !== HASH_CHUNK || !Array.isArray(manifest.source?.chunks) || !Array.isArray(manifest.target?.chunks)) {
    bad('manifest has no chunk checksums');
  }
  for (const e of manifest.entries) {
    const part = manifest.parts[e.delta.part];
    if (!part || e.delta.offset + e.delta.length > part.length) bad(`delta for ${e.name} out of range`);
  }
  return manifest;
}

/**
 * Hands out entry deltas from the part files, loading each part once (and the next one ahead),
 * keeping at most two in memory.
 * @param {object} manifest
 * @param {(part: {file: string, length: number, sha256: string}, index: number) => Promise<Uint8Array>} loadPart
 * @param {(bytes: Uint8Array) => Promise<string>} [digest]
 */
export function createPartReader(manifest, loadPart, digest = defaultDigest) {
  const loaded = new Map(); // part index -> Promise<Uint8Array>
  const get = (i) => {
    if (i >= manifest.parts.length) return null;
    if (!loaded.has(i)) {
      const part = manifest.parts[i];
      const p = loadPart(part, i).then(async (bytes) => {
        if (bytes.length !== part.length || (await digest(bytes)) !== part.sha256) {
          throw new PatchError('bad-patch', `Patch file ${part.file} is damaged or incomplete. Reload and try again.`);
        }
        return bytes;
      });
      p.catch(() => {}); // reported when awaited
      loaded.set(i, p);
    }
    return loaded.get(i);
  };
  return {
    prefetch: (i = 0) => void get(i),
    async read(entry) {
      const i = entry.delta.part;
      for (const k of loaded.keys()) if (k < i) loaded.delete(k);
      const bytes = await get(i);
      get(i + 1);
      return bytes.subarray(entry.delta.offset, entry.delta.offset + entry.delta.length);
    },
  };
}

// ---------------------------------------------------------------------------------------------
// Hashing. Whole archives are checked as a list of SHA-256 digests of 8 MiB chunks (manifest
// source.chunks / target.chunks), which browsers compute natively with WebCrypto. Node callers
// can also pass createHash to check the whole-file SHA-256 (manifest source.sha256 / target.sha256).

export const HASH_CHUNK = 8 * 1024 * 1024;

const toHex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');

/** SHA-256 hex of a byte array: WebCrypto when available (browsers, Node), else pure JS. */
export const defaultDigest =
  globalThis.crypto && globalThis.crypto.subtle
    ? async (bytes) => toHex(await globalThis.crypto.subtle.digest('SHA-256', bytes))
    : async (bytes) => new Sha256().update(bytes).hex();

/** Per-chunk digests of a stream of bytes; fails at the first chunk that differs from `expected`. */
class ChunkCheck {
  constructor(expected, digest, fail) {
    this.expected = expected;
    this.digest = digest;
    this.fail = fail;
    this.buf = new Uint8Array(HASH_CHUNK);
    this.len = 0;
    this.index = 0;
  }

  async update(bytes) {
    let at = 0;
    while (at < bytes.length) {
      if (this.len === 0 && bytes.length - at >= HASH_CHUNK) {
        await this._check(bytes.subarray(at, at + HASH_CHUNK));
        at += HASH_CHUNK;
        continue;
      }
      const take = Math.min(HASH_CHUNK - this.len, bytes.length - at);
      this.buf.set(bytes.subarray(at, at + take), this.len);
      this.len += take;
      at += take;
      if (this.len === HASH_CHUNK) {
        await this._check(this.buf);
        this.len = 0;
      }
    }
  }

  async _check(chunk) {
    const i = this.index++;
    if (i >= this.expected.length || (await this.digest(chunk)) !== this.expected[i]) this.fail(i);
  }

  async finish() {
    if (this.len) await this._check(this.buf.subarray(0, this.len));
    if (this.index !== this.expected.length) this.fail(this.index);
  }
}

/** Chunk digests of a whole source (used by the patch generator). */
export async function chunkDigests(src, digest = defaultDigest) {
  const out = [];
  for (let at = 0; at < src.size; at += HASH_CHUNK) out.push(await digest(await src.read(at, Math.min(HASH_CHUNK, src.size - at))));
  return out;
}

// ---------------------------------------------------------------------------------------------
// Sources and helpers

/** A File/Blob as a patch source (browser). */
export function blobSource(blob) {
  return {
    size: blob.size,
    read: async (offset, length) => new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer()),
  };
}

/** Reads [start, end) in `size` pieces, one read ahead, calling fn(chunk) in order. */
async function eachChunk(src, start, end, fn, size = CHUNK) {
  let next = start < end ? src.read(start, Math.min(size, end - start)) : null;
  for (let at = start; at < end; ) {
    const len = Math.min(size, end - at);
    const chunk = await next;
    if (chunk.length !== len) throw new PatchError('io', 'Could not read the whole file (was it moved or changed?)');
    at += len;
    next = at < end ? src.read(at, Math.min(size, end - at)) : null;
    await fn(chunk);
  }
}

const notOriginal = () =>
  new PatchError(
    'damaged-file',
    'This SF33RD.AFS does not match the original PS2 US v1.0 file (it may be damaged or already modified). Copy it from the disc again.',
  );

/**
 * Checks the whole source against the manifest: every 8 MiB chunk digest, plus the whole-file
 * SHA-256 when createHash is given.
 */
export async function verifySourceFile(manifest, src, { digest = defaultDigest, createHash = null, onProgress = () => {} } = {}) {
  const check = new ChunkCheck(manifest.source.chunks, digest, () => {
    throw notOriginal();
  });
  const full = createHash ? createHash() : null;
  let done = 0;
  await eachChunk(
    src,
    0,
    src.size,
    async (chunk) => {
      if (full) full.update(chunk);
      await check.update(chunk);
      done += chunk.length;
      onProgress({ phase: 'verify', loaded: done, total: src.size });
    },
    HASH_CHUNK,
  );
  await check.finish();
  if (full && full.hex() !== manifest.source.sha256) throw notOriginal();
}

/**
 * Quick check before any heavy work: right size and the exact table of contents.
 * Throws a PatchError with a player-facing message.
 */
export async function checkSource(manifest, src, { digest = defaultDigest } = {}) {
  if (src.size !== manifest.source.size) {
    throw new PatchError(
      'wrong-file',
      `This file is ${src.size.toLocaleString('en-US')} bytes; SF33RD.AFS from Street Fighter III 3rd Strike ` +
        `(PS2, US v1.0) is ${manifest.source.size.toLocaleString('en-US')} bytes. Pick that file.`,
    );
  }
  const afs = await readAfs(src);
  if (afs.count !== manifest.source.entries || (await digest(afs.tocBytes)) !== manifest.source.tocSha256) {
    throw new PatchError(
      'wrong-version',
      'This SF33RD.AFS is a different version of the game. The patch needs the PS2 US v1.0 file.',
    );
  }
  return afs;
}

// ---------------------------------------------------------------------------------------------
// Apply

/**
 * @param {object} o
 * @param {object} o.manifest                      the parsed pumpfighters.json
 * @param {(entry: object) => Promise<Uint8Array>} o.readDelta  delta bytes of one manifest entry (see createPartReader)
 * @param {{size: number, read: (off: number, len: number) => Promise<Uint8Array>}} o.source  the player's SF33RD.AFS
 * @param {{write: (chunk: Uint8Array) => (void|Promise<void>)}} o.sink  receives the new archive front to back
 * @param {(p: {phase: 'check'|'verify'|'patch', loaded: number, total: number}) => void} [o.onProgress]
 * @param {(bytes: Uint8Array) => Promise<string>} [o.digest]  one-shot SHA-256 hex (default WebCrypto)
 * @param {() => {update(b: Uint8Array): any, hex(): string}} [o.createHash]  also check whole-file SHA-256s
 * @param {boolean} [o.verifySource] check the whole source before writing (default true)
 * @returns {Promise<{size: number, sha256: string|null}>}
 */
export async function applyPatch({
  manifest,
  readDelta,
  source,
  sink,
  onProgress = () => {},
  digest = defaultDigest,
  createHash = null,
  verifySource = true,
}) {
  checkManifest(manifest);

  onProgress({ phase: 'check', loaded: 0, total: 1 });
  const afs = await checkSource(manifest, source, { digest });
  if (verifySource) await verifySourceFile(manifest, source, { digest, createHash, onProgress });

  const changes = new Map(manifest.entries.map((e) => [e.index, e]));
  for (const e of manifest.entries) {
    const src = afs.entries[e.index];
    if (!src || src.name !== e.name || src.size !== e.source.size) {
      throw new PatchError('wrong-version', `Entry ${e.index} (${e.name}) does not match the patch.`);
    }
  }
  const plan = planTarget(afs, new Map(manifest.entries.map((e) => [e.index, e.target.size])));
  if (plan.size !== manifest.target.size) throw new PatchError('bad-patch', 'Patch layout does not match its manifest.');

  // Output with running checks and progress
  const badOutput = () => {
    throw new PatchError('target-mismatch', 'The patched file did not come out right (checksum). Your original file was not changed.');
  };
  const outCheck = new ChunkCheck(manifest.target.chunks, digest, badOutput);
  const outFull = createHash ? createHash() : null;
  let written = 0;
  let lastReport = 0;
  const write = async (chunk) => {
    if (!chunk.length) return;
    if (outFull) outFull.update(chunk);
    await outCheck.update(chunk);
    await sink.write(chunk);
    written += chunk.length;
    if (written - lastReport >= 1 << 20 || written === plan.size) {
      lastReport = written;
      onProgress({ phase: 'patch', loaded: written, total: plan.size });
    }
  };
  const zeros = async (n) => {
    while (n > 0) {
      const len = Math.min(n, CHUNK);
      await write(new Uint8Array(len));
      n -= len;
    }
  };
  const copy = (start, end) => eachChunk(source, start, end, write);

  const newEntry = async (e) => {
    const src = afs.entries[e.index];
    const before = await source.read(src.offset, src.size);
    if ((await digest(before)) !== e.source.sha256) {
      throw new PatchError('damaged-file', `${e.name} in your SF33RD.AFS is not the original. Copy the file from the disc again.`);
    }
    const delta = await readDelta(e);
    if (delta.length !== e.delta.length || (await digest(delta)) !== e.delta.sha256) {
      throw new PatchError('bad-patch', `Patch data for ${e.name} is damaged.`);
    }
    const after = vcdiffDecode(delta, before, { expectedSize: e.target.size });
    if (after.length !== e.target.size || (await digest(after)) !== e.target.sha256) {
      throw new PatchError('delta-failed', `Patching ${e.name} failed.`);
    }
    return after;
  };

  if (plan.mode === 'copy') {
    await copy(0, afs.total);
  } else if (plan.mode === 'in-place') {
    let at = 0;
    const order = [...manifest.entries].sort((a, b) => afs.entries[a.index].offset - afs.entries[b.index].offset);
    for (const e of order) {
      const src = afs.entries[e.index];
      await copy(at, src.offset);
      await write(await newEntry(e));
      at = src.offset + src.size;
    }
    await copy(at, afs.total);
  } else {
    // Header: original bytes up to the first entry, with the new table of contents and name-table pointer
    const header = await source.read(0, afs.first);
    plan.placed.forEach((p, i) => {
      putU32(header, 8 + i * 8, p.offset);
      putU32(header, 12 + i * 8, p.size);
    });
    for (const pos of afs.pointerPositions) {
      putU32(header, pos, plan.nameOffset);
      putU32(header, pos + 4, afs.nameSize);
    }
    await write(header);

    for (const src of afs.entries) {
      const p = plan.placed[src.index];
      if (!p.written) continue;
      await zeros(p.offset - written);
      const e = changes.get(src.index);
      if (e) await write(await newEntry(e));
      else await copy(src.offset, src.offset + src.size);
    }

    await zeros(plan.nameOffset - written);
    const names = await source.read(afs.nameOffset, afs.nameSize);
    plan.placed.forEach((p, i) => {
      if (p.written) putU32(names, i * 48 + 44, p.size);
    });
    await write(names);
    await copy(afs.nameOffset + afs.nameSize, afs.total);
    await zeros(plan.size - written);
  }

  await outCheck.finish();
  const sha256 = outFull ? outFull.hex() : null;
  if (written !== manifest.target.size || (sha256 && sha256 !== manifest.target.sha256)) badOutput();
  return { size: written, sha256 };
}
