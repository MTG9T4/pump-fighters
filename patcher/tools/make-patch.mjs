#!/usr/bin/env node
// Builds the PUMP FIGHTERS patch from the original and the modded SF33RD.AFS.
//   node patcher/tools/make-patch.mjs [--original tools/game/SF33RD.AFS] [--modded tools/sf3mod/build/SF33RD.AFS]
//                                     [--out patch/pumpfighters.json] [--audit] [--audit-against <afs>]
// Writes the manifest (pumpfighters.json) and the deltas, one VCDIFF delta per changed entry,
// packed in entry order into parts of at most 20 MB (pumpfighters-1.deltas, -2.deltas, ...).
// Unchanged entries are not in the patch. Every delta is decoded again (by our decoder and by
// @ably/vcdiff-decoder when installed) and compared with the modded entry.
// (Not ".bin": some Windows setups hold every fetch of a *.bin URL for ~70 s.)
// --audit also scans the whole original archive for any literal (ADD) bytes of the deltas that
// appear verbatim in the original game data.
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import {
  PATCH_FORMAT,
  PATCH_VERSION,
  checkManifest,
  chunkDigests,
  HASH_CHUNK,
  planTarget,
  readAfs,
  vcdiffDecode,
} from '../afs-patch.js';
import { fileSource, nodeHash } from '../node-io.mjs';
import { vcdiffEncode } from './vcdiff-encode.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const project = resolve(here, '..', '..');
const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const originalPath = resolve(opt('--original', resolve(project, 'tools/game/SF33RD.AFS')));
const moddedPath = resolve(opt('--modded', resolve(project, 'tools/sf3mod/build/SF33RD.AFS')));
const outPath = resolve(opt('--out', resolve(project, 'patch/pumpfighters.json')));
const PART_LIMIT = 20 * 1000 * 1000;
const audit = args.includes('--audit') || args.includes('--audit-against');

let ablyDecode = null;
try {
  ablyDecode = createRequire(import.meta.url)('@ably/vcdiff-decoder').decode;
} catch (_) {
  console.warn('note: @ably/vcdiff-decoder not installed (npm install in patcher/); skipping the second decoder check');
}

const sha = (bytes) => nodeHash().update(bytes).hex();
const mb = (n) => `${(n / 1e6).toFixed(2)} MB`;

async function fileSha(src) {
  const h = nodeHash();
  for (let at = 0; at < src.size; at += 16 << 20) h.update(await src.read(at, Math.min(16 << 20, src.size - at)));
  return h.hex();
}

function entryType(name) {
  if (/^pl\d\dpl\.bin$/i.test(name)) return 'fighter palettes (plNNpl.bin)';
  if (/^pl\d\d\.bin$/i.test(name)) return 'fighter sprite banks (plNN.bin)';
  if (/^ef02_usa\.bin$/i.test(name)) return 'select/VS/win art (ef02_usa.bin)';
  if (/\.ppg$/i.test(name)) return 'textures (.ppg)';
  return 'other (.bin)';
}

const orig = await fileSource(originalPath);
const mod = await fileSource(moddedPath);
const a = await readAfs(orig);
const b = await readAfs(mod);

if (a.count !== b.count) throw new Error(`entry count differs: ${a.count} vs ${b.count}`);
for (let i = 0; i < a.count; i++) {
  if (a.entries[i].name !== b.entries[i].name) throw new Error(`entry ${i} name differs`);
}

console.log('hashing both archives…');
const [origSha, modSha] = [await fileSha(orig), await fileSha(mod)];
console.log(`original ${origSha} (${a.total} bytes)\nmodded   ${modSha} (${b.total} bytes)`);

// Which entries changed
const changed = [];
for (let i = 0; i < a.count; i++) {
  const x = a.entries[i];
  const y = b.entries[i];
  if (x.size !== y.size) {
    changed.push(i);
    continue;
  }
  if (!x.size) continue;
  const [bx, by] = [await orig.read(x.offset, x.size), await mod.read(y.offset, y.size)];
  if (Buffer.compare(bx, by) !== 0) changed.push(i);
}

// The modded archive must have exactly the layout the patcher rebuilds
const plan = planTarget(a, new Map(changed.map((i) => [i, b.entries[i].size])));
if (plan.size !== b.total) throw new Error(`layout: patcher would write ${plan.size} bytes, modded archive has ${b.total}`);
if (plan.mode === 'repack') {
  plan.placed.forEach((p, i) => {
    if (p.offset !== b.entries[i].offset || p.size !== b.entries[i].size) throw new Error(`layout differs at entry ${i}`);
  });
  if (plan.nameOffset !== b.nameOffset) throw new Error('name table offset differs');
}

const entries = [];
const parts = []; // { chunks: Uint8Array[], length }
const byType = new Map();
const place = (delta) => {
  let last = parts[parts.length - 1];
  if (!last || (last.length && last.length + delta.length > PART_LIMIT)) {
    last = { chunks: [], length: 0 };
    parts.push(last);
  }
  const at = { part: parts.length - 1, offset: last.length, length: delta.length };
  last.chunks.push(delta);
  last.length += delta.length;
  return at;
};
const targets = new Map(); // index -> { bytes, adds } for --audit

for (const i of changed) {
  const x = a.entries[i];
  const y = b.entries[i];
  const before = await orig.read(x.offset, x.size);
  const after = await mod.read(y.offset, y.size);
  const t0 = Date.now();
  const { delta, stats } = vcdiffEncode(before, after);

  const check = vcdiffDecode(delta, before, { expectedSize: after.length });
  if (Buffer.compare(check, after) !== 0) throw new Error(`delta for ${x.name} does not decode to the modded entry`);
  if (ablyDecode) {
    const second = ablyDecode(delta, before);
    if (Buffer.compare(Buffer.from(second), after) !== 0) throw new Error(`@ably/vcdiff-decoder disagrees on ${x.name}`);
  }

  entries.push({
    index: i,
    name: x.name,
    source: { size: x.size, sha256: sha(before) },
    target: { size: y.size, sha256: sha(after) },
    delta: {
      ...place(delta),
      sha256: sha(delta),
      add: stats.addBytes,
      copy: stats.copySourceBytes + stats.copyTargetBytes,
      copyFromOriginal: stats.copySourceBytes,
      run: stats.runBytes,
    },
  });
  if (audit) targets.set(i, { bytes: after, adds: stats.adds, name: x.name });

  const type = entryType(x.name);
  const t = byType.get(type) || { entries: 0, target: 0, delta: 0, add: 0, copyFromOriginal: 0, copyFromNew: 0, run: 0 };
  t.entries++;
  t.target += y.size;
  t.delta += delta.length;
  t.add += stats.addBytes;
  t.copyFromOriginal += stats.copySourceBytes;
  t.copyFromNew += stats.copyTargetBytes;
  t.run += stats.runBytes;
  byType.set(type, t);

  console.log(
    `${String(i).padStart(4)} ${x.name.padEnd(14)} ${String(x.size).padStart(8)} -> ${String(y.size).padStart(8)}` +
      `  delta ${String(delta.length).padStart(8)}  add ${String(stats.addBytes).padStart(8)}` +
      `  copy ${String(stats.copySourceBytes).padStart(8)}+${String(stats.copyTargetBytes).padStart(7)}  run ${String(stats.runBytes).padStart(6)}` +
      `  ${Date.now() - t0} ms`,
  );
}

const stem = basename(outPath).replace(/\.json$/, '');
const partFiles = parts.map((p, i) => {
  const bytes = new Uint8Array(p.length);
  let at = 0;
  for (const c of p.chunks) {
    bytes.set(c, at);
    at += c.length;
  }
  return { file: `${stem}-${i + 1}.deltas`, bytes };
});
const deltaTotal = parts.reduce((n, p) => n + p.length, 0);

const manifest = {
  format: PATCH_FORMAT,
  version: PATCH_VERSION,
  title: 'PUMP FIGHTERS',
  needs: 'SF33RD.AFS from your own copy of Street Fighter III 3rd Strike (PS2, US v1.0)',
  created: new Date().toISOString(),
  hashChunk: HASH_CHUNK,
  source: { size: a.total, sha256: origSha, entries: a.count, tocSha256: sha(a.tocBytes), chunks: await chunkDigests(orig) },
  target: { size: b.total, sha256: modSha, chunks: await chunkDigests(mod) },
  layout: plan.mode,
  deltaEncoding: 'VCDIFF (RFC 3284), default code table, no secondary compression; source segment = the original entry',
  parts: partFiles.map((p) => ({ file: p.file, length: p.bytes.length, sha256: sha(p.bytes) })),
  entries,
};
checkManifest(manifest);

const outDir = dirname(outPath);
await mkdir(outDir, { recursive: true });
const partName = new RegExp(`^${stem}-\\d+\\.(bin|deltas)$`);
for (const f of await readdir(outDir)) if (partName.test(f)) await rm(join(outDir, f));
for (const p of partFiles) await writeFile(join(outDir, p.file), p.bytes);
const json = JSON.stringify(manifest, null, 1) + '\n';
await writeFile(outPath, json);

console.log(`\n${changed.length} changed entries of ${a.count}; layout ${plan.mode}`);
console.log(`patch: ${outPath} (${json.length} bytes) + ${partFiles.length} parts, ${deltaTotal} bytes of deltas (${mb(deltaTotal)})`);
for (const p of partFiles) console.log(`  ${p.file} ${p.bytes.length} bytes`);
console.log('\nby entry type:');
const total = { entries: 0, target: 0, delta: 0, add: 0, copyFromOriginal: 0, copyFromNew: 0, run: 0 };
for (const [type, t] of byType) {
  console.log(
    `  ${type.padEnd(36)} ${String(t.entries).padStart(3)} entries  new ${mb(t.target).padStart(9)}  delta ${mb(t.delta).padStart(9)}` +
      `  ADD ${mb(t.add).padStart(9)}  COPY orig ${mb(t.copyFromOriginal).padStart(9)}  COPY new ${mb(t.copyFromNew).padStart(9)}  RUN ${mb(t.run).padStart(8)}`,
  );
  for (const k of Object.keys(total)) total[k] += t[k];
}
console.log(
  `  ${'total'.padEnd(36)} ${String(total.entries).padStart(3)} entries  new ${mb(total.target).padStart(9)}  delta ${mb(total.delta).padStart(9)}` +
    `  ADD ${mb(total.add).padStart(9)}  COPY orig ${mb(total.copyFromOriginal).padStart(9)}  COPY new ${mb(total.copyFromNew).padStart(9)}  RUN ${mb(total.run).padStart(8)}`,
);

// --audit-against <file>: scan another archive instead (e.g. the modded one, as a self-test that
// the audit does find literal bytes where they exist)
if (audit) {
  const against = opt('--audit-against', null);
  if (against) {
    const other = await fileSource(resolve(against));
    await auditAdds(other, await readAfs(other), targets);
    await other.close();
  } else {
    await auditAdds(orig, a, targets);
  }
}

await orig.close();
await mod.close();

/**
 * Finds literal (ADD) bytes that also occur verbatim in the original archive, as runs of >= 47
 * bytes (32-byte windows sampled every 16 bytes). Low-entropy windows (< 6 distinct byte values,
 * e.g. zero fill) are ignored.
 */
async function auditAdds(src, afs, targetsByIndex) {
  const K = 32;
  const STRIDE = 16;
  const B = 0x01000193;
  let BK = 1;
  for (let i = 0; i < K; i++) BK = Math.imul(BK, B);
  const windowHash = (buf, p) => {
    let h = 0;
    for (let k = 0; k < K; k++) h = (Math.imul(h, B) + buf[p + k]) | 0;
    return h;
  };

  const windows = []; // [index, pos]
  const map = new Map();
  const bits = new Uint32Array(1 << 21); // 2^26-bit prefilter
  let addTotal = 0;
  let lowEntropy = 0;
  for (const [index, t] of targetsByIndex) {
    for (const [pos, len] of t.adds) {
      addTotal += len;
      for (let p = pos + ((STRIDE - (pos % STRIDE)) % STRIDE); p + K <= pos + len; p += STRIDE) {
        const seen = new Set(t.bytes.subarray(p, p + K));
        if (seen.size < 6) {
          lowEntropy++;
          continue;
        }
        const h = windowHash(t.bytes, p);
        const id = windows.push([index, p]) - 1;
        const list = map.get(h);
        if (list) list.push(id);
        else map.set(h, [id]);
        const k = h >>> 6;
        bits[k >>> 5] |= 1 << (k & 31);
      }
    }
  }
  console.log(`\naudit: ${windows.length} sampled ADD windows (${lowEntropy} low-entropy skipped) over ${mb(addTotal)} of ADD bytes`);

  // Which original entry an offset belongs to
  const sorted = afs.entries.filter((e) => e.size).sort((x, y) => x.offset - y.offset);
  const owner = (off) => {
    let lo = 0;
    let hi = sorted.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid].offset <= off) lo = mid + 1;
      else hi = mid - 1;
    }
    const e = sorted[hi];
    return e && off < e.offset + e.size ? e.name : '(header/padding)';
  };

  const hitWindows = new Set();
  const from = new Map();
  const CH = 32 << 20;
  let carry = new Uint8Array(0);
  for (let base = 0; base < src.size; base += CH) {
    const chunk = await src.read(base, Math.min(CH, src.size - base));
    const buf = new Uint8Array(carry.length + chunk.length);
    buf.set(carry);
    buf.set(chunk, carry.length);
    const bufStart = base - carry.length;
    if (buf.length >= K) {
      let h = windowHash(buf, 0);
      for (let p = 0; ; p++) {
        const k = h >>> 6;
        if (bits[k >>> 5] & (1 << (k & 31))) {
          const list = map.get(h);
          if (list) {
            for (const id of list) {
              const [index, wp] = windows[id];
              const tb = targetsByIndex.get(index).bytes;
              let same = true;
              for (let q = 0; q < K; q++) {
                if (tb[wp + q] !== buf[p + q]) {
                  same = false;
                  break;
                }
              }
              if (same && !hitWindows.has(id)) {
                hitWindows.add(id);
                const key = `${targetsByIndex.get(index).name} <- ${owner(bufStart + p)}`;
                from.set(key, (from.get(key) || 0) + 1);
              }
            }
          }
        }
        if (p + K >= buf.length) break;
        h = (Math.imul(h, B) - Math.imul(buf[p], BK) + buf[p + K]) | 0;
      }
    }
    carry = buf.slice(buf.length - (K - 1));
  }

  // Bytes covered by matching windows
  const covered = new Map();
  for (const id of hitWindows) {
    const [index, p] = windows[id];
    if (!covered.has(index)) covered.set(index, new Uint8Array(targetsByIndex.get(index).bytes.length));
    covered.get(index).fill(1, p, p + K);
  }
  let coveredBytes = 0;
  for (const c of covered.values()) for (const v of c) coveredBytes += v;
  console.log(`audit: ${hitWindows.size} ADD windows found verbatim in the original archive, covering ~${coveredBytes} bytes`);
  for (const [key, n] of [...from].sort((x, y) => y[1] - x[1]).slice(0, 25)) console.log(`  ${n.toString().padStart(6)} windows  ${key}`);
}
