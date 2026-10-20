// Unit tests for the PUMP FIGHTERS patcher. Every archive here is synthetic, built by the test;
// no game data is used.   cd patcher && npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  HASH_CHUNK,
  PatchError,
  Sha256,
  applyPatch,
  align,
  blobSource,
  createPartReader,
  planTarget,
  readAfs,
  vcdiffDecode,
} from '../afs-patch.js';
import { loadPatch, nodeHash } from '../node-io.mjs';
import { vcdiffEncode } from '../tools/vcdiff-encode.mjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const project = resolve(here, '..', '..');
const afsPy = join(project, 'tools', 'sf3mod', 'tools', 'afs.py');
const makePatch = join(project, 'patcher', 'tools', 'make-patch.mjs');
const patchCli = join(project, 'patcher', 'patch.mjs');
const sha = (b) => createHash('sha256').update(b).digest('hex');

let ablyDecode = null;
try {
  ablyDecode = createRequire(import.meta.url)('@ably/vcdiff-decoder').decode;
} catch (_) {
  // optional second decoder
}

const hasPython = (() => {
  try {
    execFileSync('python', ['--version'], { stdio: 'ignore' });
    return existsSync(afsPy);
  } catch (_) {
    return false;
  }
})();

// ------------------------------------------------------------------ synthetic data

let seed = 12345;
const rnd = () => {
  seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
  return seed >>> 16;
};
const noise = (n) => Uint8Array.from({ length: n }, () => rnd() & 0xff);
/** Structured bytes: repeats, runs and noise, like real archive data. */
function structured(n) {
  const out = new Uint8Array(n);
  let p = 0;
  while (p < n) {
    const kind = rnd() % 3;
    const len = Math.min(n - p, 16 + (rnd() % 300));
    if (kind === 0) out.fill(rnd() & 0xff, p, p + len);
    else if (kind === 1 && p > 400) out.copyWithin(p, p - 1 - (rnd() % 400), p - 1 - (rnd() % 400) + len);
    else out.set(noise(len), p);
    p += len;
  }
  return out;
}

/**
 * Builds an AFS archive like the PS2 one: header + TOC, name-table pointer (after the TOC or just
 * before the first entry), entries on 2048-byte sectors, 48-byte name rows whose +44 field is NOT
 * the size (as on the disc), and a trailer.
 */
function buildAfs(entries, { pointerBeforeFirst = false, trailer = 48 } = {}) {
  const count = entries.length;
  const first = align(8 + count * 8 + 8 + 64);
  let pos = first;
  const offsets = entries.map((e) => {
    if (e.zeroOffset) return 0;
    pos = align(pos);
    const at = pos;
    pos += e.data.length;
    return at;
  });
  const nameOffset = align(pos);
  const nameSize = count * 48;
  const total = align(nameOffset + nameSize + trailer);
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  out.set([0x41, 0x46, 0x53, 0]);
  dv.setUint32(4, count, true);
  entries.forEach((e, i) => {
    dv.setUint32(8 + i * 8, offsets[i], true);
    dv.setUint32(12 + i * 8, e.data.length, true);
    if (offsets[i]) out.set(e.data, offsets[i]);
    const row = nameOffset + i * 48;
    for (let k = 0; k < e.name.length; k++) out[row + k] = e.name.charCodeAt(k);
    for (let k = 32; k < 44; k++) out[row + k] = (i * 7 + k) & 0xff; // dates
    dv.setUint32(row + 44, 0xabc000 + i, true); // not the size
  });
  out.set(noise(40), 8 + count * 8 + 8); // header bytes the patcher must keep
  const ptr = pointerBeforeFirst ? first - 8 : 8 + count * 8;
  dv.setUint32(ptr, nameOffset, true);
  dv.setUint32(ptr + 4, nameSize, true);
  return out;
}

const memSource = (bytes) => ({ size: bytes.length, read: async (o, n) => bytes.slice(o, o + n) });
function memSink() {
  const chunks = [];
  return {
    write: (c) => void chunks.push(c.slice()),
    bytes() {
      return Buffer.concat(chunks);
    },
  };
}

/** Writes original + modified entries to disk, repacks with afs.py, builds the patch with make-patch.mjs. */
function fixture(name, entries, changes, opts) {
  const dir = mkdtempSync(join(tmpdir(), `pfpatch-${name}-`));
  const original = buildAfs(entries, opts);
  writeFileSync(join(dir, 'orig.afs'), original);
  mkdirSync(join(dir, 'overrides'));
  for (const [entryName, data] of Object.entries(changes)) writeFileSync(join(dir, 'overrides', entryName), data);
  execFileSync('python', [afsPy, 'repack', join(dir, 'orig.afs'), join(dir, 'overrides'), join(dir, 'mod.afs')]);
  execFileSync(process.execPath, [
    makePatch,
    '--original', join(dir, 'orig.afs'),
    '--modded', join(dir, 'mod.afs'),
    '--out', join(dir, 'patch', 'fx.json'),
  ]);
  const modded = readFileSync(join(dir, 'mod.afs'));
  const manifest = JSON.parse(readFileSync(join(dir, 'patch', 'fx.json'), 'utf8'));
  return { dir, original, modded, manifest, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

async function apply(fx, source, extra = {}) {
  const { manifest, reader } = await loadPatch(join(fx.dir, 'patch', 'fx.json'));
  reader.prefetch(0);
  const sink = memSink();
  const result = await applyPatch({ manifest, readDelta: reader.read, source, sink, ...extra });
  return { result, out: sink.bytes() };
}

// ------------------------------------------------------------------ SHA-256

test('pure-JS SHA-256 matches node:crypto for every padding boundary and split', () => {
  const data = noise(4096);
  for (const n of [0, 1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1000, 4096]) {
    const b = data.subarray(0, n);
    assert.equal(new Sha256().update(b).hex(), sha(b), `length ${n}`);
    const split = new Sha256();
    for (let i = 0; i < n; i += 7) split.update(b.subarray(i, Math.min(n, i + 7)));
    assert.equal(split.hex(), sha(b), `split ${n}`);
  }
});

// ------------------------------------------------------------------ VCDIFF

test('VCDIFF round trip: edits, moves, runs, overlapping self-copies, empty source', () => {
  const src = structured(200000);
  const cases = {
    identical: src,
    edited: (() => {
      const t = src.slice();
      for (let i = 0; i < 300; i++) t[rnd() % t.length] ^= 0x5a;
      return t;
    })(),
    moved: Buffer.concat([src.subarray(120000), noise(5000), src.subarray(0, 120000)]),
    grown: Buffer.concat([src.subarray(0, 50000), structured(80000), src.subarray(50000)]),
    runs: Buffer.concat([new Uint8Array(70000).fill(7), Uint8Array.from([1, 2, 3, 4, 5]), new Uint8Array(1000)]),
    periodic: Uint8Array.from({ length: 50000 }, (_, i) => [9, 8, 7][i % 3]),
    tiny: Uint8Array.from([1]),
  };
  for (const [name, target] of Object.entries(cases)) {
    const { delta, stats } = vcdiffEncode(src, target);
    assert.deepEqual(Buffer.from(vcdiffDecode(delta, src)), Buffer.from(target), name);
    assert.equal(stats.addBytes + stats.copySourceBytes + stats.copyTargetBytes + stats.runBytes, target.length, `${name} stats`);
    if (ablyDecode) assert.deepEqual(Buffer.from(ablyDecode(delta, src)), Buffer.from(target), `${name} (@ably/vcdiff-decoder)`);
  }
  // A move of original bytes must become copies, not literals
  assert.ok(vcdiffEncode(src, cases.moved).stats.addBytes <= 5000 + 64);

  const noSource = structured(30000);
  const { delta } = vcdiffEncode(new Uint8Array(0), noSource);
  assert.deepEqual(Buffer.from(vcdiffDecode(delta, new Uint8Array(0))), Buffer.from(noSource));
});

test('VCDIFF decoder: hand-made delta with xdelta3 app header, Adler-32 and a VCD_TARGET window', () => {
  // Window 1 (VCD_SOURCE + VCD_ADLER32): ADD "ab", COPY 4 from source addr 0 (mode VCD_SELF)
  // Window 2 (VCD_TARGET, segment = first 3 target bytes): COPY 3 from 0
  const src = Uint8Array.from([0x10, 0x11, 0x12, 0x13]);
  const t1 = Uint8Array.from([0x61, 0x62, 0x10, 0x11, 0x12, 0x13]);
  const adler = (b) => {
    let x = 1, y = 0;
    for (const v of b) {
      x = (x + v) % 65521;
      y = (y + x) % 65521;
    }
    return ((y << 16) | x) >>> 0;
  };
  const a = adler(t1);
  // ADD size 2 = opcode 3; COPY size 4 mode 0 = opcode 20 (19 = COPY size 0 mode 0, 20 = size 4)
  const win1Body = [6, 0, 2, 2, 1, (a >>> 24) & 255, (a >>> 16) & 255, (a >>> 8) & 255, a & 255, 0x61, 0x62, 3, 20, 0];
  // target 3, no data, 2 instruction bytes (19 = COPY size-follows mode 0, size 3), 1 address byte (0)
  const win2Body = [3, 0, 0, 2, 1, 19, 3, 0];
  const delta = Uint8Array.from([
    0xd6, 0xc3, 0xc4, 0x00, 0x04, 3, 0x41, 0x42, 0x43, // header with a 3-byte app header
    0x05, 4, 0, win1Body.length, ...win1Body, // VCD_SOURCE | VCD_ADLER32, segment 4 @ 0
    0x02, 3, 0, win2Body.length, ...win2Body, // VCD_TARGET, segment 3 @ 0
  ]);
  assert.deepEqual([...vcdiffDecode(delta, src)], [...t1, 0x61, 0x62, 0x10]);

  const badSum = delta.slice();
  badSum[19] ^= 1; // flip a bit of the checksum
  assert.throws(() => vcdiffDecode(badSum, src), /checksum/);

  const secondary = delta.slice();
  secondary[4] = 0x01;
  assert.throws(() => vcdiffDecode(secondary, src), (e) => e instanceof PatchError && /secondary compression/.test(e.message));
  assert.throws(() => vcdiffDecode(delta.subarray(0, 20), src), PatchError);
  assert.throws(() => vcdiffDecode(Uint8Array.from([1, 2, 3, 4]), src), /not a VCDIFF/);
});

// ------------------------------------------------------------------ AFS layout

test('readAfs finds the name table after the TOC or just before the first entry', async () => {
  const entries = [{ name: 'a.bin', data: noise(100) }, { name: 'b.bin', data: noise(3000) }];
  for (const pointerBeforeFirst of [false, true]) {
    const afs = await readAfs(memSource(buildAfs(entries, { pointerBeforeFirst })));
    assert.deepEqual(afs.entries.map((e) => e.name), ['a.bin', 'b.bin']);
    assert.equal(afs.pointerPositions.length, 1);
  }
  await assert.rejects(readAfs(memSource(new Uint8Array(4096))), (e) => e.code === 'not-afs');
});

// ------------------------------------------------------------------ end to end against afs.py repack

const entrySet = () => [
  { name: 'default.bin', data: structured(61440) },
  { name: 'empty.bin', data: new Uint8Array(0), zeroOffset: true },
  { name: 'pl01.bin', data: structured(300000) },
  { name: 'big.adx', data: noise(HASH_CHUNK + 300000) }, // spans hash chunks, never changes
  { name: 'pl01pl.bin', data: structured(3584) },
  { name: 'title.bin', data: structured(11264) },
  { name: 'shrink.ppg', data: structured(50348) },
];

test('changed sizes: patch output equals afs.py repack byte for byte', { skip: !hasPython && 'python + tools/sf3mod/tools/afs.py needed' }, async () => {
  const entries = entrySet();
  const grown = Buffer.concat([entries[2].data.subarray(0, 100000), structured(250000), entries[2].data.subarray(100000)]);
  const fx = fixture('grow', entries, {
    'pl01.bin': grown,
    'shrink.ppg': entries[6].data.subarray(0, 40000),
    'title.bin': Buffer.concat([entries[5].data.subarray(5000), entries[5].data.subarray(0, 5000)]),
  });
  try {
    assert.equal(fx.manifest.layout, 'repack');
    assert.deepEqual(fx.manifest.entries.map((e) => e.name), ['pl01.bin', 'title.bin', 'shrink.ppg']);
    // Unchanged entries are not in the patch; the moved title.bin is almost all copies
    const title = fx.manifest.entries.find((e) => e.name === 'title.bin');
    assert.ok(title.delta.add < 64, `title.bin carries ${title.delta.add} literal bytes`);

    for (const extra of [{ createHash: nodeHash }, {}]) {
      const { result, out } = await apply(fx, memSource(fx.original), extra);
      assert.equal(out.length, fx.modded.length);
      assert.ok(out.equals(fx.modded), 'byte for byte');
      assert.equal(result.size, fx.modded.length);
    }
    // Browser source type
    const { out } = await apply(fx, blobSource(new Blob([fx.original])));
    assert.ok(out.equals(fx.modded));
    // The plan reproduces afs.py's offsets
    const afs = await readAfs(memSource(fx.original));
    const plan = planTarget(afs, new Map(fx.manifest.entries.map((e) => [e.index, e.target.size])));
    const modAfs = await readAfs(memSource(fx.modded));
    assert.deepEqual(plan.placed.map((p) => p.offset), modAfs.entries.map((e) => e.offset));
  } finally {
    fx.cleanup();
  }
});

test('same-size changes are written in place (padding kept), pointer before first entry', { skip: !hasPython && 'python needed' }, async () => {
  const entries = entrySet().filter((e) => !e.zeroOffset);
  const pal = entries.find((e) => e.name === 'pl01pl.bin').data.slice();
  pal.fill(0x7f, 100, 400);
  const fx = fixture('inplace', entries, { 'pl01pl.bin': pal }, { pointerBeforeFirst: true });
  try {
    assert.equal(fx.manifest.layout, 'in-place');
    const { out } = await apply(fx, memSource(fx.original), { createHash: nodeHash });
    assert.ok(out.equals(fx.modded));
  } finally {
    fx.cleanup();
  }
});

test('player-facing errors: wrong file, other version, damaged original, damaged patch', { skip: !hasPython && 'python needed' }, async () => {
  const entries = entrySet();
  const fx = fixture('errors', entries, { 'pl01.bin': structured(310000) });
  try {
    const codeOf = async (source, extra) => {
      try {
        await apply(fx, source, extra);
        return 'ok';
      } catch (e) {
        assert.ok(e instanceof PatchError, String(e));
        return e.code;
      }
    };
    assert.equal(await codeOf(memSource(fx.original.subarray(0, fx.original.length - 2048))), 'wrong-file');

    const otherToc = fx.original.slice();
    otherToc[12] ^= 1; // an entry size in the TOC
    assert.equal(await codeOf(memSource(otherToc)), 'wrong-version');

    const damaged = fx.original.slice();
    const big = (await readAfs(memSource(fx.original))).entries.find((e) => e.name === 'big.adx');
    damaged[big.offset + HASH_CHUNK + 5] ^= 0xff; // an unchanged entry, second hash chunk
    assert.equal(await codeOf(memSource(damaged)), 'damaged-file');
    assert.equal(await codeOf(memSource(damaged), { createHash: nodeHash }), 'damaged-file');
    // Without the up-front check, the output check still catches it
    assert.equal(await codeOf(memSource(damaged), { verifySource: false }), 'target-mismatch');

    const changedEntry = fx.original.slice();
    const pl = (await readAfs(memSource(fx.original))).entries.find((e) => e.name === 'pl01.bin');
    changedEntry[pl.offset + 10] ^= 0xff;
    assert.equal(await codeOf(memSource(changedEntry), { verifySource: false }), 'damaged-file');

    // A damaged part file
    const part = join(fx.dir, 'patch', fx.manifest.parts[0].file);
    const bytes = readFileSync(part);
    bytes[bytes.length >> 1] ^= 0xff;
    writeFileSync(part, bytes);
    assert.equal(await codeOf(memSource(fx.original)), 'bad-patch');
  } finally {
    fx.cleanup();
  }
});

test('part reader keeps at most two parts and fails clearly on a short download', async () => {
  const parts = [noise(100), noise(100), noise(100)];
  const manifest = {
    parts: parts.map((p, i) => ({ file: `p${i}`, length: p.length, sha256: sha(p) })),
  };
  const loads = [];
  const reader = createPartReader(manifest, async (_, i) => {
    loads.push(i);
    return i === 2 ? parts[i].subarray(0, 50) : parts[i];
  });
  const got = await reader.read({ delta: { part: 1, offset: 10, length: 5 } });
  assert.deepEqual([...got], [...parts[1].subarray(10, 15)]);
  assert.deepEqual(loads, [1, 2]); // the next part is fetched ahead
  await assert.rejects(reader.read({ delta: { part: 2, offset: 0, length: 5 } }), (e) => e.code === 'bad-patch');
});

test('command line patcher writes the file and refuses to overwrite the original', { skip: !hasPython && 'python needed' }, async () => {
  const fx = fixture('cli', entrySet(), { 'pl01.bin': structured(320000) });
  try {
    const out = join(fx.dir, 'out.afs');
    execFileSync(process.execPath, [patchCli, join(fx.dir, 'orig.afs'), out, '--patch', join(fx.dir, 'patch', 'fx.json')]);
    assert.ok(readFileSync(out).equals(fx.modded));
    assert.throws(() =>
      execFileSync(process.execPath, [patchCli, join(fx.dir, 'orig.afs'), join(fx.dir, 'orig.afs')], { stdio: 'pipe' }),
    );
    // A wrong file leaves no output behind
    writeFileSync(join(fx.dir, 'wrong.afs'), randomBytes(5000));
    assert.throws(() =>
      execFileSync(process.execPath, [patchCli, join(fx.dir, 'wrong.afs'), join(fx.dir, 'x.afs'), '--patch', join(fx.dir, 'patch', 'fx.json')], { stdio: 'pipe' }),
    );
    assert.ok(!existsSync(join(fx.dir, 'x.afs')) && !existsSync(join(fx.dir, 'x.afs.part')));
  } finally {
    fx.cleanup();
  }
});
