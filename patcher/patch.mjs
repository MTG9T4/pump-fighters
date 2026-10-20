#!/usr/bin/env node
// PUMP FIGHTERS patcher (command line).
//   node patcher/patch.mjs <your original SF33RD.AFS> <output SF33RD.AFS> [--patch <pumpfighters.json>]
// The patch is the manifest plus the pumpfighters-N.deltas parts next to it (default: ../patch/).
// Your original file is only read. The output is written next to it as <output>.part and renamed
// when its checksum is right.
import { rename, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyPatch, PatchError } from './afs-patch.js';
import { fileSink, fileSource, loadPatch, nodeHash } from './node-io.mjs';

const args = process.argv.slice(2);
const flag = args.indexOf('--patch');
const patchPath = flag >= 0 ? args.splice(flag, 2)[1] : fileURLToPath(new URL('../patch/pumpfighters.json', import.meta.url));
const [input, output] = args;

if (!input || !output) {
  console.error('usage: node patch.mjs <original SF33RD.AFS> <output SF33RD.AFS> [--patch pumpfighters.json]');
  process.exit(2);
}
if (resolve(input).toLowerCase() === resolve(output).toLowerCase()) {
  console.error('error: the output must be a different file; your original is never overwritten');
  process.exit(2);
}

const part = `${output}.part`;
const source = await fileSource(input);
let sink;
let lastLine = '';

try {
  const { manifest, reader } = await loadPatch(patchPath);
  reader.prefetch(0);
  sink = await fileSink(part);
  const t0 = Date.now();
  const result = await applyPatch({
    manifest,
    readDelta: reader.read,
    source,
    sink,
    createHash: nodeHash,
    onProgress: ({ phase, loaded, total }) => {
      const label = { check: 'checking', verify: 'verifying original', patch: 'writing' }[phase];
      const line = `${label} ${Math.floor((loaded / total) * 100)}%`;
      if (line !== lastLine && process.stdout.isTTY) process.stdout.write(`\r${line}   `);
      lastLine = line;
    },
  });
  await sink.close();
  sink = null;
  await rename(part, output);
  if (process.stdout.isTTY) process.stdout.write('\r');
  console.log(`done: ${output} (${result.size} bytes, sha256 ${result.sha256}) in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
} catch (error) {
  if (sink) await sink.close().catch(() => {});
  await rm(part, { force: true });
  if (process.stdout.isTTY) process.stdout.write('\n');
  console.error(`error: ${error instanceof PatchError ? error.message : error.stack || error}`);
  process.exitCode = 1;
} finally {
  await source.close();
}
