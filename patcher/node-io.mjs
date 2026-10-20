// Node helpers for afs-patch.js: a file on disk as a patch source, a sequential file sink and
// the native SHA-256.
import { createHash } from 'node:crypto';
import { open } from 'node:fs/promises';

export const nodeHash = () => {
  const h = createHash('sha256');
  return {
    update(bytes) {
      h.update(bytes);
      return this;
    },
    hex: () => h.digest('hex'),
  };
};

/** Opens a file as { size, read(offset, length), close() }. */
export async function fileSource(path) {
  const fh = await open(path, 'r');
  const { size } = await fh.stat();
  return {
    size,
    async read(offset, length) {
      const buf = new Uint8Array(length);
      let got = 0;
      while (got < length) {
        const { bytesRead } = await fh.read(buf, got, length - got, offset + got);
        if (!bytesRead) break;
        got += bytesRead;
      }
      return got === length ? buf : buf.subarray(0, got);
    },
    close: () => fh.close(),
  };
}

/** Sequential writer: { write(chunk), close() }. Creates or truncates the file. */
export async function fileSink(path) {
  const fh = await open(path, 'w');
  let pos = 0;
  return {
    async write(chunk) {
      let done = 0;
      while (done < chunk.length) {
        const { bytesWritten } = await fh.write(chunk, done, chunk.length - done, pos + done);
        done += bytesWritten;
      }
      pos += chunk.length;
    },
    close: () => fh.close(),
  };
}

/** Loads patch/pumpfighters.json and returns { manifest, reader } reading parts next to it. */
export async function loadPatch(manifestPath) {
  const { readFile } = await import('node:fs/promises');
  const { dirname, join } = await import('node:path');
  const { checkManifest, createPartReader } = await import('./afs-patch.js');
  const manifest = checkManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  const reader = createPartReader(
    manifest,
    async (part) => new Uint8Array(await readFile(join(dirname(manifestPath), part.file))),
  );
  return { manifest, reader };
}
