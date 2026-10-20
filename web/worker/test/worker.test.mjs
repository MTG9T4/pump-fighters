// Unit tests for the static-host Worker with a mocked ASSETS binding.
//   node --test web/worker/test/worker.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';

const worker = (await import('../src/index.js')).default;

// ---------------------------------------------------------------- mocks

function assets(status = 200, contentType = 'text/html', body = '<html>page</html>') {
  const seen = [];
  return {
    seen,
    async fetch(request) {
      seen.push(new URL(request.url).pathname);
      return new Response(status === 304 ? null : body, { status, headers: { 'content-type': contentType, etag: '"abc"' } });
    },
  };
}

async function call(env, path, method = 'GET') {
  const res = await worker.fetch(new Request(`https://fighters.example${path}`, { method }), env);
  return { res, text: method === 'HEAD' || res.status === 304 ? null : await res.text() };
}

// ---------------------------------------------------------------- tests

test('every request is served from the static assets', async () => {
  const env = { ASSETS: assets() };
  for (const path of ['/', '/index.html', '/patch.html', '/app.js', '/patch/pumpfighters.json', '/icons/icon-192.png', '/missing.txt']) {
    const { res, text } = await call(env, path);
    assert.equal(res.status, 200, path);
    assert.equal(text, '<html>page</html>', path);
  }
  assert.deepEqual(env.ASSETS.seen, ['/', '/index.html', '/patch.html', '/app.js', '/patch/pumpfighters.json', '/icons/icon-192.png', '/missing.txt']);
});

test('pages, engine files and the patch get no-cache and the page headers', async () => {
  for (const path of ['/', '/index.html', '/app.js', '/pumpcombat.wasm', '/style.css', '/manifest.webmanifest', '/patch/pumpfighters-0.deltas']) {
    const { res } = await call({ ASSETS: assets() }, path);
    assert.equal(res.headers.get('cache-control'), 'no-cache', path);
    assert.equal(res.headers.get('x-robots-tag'), 'noindex, nofollow', path);
    assert.equal(res.headers.get('x-content-type-options'), 'nosniff', path);
    assert.equal(res.headers.get('referrer-policy'), 'no-referrer', path);
  }
});

test('icons are cached for a day, other files keep the asset defaults', async () => {
  let { res } = await call({ ASSETS: assets(200, 'image/png', 'png') }, '/icons/icon-192.png');
  assert.equal(res.headers.get('cache-control'), 'public, max-age=86400');
  ({ res } = await call({ ASSETS: assets(200, 'image/png', 'png') }, '/other.png'));
  assert.equal(res.headers.get('cache-control'), null);
});

test('status, content type and ETag from the assets pass through', async () => {
  let { res } = await call({ ASSETS: assets(404, 'text/plain', 'nope') }, '/missing.txt');
  assert.equal(res.status, 404);
  assert.equal(res.headers.get('content-type'), 'text/plain');

  ({ res } = await call({ ASSETS: assets(304) }, '/app.js'));
  assert.equal(res.status, 304);
  assert.equal(res.headers.get('etag'), '"abc"');

  ({ res } = await call({ ASSETS: assets(200, 'application/wasm', 'wasm') }, '/pumpcombat.wasm', 'HEAD'));
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'application/wasm');
});

test('no COOP/COEP headers: the game is single-threaded', async () => {
  const { res } = await call({ ASSETS: assets() }, '/index.html');
  assert.equal(res.headers.get('cross-origin-opener-policy'), null);
  assert.equal(res.headers.get('cross-origin-embedder-policy'), null);
});
