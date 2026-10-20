// PUMP FIGHTERS web build host: a static host for web/dist (the page, the WebAssembly build and
// the public patch files). Players bring their own SF33RD.AFS; it is patched in their browser,
// so no game data is ever served from here.
//
// - Static files come from the ASSETS binding; the Worker only adds response headers.
// - The game is single-threaded (no SharedArrayBuffer), so no COOP/COEP headers are needed.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const response = await env.ASSETS.fetch(request);
    return withPageHeaders(response, url.pathname);
  },
};

// ---------------------------------------------------------------- page headers

function withPageHeaders(response, pathname) {
  const out = new Response(response.body, response);
  out.headers.set('X-Robots-Tag', 'noindex, nofollow');
  out.headers.set('X-Content-Type-Options', 'nosniff');
  out.headers.set('Referrer-Policy', 'no-referrer');

  // The engine files change with every build: always revalidate them (assets send ETags)
  // (and the patch for "bring your own file": manifest + delta parts, no game data)
  if (/\.(html|js|wasm|css|webmanifest)$/.test(pathname) || pathname.endsWith('/') || pathname.startsWith('/patch/')) {
    out.headers.set('Cache-Control', 'no-cache');
  } else if (pathname.startsWith('/icons/')) {
    out.headers.set('Cache-Control', 'public, max-age=86400');
  }

  return out;
}
