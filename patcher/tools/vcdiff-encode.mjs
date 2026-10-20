// VCDIFF (RFC 3284) encoder for the PUMP FIGHTERS patch: the same wire format xdelta3 writes with
// `-S none` (no secondary compression), default code table, one window per entry whose source
// segment is the whole original entry. Any xdelta3 / open-vcdiff decoder can apply the result.
//
// Matching favours copies: every run of >= 4 bytes that also exists in the original entry (or
// earlier in the new entry) becomes a COPY, so the deltas carry as few literal (ADD) bytes as possible.

import { defaultCodeTable } from '../afs-patch.js';

const NOOP = 0, ADD = 1, RUN = 2, COPY = 3;
const S_NEAR = 4;
const S_SAME = 3;

const TABLE = defaultCodeTable();
const SINGLE = new Map(); // "inst,size,mode" -> opcode (size 0 = size follows)
const PAIR = new Map(); // "i1,s1,m1,i2,s2,m2" -> opcode
TABLE.forEach(([i1, s1, m1, i2, s2, m2], op) => {
  if (i2 === NOOP) SINGLE.set(`${i1},${s1},${m1}`, op);
  else PAIR.set(`${i1},${s1},${m1},${i2},${s2},${m2}`, op);
});

class Bytes {
  constructor(n = 1 << 16) {
    this.b = new Uint8Array(n);
    this.n = 0;
  }
  room(k) {
    if (this.n + k <= this.b.length) return;
    const g = new Uint8Array(Math.max(this.b.length * 2, this.n + k));
    g.set(this.b.subarray(0, this.n));
    this.b = g;
  }
  push(v) {
    this.room(1);
    this.b[this.n++] = v;
  }
  set(arr) {
    this.room(arr.length);
    this.b.set(arr, this.n);
    this.n += arr.length;
  }
  varint(v) {
    const tmp = [];
    do {
      tmp.push(v % 128);
      v = Math.floor(v / 128);
    } while (v > 0);
    for (let i = tmp.length - 1; i >= 0; i--) this.push(tmp[i] | (i ? 0x80 : 0));
  }
  get bytes() {
    return this.b.subarray(0, this.n);
  }
}

const varLen = (v) => {
  let n = 1;
  while (v >= 128) {
    v = Math.floor(v / 128);
    n++;
  }
  return n;
};

/**
 * @param {Uint8Array} source original entry
 * @param {Uint8Array} target new entry
 * @param {{chainDepth?: number}} [opts]
 * @returns {{delta: Uint8Array, stats: {addBytes: number, copySourceBytes: number, copyTargetBytes: number, runBytes: number, adds: number[][]}}}
 */
export function vcdiffEncode(source, target, { chainDepth = 128 } = {}) {
  const S = source.length;
  const T = target.length;
  const U = new Uint8Array(S + T);
  U.set(source);
  U.set(target, S);
  const end = S + T;

  const HB = 20;
  const head = new Int32Array(1 << HB).fill(-1);
  const prev = new Int32Array(Math.max(end, 1));
  const hashAt = (i) => (Math.imul(U[i] | (U[i + 1] << 8) | (U[i + 2] << 16) | (U[i + 3] << 24), 0x9e3779b1) >>> (32 - HB));
  const insert = (i) => {
    if (i + 4 > end) return;
    const h = hashAt(i);
    prev[i] = head[h];
    head[h] = i;
  };
  for (let i = 0; i + 4 <= S; i++) insert(i);

  const data = new Bytes();
  const inst = new Bytes();
  const addr = new Bytes();
  const near = [0, 0, 0, 0];
  let nextSlot = 0;
  const same = new Float64Array(S_SAME * 256);
  let pending = null;
  const stats = { addBytes: 0, copySourceBytes: 0, copyTargetBytes: 0, runBytes: 0, adds: [] };

  const flushSingle = (x) => {
    const op = SINGLE.get(`${x.inst},${x.size},${x.mode}`);
    if (op !== undefined && x.size !== 0) {
      inst.push(op);
    } else {
      inst.push(SINGLE.get(`${x.inst},0,${x.mode}`));
      inst.varint(x.size);
    }
  };
  const emit = (x) => {
    if (pending) {
      const op = PAIR.get(`${pending.inst},${pending.size},${pending.mode},${x.inst},${x.size},${x.mode}`);
      if (op !== undefined) {
        inst.push(op);
        pending = null;
        return;
      }
      flushSingle(pending);
    }
    pending = x;
  };

  const addLiteral = (from, to) => {
    // from/to are positions in U (target part)
    let at = from;
    while (at < to) {
      const len = to - at;
      data.set(U.subarray(at, at + len));
      emit({ inst: ADD, size: len, mode: 0 });
      stats.addBytes += len;
      stats.adds.push([at - S, len]);
      at += len;
    }
  };

  const copy = (a, len, here) => {
    // choose the cheapest address mode
    let mode = 0;
    let cost = varLen(a);
    let value = a;
    const hv = here - a;
    if (varLen(hv) < cost) {
      mode = 1;
      cost = varLen(hv);
      value = hv;
    }
    for (let i = 0; i < S_NEAR; i++) {
      const d = a - near[i];
      if (d >= 0 && varLen(d) < cost) {
        mode = 2 + i;
        cost = varLen(d);
        value = d;
      }
    }
    const slot = a % (S_SAME * 256);
    const sameHit = same[slot] === a;
    if (sameHit && cost > 1) {
      mode = 2 + S_NEAR + Math.floor(slot / 256);
      addr.push(slot % 256);
    } else {
      addr.varint(value);
    }
    near[nextSlot] = a;
    nextSlot = (nextSlot + 1) % S_NEAR;
    same[slot] = a;
    emit({ inst: COPY, size: len, mode });
    if (a < S) stats.copySourceBytes += len;
    else stats.copyTargetBytes += len;
  };

  // Bytes a COPY of `a` at position `here` would cost in the address section (cache state as now)
  const addrCost = (a, here) => {
    let cost = Math.min(varLen(a), varLen(here - a));
    for (let i = 0; i < S_NEAR && cost > 1; i++) if (a >= near[i]) cost = Math.min(cost, varLen(a - near[i]));
    if (cost > 1 && same[a % (S_SAME * 256)] === a) cost = 1;
    return cost;
  };

  // Best match by bytes saved over literals. A copy costs an opcode (+ size bytes past 18) plus
  // its address; copies from the original entry are kept even when they only break even, so
  // original bytes are referenced rather than carried.
  const findMatch = (p) => {
    if (p + 4 > end) return { len: 0, at: -1, gain: -1 };
    let bestLen = 0;
    let bestAt = -1;
    let bestGain = -1;
    let c = head[hashAt(p)];
    for (let depth = 0; c >= 0 && depth < chainDepth; depth++, c = prev[c]) {
      if (c >= p) continue;
      const max = c < S ? Math.min(end - p, S - c) : end - p;
      if (max < 4 || (max <= bestLen && bestGain >= 0)) continue;
      let k = 0;
      while (k < max && U[c + k] === U[p + k]) k++;
      if (k < 4) continue;
      const cost = 1 + (k > 18 ? varLen(k) : 0) + addrCost(c, p);
      const gain = k - cost + (c < S ? 1 : 0);
      if (gain > bestGain || (gain === bestGain && k > bestLen)) {
        bestLen = k;
        bestAt = c;
        bestGain = gain;
        if (k === max && k > 256) break;
      }
    }
    return bestGain >= 0 ? { len: bestLen, at: bestAt, gain: bestGain } : { len: 0, at: -1, gain: -1 };
  };

  let p = S;
  let litStart = S;
  while (p < end) {
    let run = 1;
    while (p + run < end && U[p + run] === U[p]) run++;
    let m = findMatch(p);

    // one-step lazy matching: a clearly longer match one byte later wins
    if (m.len >= 4 && m.len < 64 && p + 1 < end) {
      const m2 = findMatch(p + 1);
      if (m2.gain > m.gain + 1) {
        insert(p);
        p++;
        m = m2;
        run = 1;
        while (p + run < end && U[p + run] === U[p]) run++;
      }
    }

    if (run >= 4 && run >= m.len) {
      addLiteral(litStart, p);
      data.push(U[p]);
      emit({ inst: RUN, size: run, mode: 0 });
      stats.runBytes += run;
      for (let i = 0; i < run; i++) insert(p + i);
      p += run;
      litStart = p;
    } else if (m.len >= 4) {
      addLiteral(litStart, p);
      copy(m.at, m.len, p);
      for (let i = 0; i < m.len; i++) insert(p + i);
      p += m.len;
      litStart = p;
    } else {
      insert(p);
      p++;
    }
  }
  addLiteral(litStart, end);
  if (pending) flushSingle(pending);

  const body = new Bytes(data.n + inst.n + addr.n + 32);
  body.varint(T);
  body.push(0); // Delta_Indicator: no compressed sections
  body.varint(data.n);
  body.varint(inst.n);
  body.varint(addr.n);
  body.set(data.bytes);
  body.set(inst.bytes);
  body.set(addr.bytes);

  const file = new Bytes(body.n + 32);
  file.set([0xd6, 0xc3, 0xc4, 0x00, 0x00]); // magic, version 0, no header flags
  if (S > 0) {
    file.push(0x01); // VCD_SOURCE
    file.varint(S);
    file.varint(0);
  } else {
    file.push(0x00);
  }
  file.varint(body.n);
  file.set(body.bytes);

  return { delta: file.bytes.slice(), stats };
}
