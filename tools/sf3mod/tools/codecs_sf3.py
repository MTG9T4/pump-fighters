"""Byte-for-byte ports of Lz77Dec.c and arcade_cg.c's LZ-p6 codec."""


def lz77_decode(src, size):
    out = bytearray(); pos = 0

    def byte():
        nonlocal pos
        if pos >= len(src):
            raise ValueError('truncated LZ77 stream')
        v = src[pos]; pos += 1
        return v

    def copy(distance, count, step=0):
        if distance > len(out):
            raise ValueError('LZ77 back-reference before output')
        for _ in range(count):
            out.append((out[-distance] + step) & 255)

    while len(out) < size:
        op = byte()
        if not op & 0x80:
            v = op << 8 | byte()
            copy((v >> 4) & 0x7ff or 0x800, v & 15 or 16)
        elif op & 0x40:
            distance = ((op << 8 | byte()) & 0x3fff) or 0x4000
            count = byte()
            step = byte() if count & 0x80 else 0
            copy(distance, count & 0x7f or 128, step)
        else:
            op &= 0x3f
            if op not in range(1, 7):
                raise ValueError(f'invalid LZ77 opcode {op}')
            value = byte() if op >= 3 else None
            step = byte() if op >= 5 else 0
            count = byte()
            if op % 2 == 0:
                count = count << 8 | byte()
                count = count or 65536
            else:
                count = count or 256
            for i in range(count):
                out.append(byte() if value is None else (value + i * step) & 255)
        if len(out) > size:
            raise ValueError('LZ77 output exceeds expected size')
    return bytes(out)


def lz77_encode(raw):
    """Valid lossless literal-block encoder; preserves compression mode 1."""
    out = bytearray()
    for i in range(0, len(raw), 65536):
        block = raw[i:i + 65536]
        out.extend(b'\x82' + (len(block) & 65535).to_bytes(2, 'big') + block)
    return bytes(out)


def p6_decode(src, size):
    out = bytearray(); pos = 0
    while len(out) < size:
        if pos >= len(src):
            raise ValueError('truncated LZ-p6 stream')
        op = src[pos]; pos += 1
        if op < 0x40:
            out.append(op)
        elif op < 0xc0:
            if op < 0x80:
                distance, count = ((op & 63) >> 2) + 1, (op & 3) + 2
            else:
                if pos >= len(src):
                    raise ValueError('truncated LZ-p6 back-reference')
                v = (op & 63) << 8 | src[pos]; pos += 1
                distance, count = (v >> 6) + 1, (v & 63) + 2
            if distance > len(out):
                raise ValueError('LZ-p6 back-reference before output')
            for _ in range(count):
                out.append(out[-distance])
        else:
            for _ in range((op & 15) + 2):
                if pos >= len(src):
                    raise ValueError('truncated LZ-p6 pairs')
                v = src[pos]; pos += 1
                out.extend(((op & 0x30) | v >> 4, (op & 0x30) | v & 15))
        if len(out) > size:
            raise ValueError('LZ-p6 output exceeds expected size')
    return bytes(out), pos


def p6_encode_greedy(pixels):
    """Port of compress_p6 (greedy overlapping refs, pairs, literals)."""
    if any(v >= 64 for v in pixels):
        raise ValueError('LZ-p6 indexes must be below 64')
    out = bytearray(); i = 0; length = len(pixels)
    while i < length:
        best_length = best_distance = 0
        for distance in range(1, min(256, i) + 1):
            match = 0
            while match < 65 and i + match < length and pixels[i + match] == pixels[i - distance + match]:
                match += 1
            if match > best_length:
                best_length, best_distance = match, distance
        pairs = 0
        while (pairs < 17 and i + pairs * 2 + 1 < length and
               pixels[i + pairs * 2] & 0x30 == pixels[i] & 0x30 and
               pixels[i + pairs * 2 + 1] & 0x30 == pixels[i] & 0x30):
            pairs += 1
        short_length = min(best_length, 5)
        short_saving = short_length - 1 if best_distance <= 16 and best_length >= 2 else 0
        long_saving = best_length - 2 if best_length >= 3 else 0
        pairs_saving = pairs - 1 if pairs >= 2 else 0
        if pairs_saving > 0 and pairs_saving >= short_saving and pairs_saving >= long_saving:
            out.append(0xc0 | (pixels[i] & 0x30) | (pairs - 2))
            for k in range(pairs):
                out.append((pixels[i + k * 2] & 15) << 4 | (pixels[i + k * 2 + 1] & 15))
            i += pairs * 2
        elif short_saving > 0 and short_saving >= long_saving:
            out.append(0x40 | (best_distance - 1) << 2 | (short_length - 2))
            i += short_length
        elif long_saving > 0:
            value = (best_distance - 1) << 6 | (best_length - 2)
            out.extend((0x80 | value >> 8, value & 255))
            i += best_length
        else:
            out.append(pixels[i]); i += 1
    return bytes(out)


def p6_encode(pixels):
    """Optimal-parse LZ-p6 encoder (shortest path over literals, short/long refs and pairs).
    Produces the same stream format as compress_p6, typically much smaller than the greedy port."""
    import numpy as np
    px = np.frombuffer(bytes(pixels), dtype=np.uint8)
    if px.size and px.max() >= 64:
        raise ValueError('LZ-p6 indexes must be below 64')
    n = px.size
    if n == 0:
        return b''
    D = min(256, n - 1)
    # run[d-1, i]: how many pixels from i match the pixels d back (overlap allowed), capped at 65
    run = np.zeros((max(D, 1), n + 1), dtype=np.int16)
    if D:
        eq = np.zeros((D, n), dtype=bool)
        for d in range(1, D + 1):
            eq[d - 1, d:] = px[d:] == px[:-d]
        for i in range(n - 1, -1, -1):
            run[:, i] = np.where(eq[:, i], np.minimum(run[:, i + 1] + 1, 65), 0)
    bank = px >> 4 & 3
    same = np.zeros(n + 1, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        same[i] = same[i + 1] + 1 if i + 1 < n and bank[i + 1] == bank[i] else 1
    INF = 1 << 30
    cost = [0] * (n + 1); choice = [None] * n
    for i in range(n - 1, -1, -1):
        best = 1 + cost[i + 1]; ch = ('lit',)
        if D and i:
            r = run[:min(D, i), i]
            dl = int(r.argmax()); Ll = int(r[dl])
            rs = run[:min(16, i), i]
            ds = int(rs.argmax()); Ls = min(5, int(rs[ds]))
            for L in range(2, Ls + 1):
                c = 1 + cost[i + L]
                if c < best: best, ch = c, ('short', ds + 1, L)
            for L in range(2, min(65, Ll) + 1):
                c = 2 + cost[i + L]
                if c < best: best, ch = c, ('long', dl + 1, L)
        maxpairs = min(17, int(same[i]) // 2)
        for k in range(2, maxpairs + 1):
            c = 1 + k + cost[i + 2 * k]
            if c < best: best, ch = c, ('pairs', k)
        cost[i] = best; choice[i] = ch
    out = bytearray(); i = 0
    while i < n:
        ch = choice[i]
        if ch[0] == 'lit':
            out.append(int(px[i])); i += 1
        elif ch[0] == 'short':
            _, d, L = ch; out.append(0x40 | (d - 1) << 2 | (L - 2)); i += L
        elif ch[0] == 'long':
            _, d, L = ch; v = (d - 1) << 6 | (L - 2); out.extend((0x80 | v >> 8, v & 255)); i += L
        else:
            _, k = ch; out.append(0xc0 | (int(px[i]) & 0x30) | (k - 2))
            for j in range(k): out.append((int(px[i + 2 * j]) & 15) << 4 | (int(px[i + 2 * j + 1]) & 15))
            i += 2 * k
    return bytes(out)
