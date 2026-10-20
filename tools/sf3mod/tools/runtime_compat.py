"""Archive-side compatibility with the original ports' in-place loader fixups."""
import struct

from spritebank import BANKS, SpriteBank

AKUMA_MARKER = b'PC_AKUMA166_V1\0'


def prepare_character_bank(name, data):
    """Compensate for texgroup.c's Akuma tile deletion, without editing the renderer.

    Only rebuilt, relocated lists need compensation. Stock lists expect the stock
    fixup. The duplicated first tile also draws identically when a port's arcade
    balance mode skips the fixup (both copies have the same position and pixels).
    """
    if name.lower() != 'pl14.bin' or data.endswith(AKUMA_MARKER):
        return data
    info = BANKS['pl14.bin']
    bank = SpriteBank(data, info)
    if bank.pattern_offsets[166] < info['size']:
        return data
    tiles = bank.entries(166)
    if not tiles:
        raise ValueError('cannot compensate an empty Akuma pattern 166 for the loader fixup')
    x, y, attr, code = tiles[0]
    original = [tiles[0], (0, 0, attr, code), *tiles[1:]]
    if len(original) > 32767:
        raise ValueError('Akuma pattern 166 exceeds the signed loader tile-count limit')
    result = bytearray(data)
    result.extend(b'\0' * (-len(result) % 4))
    struct.pack_into('<I', result, 166 * 4, len(result))
    result.extend(struct.pack('<H', len(original)))
    for tile in original:
        result.extend(struct.pack('<hhHH', *tile))
    result.extend(AKUMA_MARKER)
    return bytes(result)
