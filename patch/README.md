# Patch files

The PUMP FIGHTERS patch is attached to each [GitHub Release](https://github.com/MTG9T4/pump-fighters/releases), not committed here
(about 54 MB):

- `pumpfighters.json`: the manifest (sizes and SHA-256 hashes of the original and modded archive, and of
  every changed entry)
- `pumpfighters-1.deltas`, `pumpfighters-2.deltas`, ...: one VCDIFF delta (RFC 3284) per changed entry

Download all of them into this folder, then:

```
node patcher/patch.mjs <your SF33RD.AFS> <out SF33RD.AFS>
```

or point at them from anywhere with `--patch path/to/pumpfighters.json`.

The deltas only work against the original `SF33RD.AFS` (PS2 US v1.0, 642,492,416 bytes, SHA-256
`f9fa50f3a124ec9fa9465aa9c8546c2d867887eb39f711a070762a0324ba5604`). They contain the mod's changes, not
the game.
