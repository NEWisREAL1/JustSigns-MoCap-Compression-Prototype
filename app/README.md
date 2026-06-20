# MoCap Viewer App

A browser test bench for visually comparing the original mocap clip against
whatever compressed/encoded variants the Python side (`src/compressors/`)
has produced. It loads several versions of the same clip side-by-side onto
cloned rigs so you can eyeball drift introduced by each codec.

## Workflow

```
viewer.html
  -> viewer.js
       -> fetch each CONFIG.clips[i].url            (raw JSON files)
       -> codecs/index.js: compileAnimationClip(json)
            -> for each track in animationClip.tracks:
                 - pick the decoder whose canDecodeTrack(track.type) is true
                 - pick that track's "<basetype>_global_attrs" block, if any
                 - decoder.decodeTrack(track, globalAttrs) -> THREE.KeyframeTrack
            -> THREE.AnimationClip
       -> clone the GLTF rig per clip, attach an AnimationMixer + the clip
       -> render: one mesh row + one skeleton-only row, side by side
```

Everything happens client-side; there is no build step. `viewer.html` loads
`three` straight from a CDN via an import map and runs `viewer.js` as a
native ES module.

## Codec registry (`app/js/codecs/`)

Each codec is a small module exporting:

```js
export function canDecodeTrack(type) { ... }   // does this module own `type`?
export function decodeTrack(trackDef, globalAttrs) { ... } // -> THREE.KeyframeTrack
export const decoder = { name, canDecodeTrack, decodeTrack };
```

`index.js` holds the `DECODERS` registry and tries them in order, first match
wins (by `track.type` suffix, e.g. `..._raw_b64`, `..._quantize_b64`,
`..._bspline_b64`). `index.js` is also responsible for handing each track the
right global-attrs block: `ClipCompressor` (Python) stores one
`number_global_attrs` and one `quaternion_global_attrs` per clip, so the
registry resolves a track's base type first and looks up
`<basetype>_global_attrs` before calling the decoder.

| JS module             | Mirrors (Python)                          | Track type suffix   |
|------------------------|--------------------------------------------|----------------------|
| `plaintext.js`         | n/a -- uncompressed source clips            | `quaternion`, `number`, `vector` (no suffix) |
| `rawBase64.js`         | `compressors/raw_base64.py`                 | `_raw_b64`           |
| `quantization.js`      | `compressors/quantization.py`               | `_quantize_b64`      |
| `bspline.js`           | `compressors/bspline.py` + `cagd/*`         | `_bspline_b64`       |
| `common.js`            | shared base64/quantization helpers          | n/a                  |
| `splineMath.js`        | `cagd/splines.py` + `cagd/lspia.py` (eval-only half) | n/a       |

`splineMath.js` only implements curve *evaluation* (basis functions, knot
regeneration) -- never fitting. The bspline codec stores control points but
not the knot vector; it's regenerated deterministically from
`(degree, num_cps, frame_count)`, exactly like `LSPIASolver._init_knot_vector`
does on the Python side.

## Adding a new decoder

1. Add the codec's Python compressor under `src/compressors/` (already done,
   that part isn't this app's concern).
2. Create `app/js/codecs/<name>.js` exporting `canDecodeTrack`/`decodeTrack`/
   `decoder` as above. Reuse `common.js` helpers (`decodeQuantizedNode`,
   `buildTrack`, `normalizeQuaternionValues`, ...) wherever the wire format
   overlaps with an existing codec.
3. Register it in `codecs/index.js`'s `DECODERS` array.
4. Add its output file to `CONFIG.clips` in `viewer.js` to see it rendered.

## Known constraints worth knowing about

- `quantization.js` assumes 8-bit codes (`Uint8Array`) because the bit depth
  used during encoding isn't recorded in the JSON -- it has to match whatever
  `QuantizeCompressor(base_type=...)` was actually used in Python.
- `bspline.js` assumes every track of a clip shares one timing scheme
  (same as `FixedSizeBSplineCompressor`'s own assumption) -- times come from
  the clip-level `global_times`, not per-track.
