# MoCap Viewer App

A browser test bench for visually comparing the original mocap clip against
whatever compressed/encoded variants the Python side (`src/compression/`)
has produced. It loads several versions of the same clip side-by-side onto
cloned rigs so you can eyeball drift introduced by each codec.

## Workflow

```
viewer.html
  -> viewer.js
       -> fetch each CONFIG.clips[i].url            (raw JSON files)
       -> codecs/index.js: compileAnimationClip(json)
            -> if animationClip.tracks exists: pass it through as-is
               (uncompressed source clip)
            -> else: decode blendshape_tracks_data and quaternion_tracks_data
               independently, each via whichever registered decoder's
               canDecode(blob) matches, then concatenate the resulting tracks
            -> THREE.AnimationClip
       -> createRig() clones the GLTF model + skeleton helper per clip
       -> render via controlPanel.js's per-clip toggles/position fields
```

Everything happens client-side; there is no build step. `viewer.html` loads
`three` straight from a CDN via an import map and runs `viewer.js` as a
native ES module.

## Two clip shapes on the wire

See `docs/Compressed_Data_Format.md` for the authoritative field-by-field
reference. In short:

- **Uncompressed clip**: `animationClip.tracks` is a flat list of
  `{ name, type, times, values }`. Handled inline in `index.js`, no decoder
  needed -- there's nothing to pick between.
- **Compressed clip**: `ClipCompressor.compress()` (Python) pops `tracks`
  and replaces it with two independent, self-describing blobs:
  `blendshape_tracks_data` and `quaternion_tracks_data`. Each blob carries
  its own `compression_type` (e.g. `"raw_base64"`, `"direct_quantize"`) so
  the registry can pick the right decoder *per blob*, not per track.

## Codec registry (`app/js/codecs/`)

Each codec is a small module exporting:

```js
export function canDecode(blob) { ... }   // does this module own this blob?
export function decode(blob) { ... }      // -> [{ name, type, times, values }, ...]
export const decoder = { name, canDecode, decode };
```

`index.js` holds the `DECODERS` registry and tries them in order, first
match wins.

| JS module          | Mirrors (Python)                          | Matched via                          |
|---------------------|--------------------------------------------|----------------------------------------|
| `rawBase64.js`      | `compression/baseline.py: RawBase64Compressor` | `blob.compression_type === 'raw_base64'` |
| `quantization.js`   | `compression/baseline.py: QuantizeCompressor`  | `blob.compression_type === 'direct_quantize'` |
| `blendshapes.js`    | `compression/blendshapes.py: BlendShapesSchemeCompressor` | `blob.compress_type === 'blendshapes_scheme'` |
| `common.js`         | shared base64/quantization helpers         | n/a                                     |

Note `blendshapes.js` is matched via `compress_type`, not `compression_type`
-- that's the actual key `BlendShapesSchemeCompressor.compress()` emits on
the Python side (the two baseline compressors use the latter), not a typo.

`quantize()`/`dequantize()` (`src/compression/utils.py`) themselves only
ever work on one flat array with one `scale`/`zero` pair -- `dequantizeCodes`
mirrors that and has no per-axis case. But `QuantizeCompressor` calls that
function differently depending on `type_name`: a `"number"` track gets one
flat `scale`/`zero` over its whole values array, while a `"quaternion"`
track gets *four independent* quantizations, one per `x`/`y`/`z`/`w`
component (`q_values: { x, y, z, w }`, each its own `{ codes_b64, scale,
zero }`). `quantization.js` branches on `blob.type_name` the same way and
interleaves the four decoded component arrays back into `[x,y,z,w,x,y,z,w,...]`
before handing them to `buildTrack`. Bit depth (`q_bits`) is recorded in the
blob, so `codeArrayTypeForBits()` picks the matching typed array instead of
assuming a fixed width.

`QuantizeCompressor` also takes a Python-only `renormalize` constructor flag
(used for quaternion tracks, e.g. `out/quan/*.json`) that isn't serialized
into the blob, so the JS decoder can't read it back out of the data. Instead
`quantization.js` always renormalizes when `blob.type_name === 'quaternion'`,
matching how the notebooks actually use the flag.

`blendshapes.js` has the same kind of gap: the blob's frame indices
(`f_times_b64`) are always encoded with the compressor's constructor default
`frame_code_type` (`np.uint16`), but that dtype is never recorded as `f_bits`
in the blob (unlike `q_bits` for the value codes). The JS decoder hardcodes
`Uint16Array` to match the Python default -- if a clip is ever produced with
a non-default `frame_code_type`, both the docs' proposed `f_bits` field and
this decoder will need updating together.

## Adding a new decoder

1. Implement (and finish) the codec's Python compressor under
   `src/compression/` first -- the JS decoder should always be a mirror of a
   working Python `decompress()`, not a guess.
2. Create `app/js/codecs/<name>.js` exporting `canDecode`/`decode`/`decoder`
   as above. Reuse `common.js` helpers (`decodeBase64ToArray`,
   `dequantizeCodes`, `codeArrayTypeForBits`, `buildTrack`) wherever the wire
   format overlaps with an existing codec.
3. Register it in `codecs/index.js`'s `DECODERS` array.
4. Add its output file to `CONFIG.clips` in `viewer.js` to see it rendered.
