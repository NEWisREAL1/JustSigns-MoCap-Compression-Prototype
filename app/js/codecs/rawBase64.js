import { decodeBase64ToArray } from './common.js';

// Mirrors RawBase64Compressor in src/compression/baseline.py. Unlike the old
// per-track-type-suffix scheme, the new format decides the codec once per
// `*_tracks_data` blob via `compression_type`, not per individual track.

export function canDecode(blob) {
    return blob.compression_type === 'raw_base64';
}

export function decode(blob) {
    return blob.tracks.map(track => ({
        name: track.name,
        type: blob.type_name,
        times: decodeBase64ToArray(track.times_b64, Float64Array),
        values: decodeBase64ToArray(track.values_b64, Float64Array),
    }));
}

export const decoder = {
    name: 'raw_base64',
    canDecode,
    decode,
};
