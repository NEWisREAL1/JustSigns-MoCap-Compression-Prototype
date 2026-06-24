import { codeArrayTypeForBits, decodeBase64ToArray, dequantizeCodes } from './common.js';

// Mirrors BlendShapesSchemeCompressor in src/compression/blendshapes.py.
// Unlike the baseline codecs, a single blob fans out into *multiple* tracks
// per group (`group.names`), all sharing the same decoded `times`/`values`.
//
// Frame indices (`f_times_b64`) are not currently self-describing -- the
// Python side never serializes `f_bits`/the frame code dtype into the blob,
// it just always encodes with its constructor default `frame_code_type`
// (np.uint16). Matched here as a hardcoded Uint16Array for the same reason.

const FRAME_CODE_TYPE = Uint16Array;

export function canDecode(blob) {
    return blob.compress_type === 'blendshapes_scheme';
}

export function decode(blob) {
    const CodeType = codeArrayTypeForBits(blob.q_bits);
    const tracks = [];

    for (const group of blob.groups) {
        const frameCodes = decodeBase64ToArray(group.f_times_b64, FRAME_CODE_TYPE);
        // Plain `.map` on a typed array re-truncates the division back to
        // the integer element type, so accumulate into a Float64Array instead.
        const times = new Float64Array(frameCodes.length);
        for (let i = 0; i < frameCodes.length; i++) {
            times[i] = frameCodes[i] / blob.f_fps;
        }

        const valueCodes = decodeBase64ToArray(group.q_values.codes_b64, CodeType);
        const values = dequantizeCodes(valueCodes, group.q_values.scale, group.q_values.zero);

        for (const name of group.names) {
            tracks.push({ name, type: blob.type_name, times, values });
        }
    }

    return tracks;
}

export const decoder = {
    name: 'blendshapes_scheme',
    canDecode,
    decode,
};
