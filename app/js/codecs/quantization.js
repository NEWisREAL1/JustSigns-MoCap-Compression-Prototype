import { codeArrayTypeForBits, decodeBase64ToArray, dequantizeCodes } from './common.js';

// Mirrors QuantizeCompressor in src/compression/baseline.py. For quaternion
// tracks each component (x/y/z/w) is quantized independently with its own
// scale/zero; everything else (e.g. blendshape "number" tracks) still uses
// one flat scale/zero over the whole values array.
//
// `renormalize` is a constructor-only flag on the Python side -- it isn't
// serialized into the blob, so it can't be read back out of the data. The
// JS side instead applies the same convention the notebooks use: only
// quaternion tracks get renormalized after dequantizing.

const QUATERNION_AXES = ['x', 'y', 'z', 'w'];

export function canDecode(blob) {
    return blob.compression_type === 'direct_quantize';
}

function decodeQuaternionValues(qValues, CodeType) {
    const axisValues = QUATERNION_AXES.map(axis => {
        const codes = decodeBase64ToArray(qValues[axis].codes_b64, CodeType);
        return dequantizeCodes(codes, qValues[axis].scale, qValues[axis].zero);
    });

    const keyCount = axisValues[0].length;
    const values = new Float32Array(keyCount * 4);

    for (let i = 0; i < keyCount; i++) {
        for (let axis = 0; axis < 4; axis++) {
            values[i * 4 + axis] = axisValues[axis][i];
        }
    }

    return values;
}

function renormalizeQuaternionValues(values) {
    for (let i = 0; i < values.length; i += 4) {
        const norm = Math.hypot(values[i], values[i + 1], values[i + 2], values[i + 3]);
        values[i] /= norm;
        values[i + 1] /= norm;
        values[i + 2] /= norm;
        values[i + 3] /= norm;
    }
}

export function decode(blob) {
    const CodeType = codeArrayTypeForBits(blob.q_bits);
    const isQuaternion = blob.type_name === 'quaternion';

    return blob.tracks.map(track => {
        const timeCodes = decodeBase64ToArray(track.q_times.codes_b64, CodeType);
        const times = dequantizeCodes(timeCodes, track.q_times.scale, track.q_times.zero);

        let values;
        if (isQuaternion) {
            values = decodeQuaternionValues(track.q_values, CodeType);
            renormalizeQuaternionValues(values);
        } else {
            const valueCodes = decodeBase64ToArray(track.q_values.codes_b64, CodeType);
            values = dequantizeCodes(valueCodes, track.q_values.scale, track.q_values.zero);
        }

        return {
            name: track.name,
            type: blob.type_name,
            times,
            values,
        };
    });
}

export const decoder = {
    name: 'direct_quantize',
    canDecode,
    decode,
};
