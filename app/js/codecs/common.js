import * as THREE from 'three';

export function decodeBase64ToArray(b64String, ArrayType) {
    const binaryString = atob(b64String);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);

    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    return new ArrayType(bytes.buffer);
}

export function buildTrack(baseType, name, times, values) {
    if (baseType === 'quaternion') return new THREE.QuaternionKeyframeTrack(name, times, values);
    if (baseType === 'number') return new THREE.NumberKeyframeTrack(name, times, values);
    if (baseType === 'vector') return new THREE.VectorKeyframeTrack(name, times, values);
    return null;
}

/**
 * Maps a quantizer's stored bit depth (`q_bits` in the wire format) to the
 * typed array that can hold its codes. Mirrors how src/compression/utils.py
 * `quantize()` picks `2 ** num_bits - 1` as its code range.
 */
export function codeArrayTypeForBits(bits) {
    if (bits === 8) return Uint8Array;
    if (bits === 16) return Uint16Array;
    if (bits === 32) return Uint32Array;
    throw new Error(`Unsupported quantization bit depth: ${bits}`);
}

/**
 * Mirrors src/compression/utils.py: dequantize(codes, scale, zero).
 * scale/zero are always plain scalars in the current format -- quantize()
 * reduces the whole flattened array to a single min/max, even for
 * multi-component data like quaternions, so there's no per-axis case here.
 */
export function dequantizeCodes(codes, scale, zero) {
    const values = new Float32Array(codes.length);
    for (let i = 0; i < codes.length; i++) {
        values[i] = scale * (codes[i] - zero);
    }
    return values;
}
