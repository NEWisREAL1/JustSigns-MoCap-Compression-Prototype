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

export function normalizeTrackType(type) {
    return type.replace(/_(raw_b64|quantize_b64|bspline_b64|b64)$/, '');
}

export function buildTrack(baseType, name, times, values) {
    if (baseType === 'quaternion') return new THREE.QuaternionKeyframeTrack(name, times, values);
    if (baseType === 'number') return new THREE.NumberKeyframeTrack(name, times, values);
    if (baseType === 'vector') return new THREE.VectorKeyframeTrack(name, times, values);
    return null;
}

/**
 * Dequantize a flat array of integer codes back to floats.
 * Mirrors QuantizeCompressor._dequantize in src/compressors/quantization.py:
 * `scale`/`zero` are either a single scalar (1-D data, e.g. times) or one
 * value per trailing axis (e.g. one per quaternion component).
 */
export function dequantizeCodes(codes, scale, zero) {
    const scales = Array.isArray(scale) ? scale : [scale];
    const zeros = Array.isArray(zero) ? zero : [zero];
    const dimension = scales.length;
    const values = new Float32Array(codes.length);

    for (let i = 0; i < codes.length; i++) {
        const axis = dimension === 1 ? 0 : i % dimension;
        values[i] = scales[axis] * (codes[i] - zeros[axis]);
    }

    return values;
}

/**
 * Decode + dequantize a `{ codes_b64, scale, zero }` node, the shape every
 * quantized field (times, values, control_pts, ...) is packed in.
 */
export function decodeQuantizedNode(node, CodeType) {
    const codes = decodeBase64ToArray(node.codes_b64, CodeType);
    return dequantizeCodes(codes, node.scale, node.zero);
}

export function normalizeQuaternionValues(values) {
    for (let i = 0; i < values.length; i += 4) {
        const x = values[i];
        const y = values[i + 1];
        const z = values[i + 2];
        const w = values[i + 3];
        const norm = Math.sqrt(x * x + y * y + z * z + w * w) + 1e-8;
        values[i] /= norm;
        values[i + 1] /= norm;
        values[i + 2] /= norm;
        values[i + 3] /= norm;
    }

    return values;
}