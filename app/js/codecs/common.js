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
    return type.replace(/_(raw_b64|quantize_b64|b64)$/, '');
}

export function buildTrack(baseType, name, times, values) {
    if (baseType === 'quaternion') return new THREE.QuaternionKeyframeTrack(name, times, values);
    if (baseType === 'number') return new THREE.NumberKeyframeTrack(name, times, values);
    if (baseType === 'vector') return new THREE.VectorKeyframeTrack(name, times, values);
    return null;
}

export function dequantizeTrack(trackDef, DataType) {
    const codesTimes = decodeBase64ToArray(trackDef.times.codes_b64, DataType);
    const codesValues = decodeBase64ToArray(trackDef.values.codes_b64, DataType);

    const keyCount = codesTimes.length;
    const dimension = codesValues.length / keyCount;

    const times = new Float32Array(keyCount);
    const values = new Float32Array(keyCount * dimension);

    const timeScale = Array.isArray(trackDef.times.scale) ? trackDef.times.scale : [trackDef.times.scale];
    const timeZero = Array.isArray(trackDef.times.zero) ? trackDef.times.zero : [trackDef.times.zero];
    const valueScale = Array.isArray(trackDef.values.scale) ? trackDef.values.scale : [trackDef.values.scale];
    const valueZero = Array.isArray(trackDef.values.zero) ? trackDef.values.zero : [trackDef.values.zero];

    for (let i = 0; i < keyCount; i++) {
        times[i] = timeScale[0] * (codesTimes[i] - timeZero[0]);
        for (let d = 0; d < dimension; d++) {
            const idx = i * dimension + d;
            values[idx] = (valueScale.length > 1 ? valueScale[d] : valueScale[0]) * (codesValues[idx] - (valueZero.length > 1 ? valueZero[d] : valueZero[0]));
        }
    }

    return { times, values };
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