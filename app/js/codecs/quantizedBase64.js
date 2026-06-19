import { buildTrack, dequantizeTrack, normalizeQuaternionValues, normalizeTrackType } from './common.js';

export function canDecodeTrack(type) {
    return type.endsWith('_quantize_b64') || type.endsWith('_b64');
}

export function decodeTrack(trackDef) {
    const baseType = normalizeTrackType(trackDef.type);
    const decoded = dequantizeTrack(trackDef, Uint8Array);

    if (baseType === 'quaternion') {
        normalizeQuaternionValues(decoded.values);
    }

    return buildTrack(baseType, trackDef.name, decoded.times, decoded.values);
}

export const decoder = {
    name: 'quantized_base64',
    canDecodeTrack,
    decodeTrack,
};