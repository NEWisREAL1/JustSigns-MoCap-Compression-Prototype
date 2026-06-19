import { buildTrack, decodeBase64ToArray, normalizeTrackType } from './common.js';

export function canDecodeTrack(type) {
    return type.endsWith('_raw_b64');
}

export function decodeTrack(trackDef) {
    const baseType = normalizeTrackType(trackDef.type);
    const times = decodeBase64ToArray(trackDef.times, Float64Array);
    const values = decodeBase64ToArray(trackDef.values, Float64Array);
    return buildTrack(baseType, trackDef.name, times, values);
}

export const decoder = {
    name: 'raw_base64',
    canDecodeTrack,
    decodeTrack,
};