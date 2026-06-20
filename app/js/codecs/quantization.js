import { buildTrack, decodeQuantizedNode, normalizeQuaternionValues, normalizeTrackType } from './common.js';

// QuantizeCompressor's base_type defaults to (and is always used as) np.uint8
// in every Python pipeline so far. The dtype isn't recorded on the wire, so
// this must match whatever bit depth the encoder used -- bump it here if a
// future pipeline quantizes to a different width.
const CODE_TYPE = Uint8Array;

export function canDecodeTrack(type) {
    return type.endsWith('_quantize_b64');
}

export function decodeTrack(trackDef) {
    const baseType = normalizeTrackType(trackDef.type);
    const times = decodeQuantizedNode(trackDef.times, CODE_TYPE);
    const values = decodeQuantizedNode(trackDef.values, CODE_TYPE);

    if (baseType === 'quaternion') {
        normalizeQuaternionValues(values);
    }

    return buildTrack(baseType, trackDef.name, times, values);
}

export const decoder = {
    name: 'quantization',
    canDecodeTrack,
    decodeTrack,
};
