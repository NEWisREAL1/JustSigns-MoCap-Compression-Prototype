import * as THREE from 'three';

import { normalizeTrackType } from './common.js';
import { decoder as plaintextDecoder } from './plaintext.js';
import { decoder as rawBase64Decoder } from './rawBase64.js';
import { decoder as quantizationDecoder } from './quantization.js';
import { decoder as bsplineDecoder } from './bspline.js';

// Tried in order; the first decoder whose canDecodeTrack(type) matches wins.
// To add a new codec: write a module exporting { name, canDecodeTrack,
// decodeTrack } (see any file above for the shape) and list it here.
export const DECODERS = [
    plaintextDecoder,
    rawBase64Decoder,
    quantizationDecoder,
    bsplineDecoder,
];

function decodeTrack(trackDef, animData) {
    const decoder = DECODERS.find(entry => entry.canDecodeTrack(trackDef.type));

    if (!decoder) {
        console.warn(`No decoder registered for track type "${trackDef.type}"`);
        return null;
    }

    // ClipCompressor stores per-base-type global attrs (e.g. degree/num_cps
    // for bspline-compressed quaternions) under "<basetype>_global_attrs" --
    // route each track to its own block rather than assuming "quaternion".
    const baseType = normalizeTrackType(trackDef.type);
    const globalAttrs = animData[`${baseType}_global_attrs`] ?? null;

    return decoder.decodeTrack(trackDef, globalAttrs);
}

export function compileAnimationClip(jsonData) {
    const animData = jsonData.animationClip;
    const tracks = animData.tracks.map(trackDef => decodeTrack(trackDef, animData)).filter(Boolean);

    return new THREE.AnimationClip(animData.name, animData.duration, tracks);
}