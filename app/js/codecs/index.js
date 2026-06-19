import * as THREE from 'three';

import { decoder as plaintextDecoder } from './plaintext.js';
import { decoder as rawBase64Decoder } from './rawBase64.js';
import { decoder as quantizedBase64Decoder } from './quantizedBase64.js';

export const DECODERS = [
    plaintextDecoder,
    rawBase64Decoder,
    quantizedBase64Decoder,
];

function decodeTrack(trackDef) {
    const decoder = DECODERS.find(entry => entry.canDecodeTrack(trackDef.type));

    if (!decoder) {
        return null;
    }

    return decoder.decodeTrack(trackDef);
}

export function compileAnimationClip(jsonData) {
    const animData = jsonData.animationClip;
    const tracks = animData.tracks.map(decodeTrack).filter(Boolean);

    return new THREE.AnimationClip(animData.name, animData.duration, tracks);
}