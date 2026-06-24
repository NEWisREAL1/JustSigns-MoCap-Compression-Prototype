import * as THREE from 'three';

import { buildTrack } from './common.js';
import { decoder as rawBase64Decoder } from './rawBase64.js';
import { decoder as quantizationDecoder } from './quantization.js';
import { decoder as blendshapesDecoder } from './blendshapes.js';

// Tried in order; the first decoder whose canDecode(blob) matches wins.
// To add a new compression scheme: write a module exporting
// { name, canDecode, decode } (see rawBase64.js/quantization.js for the
// shape) and list it here.
export const DECODERS = [
    rawBase64Decoder,
    quantizationDecoder,
    blendshapesDecoder,
];

function decodeBlob(blobData) {
    if (!blobData) return [];

    const decoder = DECODERS.find(entry => entry.canDecode(blobData));
    if (!decoder) {
        console.warn('No decoder registered for tracks_data blob:', blobData);
        return [];
    }

    return decoder.decode(blobData);
}

export function compileAnimationClip(jsonData) {
    const animData = jsonData.animationClip;

    // Two shapes on the wire (see docs/Compressed_Data_Format.md):
    // - an uncompressed clip still carries a flat `tracks` list.
    // - a compressed clip (ClipCompressor.compress) pops `tracks` and
    //   replaces it with `blendshape_tracks_data` / `quaternion_tracks_data`,
    //   each a self-describing blob naming its own compression scheme.
    const rawTracks = animData.tracks
        ? animData.tracks.map(track => ({ name: track.name, type: track.type, times: track.times, values: track.values }))
        : [...decodeBlob(animData.blendshape_tracks_data), ...decodeBlob(animData.quaternion_tracks_data)];

    const tracks = rawTracks
        .map(track => buildTrack(track.type, track.name, track.times, track.values))
        .filter(Boolean);

    return new THREE.AnimationClip(animData.name, animData.duration, tracks);
}
