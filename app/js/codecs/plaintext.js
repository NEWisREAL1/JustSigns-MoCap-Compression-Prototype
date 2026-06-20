import { buildTrack } from './common.js';

const LEGACY_TYPES = new Set(['quaternion', 'number', 'vector']);

export function canDecodeTrack(type) {
    return LEGACY_TYPES.has(type);
}

export function decodeTrack(trackDef, globalAttrs) {
    return buildTrack(
        trackDef.type,
        trackDef.name,
        new Float32Array(trackDef.times),
        new Float32Array(trackDef.values)
    );
}

export const decoder = {
    name: 'plaintext',
    canDecodeTrack,
    decodeTrack,
};