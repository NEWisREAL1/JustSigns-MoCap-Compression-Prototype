import { buildTrack, decodeQuantizedNode, normalizeQuaternionValues, normalizeTrackType } from './common.js';
import { evaluateBSpline, linspace, regenerateKnotVector } from './splineMath.js';

// Mirrors FixedSizeBSplineCompressor in src/compressors/bspline.py: control
// points are 16-bit quantized, the shared keyframe times are 8-bit quantized.
const CONTROL_POINT_CODE_TYPE = Uint16Array;
const TIME_CODE_TYPE = Uint8Array;

function dimensionOf(baseType) {
    if (baseType === 'quaternion') return 4;
    if (baseType === 'number') return 1;
    return 3; // unused by any current Python compressor, kept for forward-compat
}

function reshape(flat, dimension) {
    const points = [];
    for (let i = 0; i < flat.length; i += dimension) {
        points.push(Array.from(flat.slice(i, i + dimension)));
    }
    return points;
}

export function canDecodeTrack(type) {
    return type.endsWith('_bspline_b64');
}

/**
 * `globalAttrs` is the clip's `<basetype>_global_attrs` block (degree,
 * num_cps, frame_count, global_times) that FixedSizeBSplineCompressor.compress
 * writes once per clip -- every track of that base type shares it, since the
 * compressor assumes one timing scheme for all tracks.
 */
export function decodeTrack(trackDef, globalAttrs) {
    if (!globalAttrs) {
        throw new Error(`bspline track "${trackDef.name}" has no matching *_global_attrs block (degree/num_cps/frame_count/global_times)`);
    }

    const baseType = normalizeTrackType(trackDef.type);
    const dimension = dimensionOf(baseType);
    const { degree, num_cps: numControlPoints, frame_count: frameCount, global_times: globalTimes } = globalAttrs;

    // The knot vector is never stored on the wire -- it's fully determined by
    // (degree, numControlPoints, dataTimes), so it's regenerated here exactly
    // as LSPIASolver._init_knot_vector does during fitting.
    const dataTimes = linspace(0, 1, frameCount);
    const knotVector = regenerateKnotVector(degree, numControlPoints, dataTimes);
    const times = decodeQuantizedNode(globalTimes, TIME_CODE_TYPE);

    const controlPointCodes = decodeQuantizedNode(trackDef.values.control_pts, CONTROL_POINT_CODE_TYPE);
    const controlPoints = reshape(controlPointCodes, dimension);

    const values = evaluateBSpline(degree, controlPoints, knotVector, dataTimes);

    if (baseType === 'quaternion') {
        normalizeQuaternionValues(values);
    }

    return buildTrack(baseType, trackDef.name, times, values);
}

export const decoder = {
    name: 'bspline',
    canDecodeTrack,
    decodeTrack,
};
