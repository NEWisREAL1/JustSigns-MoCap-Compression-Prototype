/**
 * B-Spline evaluation math used to decode the bspline codec.
 * Mirrors src/cagd/splines.py (BSpline) and the knot-regeneration half of
 * src/cagd/lspia.py (LSPIASolver). Only what's needed to *evaluate* an
 * already-fitted curve lives here -- the LSPIA fitting itself is Python-only.
 */

export function linspace(start, end, count) {
    if (count <= 1) return new Float64Array([start]);

    const values = new Float64Array(count);
    const step = (end - start) / (count - 1);
    for (let i = 0; i < count; i++) values[i] = start + step * i;

    return values;
}

// Cox-de Boor recursion for one time sample. Mirrors BSpline.basis.
function basis(t, i, degree, knotVector, numControlPoints) {
    let res;

    if (degree === 0) {
        res = (knotVector[i] <= t && t < knotVector[i + 1]) ? 1 : 0;
    } else {
        const leftDenom = knotVector[i + degree] - knotVector[i];
        const rightDenom = knotVector[i + degree + 1] - knotVector[i + 1];
        const leftWeight = leftDenom !== 0 ? (t - knotVector[i]) / leftDenom : 0;
        const rightWeight = rightDenom !== 0 ? (knotVector[i + degree + 1] - t) / rightDenom : 0;

        res = leftWeight * basis(t, i, degree - 1, knotVector, numControlPoints)
            + rightWeight * basis(t, i + 1, degree - 1, knotVector, numControlPoints);
    }

    // Open knot vectors make every basis function vanish at the curve's end
    // except the last control point's -- correct for that at every degree,
    // not just the degree-0 base case (the recursion alone undercounts it).
    if (Math.abs(t - knotVector[knotVector.length - 1]) < 1e-9) {
        res = i === numControlPoints - 1 ? 1 : 0;
    }

    return res;
}

// Mirrors BSpline.generate_uniform_open_knots.
export function generateUniformOpenKnots(degree, numControlPoints) {
    const numInternalKnots = numControlPoints - degree - 1;
    const numExternalKnots = degree + 1;

    const head = new Array(numExternalKnots - 1).fill(0);
    const tail = new Array(numExternalKnots - 1).fill(1);
    const internal = Array.from(linspace(0, 1, numInternalKnots + 2));

    return [...head, ...internal, ...tail];
}

/**
 * Mirrors LSPIASolver._init_knot_vector ("time-averaging" adjustment).
 * The fitted knot vector is never transmitted -- it's fully determined by
 * (degree, numControlPoints, dataTimes), so the decoder regenerates it
 * deterministically instead of storing/streaming it.
 */
export function regenerateKnotVector(degree, numControlPoints, dataTimes) {
    const knotVector = generateUniformOpenKnots(degree, numControlPoints);
    const numInternal = numControlPoints - degree - 1;
    const d = dataTimes.length / (numControlPoints - degree);

    for (let j = 0; j < numInternal; j++) {
        const mathJ = j + 1;
        const knotIdx = mathJ + degree;
        const startIdx = Math.trunc(mathJ * d);

        let sum = 0;
        for (let k = 0; k < degree; k++) sum += dataTimes[startIdx + k];
        knotVector[knotIdx] = sum / degree;
    }

    return knotVector;
}

// Mirrors BSpline.__call__: evaluates the curve at every sample time.
export function evaluateBSpline(degree, controlPoints, knotVector, sampleTimes) {
    const numControlPoints = controlPoints.length;
    const dimension = controlPoints[0].length;
    const output = new Float32Array(sampleTimes.length * dimension);

    for (let s = 0; s < sampleTimes.length; s++) {
        const t = sampleTimes[s];

        for (let c = 0; c < numControlPoints; c++) {
            const weight = basis(t, c, degree, knotVector, numControlPoints);
            if (weight === 0) continue;

            for (let d = 0; d < dimension; d++) {
                output[s * dimension + d] += weight * controlPoints[c][d];
            }
        }
    }

    return output;
}
