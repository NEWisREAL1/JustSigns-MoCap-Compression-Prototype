# Motion Capture Data Compression for JustSings Project

Fitting the joints motion with splines to get parametric representation of the motion with less data to store (sequences of vectors -> sets of control points, time-parameters, knot vectors, etc.). Experimental result shows that further/more advance compression techniques should be apply to handle the complicated high frequency motion in the JustSigns dataset (as noted below).

# Current Approach — Hierarchical Approximation and High-Frequency Residuals Compression (HA-HFR Compression)


## 1. Hierarchical B-Spline Approximation for Low-Frequency Motions

B-spline acts as a impressive compressor, using LSPIA to fit the spline to the motion (position vectors/quaternions) gives parametric representation of the motion with far less number of variables to store than the keyframed data. Various techniques can be apply to the LSPIA process to get a better approximation of the motion.

In terms of data loss, B-spline works well only for low-frequency motions (since B-splines naturally acts as a low-pass filter), large number of control points is needed if one want to perfectly fit the high-frequency motions to the spline. Thus, the high-frequency residual

see [docs/BSpline_Approximation.md] for implementation details

## 2. High-Frequency Residuals Compression

For motions/parts of motions that have high-frequency nature, B-spline will likely fail to make a good approximation of it. To get a better approximation, we can compute the fitting residuals between the approximate spline and the real motion and compress those residuals instead for reconstruction. Approaches like discrete cosine transform (DCT), framerate dropping or quantization can be apply to compress such residuals data.

see [docs/HighFrequency_Residuals_Compression.md] for implementation details