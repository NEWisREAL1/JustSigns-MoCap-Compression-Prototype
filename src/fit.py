import numpy as np
from scipy.interpolate import make_lsq_spline

#
# B-Spline Utilities 
#

def generate_open_knots(degree, num_cps):
    if num_cps < degree + 1:
        raise ValueError(f"num_cps have to be >= degree + 1, got degree={degree} and num_cps={num_cps}")
    
    num_interior = num_cps - degree - 1

    head = np.repeat(0, degree + 1)
    tail = np.repeat(1, degree + 1)
    interior = np.linspace(0, 1, num_interior + 2)[1:-1]

    return np.concatenate([head, interior, tail])

def generate_arc_length_params(pts):
    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    cumu_dists = np.cumsum(step_dists)
    cumu_dists = np.concatenate(([0], cumu_dists))
    return cumu_dists / cumu_dists[-1]

def generate_params(pts, power=0.5):    # power=1 arc-length, 0.5 centripetal, 0 uniform
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) ** power
    d = np.concatenate(([0], np.cumsum(seg)))
    return d / d[-1]

#
# Curve/Spline Fitting
#

def lsq_bspline_fit(traj, degree, num_cps, params_pow=.5):
    knots = generate_open_knots(degree, num_cps)
    us = generate_params(traj, params_pow)
    return make_lsq_spline(us, traj, knots, degree), us, knots