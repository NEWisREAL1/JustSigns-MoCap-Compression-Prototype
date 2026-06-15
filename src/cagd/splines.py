import numpy as np

from src.cagd.basis import bspline_basis


class StationarySpline:
    """
    A spline that will evaluate to a single value regradless of input parameter,
    used for convenience compatability
    """

    def __init__(self, singularity):
        self.singularity = np.asarray(singularity)

        # dummy attributes (for compatability with BSpline)
        self.degree = 0
        self.control_pts = np.array([singularity], dtype=np.float64)
        self.knot_vector = np.array([], dtype=np.float64)

    def __call__(self, t):
        t_arr = np.atleast_1d(t)
        res = np.repeat([self.singularity], len(np.atleast_1d(t_arr)), axis=0)
        
        if np.isscalar(t) or np.ndim(t) == 0:
            return res[0]
        return res


class BSpline:
    """
    Wrapper for B-Spline logics and variables
    """

    def __init__(self, degree, control_pts, knot_vector):
        self.degree = degree
        self.control_pts = np.array(control_pts, dtype=np.float64)
        self.knot_vector = np.array(knot_vector, dtype=np.float64)
        self._validate()

    def __call__(self, t):
        t_arr = np.atleast_1d(t)
        n = len(self.control_pts)
        
        pts = []
        for i, cps in enumerate(self.control_pts):
            b = bspline_basis(t_arr, i, self.degree, self.knot_vector, n=n)
            b = b.reshape(b.shape + (1,) * np.ndim(cps))
            pts.append(b * cps)
            
        res = np.sum(pts, axis=0)
        
        if np.isscalar(t) or np.ndim(t) == 0:
            return res[0]
        return res

    def _validate(self):
        p = self.degree
        n = len(self.control_pts)
        m = len(self.knot_vector)

        if m != p + n + 1:
            raise ValueError(
                f"Number of knots (m), control points (n), and curve degree (p) must satisfy m = p + n + 1, got p={p}, n={n}, m={m}"
                )

