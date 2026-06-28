import numpy as np


class StationarySpline:
    """
    A spline that will evaluate to a single value regradless of input parameter,
    used for convenience compatability
    """

    def __init__(self, singularity):
        self.singularity = np.asarray(singularity)

        # dummy attributes (for compatability with BSpline)
        self.degree = 0
        self.control_points = np.array([singularity], dtype=np.float64)
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

    def __init__(self, degree, control_points, knot_vector):
        self.degree = degree
        self.control_points = np.array(control_points, dtype=np.float64)
        self.knot_vector = np.array(knot_vector, dtype=np.float64)
        self._validate()


    def __call__(self, t):
        t_arr = np.atleast_1d(t)
        n = len(self.control_points)
        
        pts = []
        for i, cps in enumerate(self.control_points):
            b = BSpline.basis(t_arr, i, self.degree, self.knot_vector, n=n)
            b = b.reshape(b.shape + (1,) * np.ndim(cps))
            pts.append(b * cps)
            
        res = np.sum(pts, axis=0)
        
        if np.isscalar(t) or np.ndim(t) == 0:
            return res[0]
        return res


    def _validate(self):
        p = self.degree
        n = len(self.control_points)
        m = len(self.knot_vector)

        if m != p + n + 1:
            raise ValueError(
                f"Number of knots (m), control points (n), and curve degree (p) must satisfy m = p + n + 1, got p={p}, n={n}, m={m}"
                )

    
    @staticmethod
    def basis(t, i, p, knot_vector, n=None):
            """
            Calculate B-Spline basis via Cox-de Boor recursion
            """
            if n is None:
                n = len(knot_vector) - p - 1

            t_arr = np.asarray(t)

            if p == 0:
                # Vectorized condition: knot_vector[i] <= t_arr < knot_vector[i + 1]
                res = ((knot_vector[i] <= t_arr) & (t_arr < knot_vector[i + 1])).astype(np.float64)
            else:
                # denominators
                d1 = t_arr - knot_vector[i]
                d2 = knot_vector[i + p + 1] - t_arr
                
                # numerators
                n1 = knot_vector[i + p] - knot_vector[i]
                n2 = knot_vector[i + p + 1] - knot_vector[i + 1]
                
                # final weights
                w1 = d1 / n1 if n1 != 0 else np.zeros_like(t_arr)
                w2 = d2 / n2 if n2 != 0 else np.zeros_like(t_arr)

                b1 = BSpline.basis(t_arr, i, p - 1, knot_vector, n=n)
                b2 = BSpline.basis(t_arr, i + 1, p - 1, knot_vector, n=n)

                res = w1 * b1 + w2 * b2

            # Vectorized check for the end-knot boundary condition
            end_knot_mask = np.isclose(t_arr, knot_vector[-1])
            if np.any(end_knot_mask):
                end_val = 1.0 if i == n - 1 else 0.0
                res = np.where(end_knot_mask, end_val, res)

            return res


    @staticmethod
    def compute_collocation_matrix(t_arr, num_cps, degree, knot_vector):
        return np.array([
            BSpline.basis(t_arr, j, degree, knot_vector)
            for j in range(num_cps)
        ]).T


    @staticmethod
    def generate_uniform_open_knots(degree, num_cps):
        num_internal_knots = num_cps - degree - 1
        num_external_knots = degree + 1

        head_knots = np.repeat(0, num_external_knots - 1)
        tail_knots = np.repeat(1, num_external_knots - 1)
        internal_knots = np.linspace(0, 1, num_internal_knots + 2)

        return np.concatenate([head_knots, internal_knots, tail_knots])

