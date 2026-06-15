import numpy as np


def bspline_basis(t, i, p, knot_vector, n=None):
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

            b1 = bspline_basis(t_arr, i, p - 1, knot_vector, n=n)
            b2 = bspline_basis(t_arr, i + 1, p - 1, knot_vector, n=n)

            res = w1 * b1 + w2 * b2

        # Vectorized check for the end-knot boundary condition
        end_knot_mask = np.isclose(t_arr, knot_vector[-1])
        if np.any(end_knot_mask):
            end_val = 1.0 if i == n - 1 else 0.0
            res = np.where(end_knot_mask, end_val, res)

        return res


def compute_collocation_matrix(t_arr, num_cps, degree, knot_vector):
    return np.array([
        bspline_basis(t_arr, j, degree, knot_vector)
        for j in range(num_cps)
    ]).T