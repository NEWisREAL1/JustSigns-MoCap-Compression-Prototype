import numpy as np

from src.cagd.splines import BSpline


class LSPIASolver:
    def __init__(self, degree=2, initial_num_cps=20, centripetal_power=0.5):
        self.degree = degree
        self.initial_num_cps = initial_num_cps
        self.centripetal_power = centripetal_power


    def fit(self, data_pts, max_iters=1000):
        data_times  = self._init_data_times(data_pts)
        knot_vector = self._init_knot_vector(data_times)
        control_pts = self._init_control_pts(data_pts, data_times, knot_vector)

        collocation_mat, weight_mat = self._build_matrices(data_times, knot_vector, control_pts)

        for i in range(max_iters):
            approx_pts    = collocation_mat @ control_pts
            error_vector  = data_pts - approx_pts
            local_updates = weight_mat @ error_vector

            control_pts = control_pts + (1) * local_updates

        return BSpline(self.degree, control_pts, knot_vector), data_times


    # ----- Initialization ----- #

    def _init_data_times(self, data_pts):
        """Centripetal time parameterization"""
        diffs = np.diff(data_pts, axis=0)
        step_dists = np.linalg.norm(diffs, axis=1) ** self.centripetal_power
        cumu_dists = np.cumsum(step_dists, axis=0)
        if cumu_dists[-1] == 0:
            cumu_dists[-1] = 1
        times = cumu_dists / cumu_dists[-1]
        return np.concatenate([[0], times])


    def _init_knot_vector(self, data_times):
        """Time-averaging knot initialization"""
        knot_vector = self.generate_uniform_open_knots(self.initial_num_cps)
        num_internal = self.initial_num_cps - self.degree - 1
        d = data_times.shape[0] / float(self.initial_num_cps - self.degree)

        for j in range(num_internal):
            math_j = j + 1                      # internal knot idx
            knot_idx = math_j + self.degree     # global knot idx
            start_idx = int(math_j * d)         # the starting index in the data array
            knot_vector[knot_idx] = np.mean(data_times[start_idx : start_idx + self.degree])

        return knot_vector


    def _init_control_pts(self, data_pts, data_times, knot_vector):
        """Greville Abscissa sampling"""
        control_pts = np.empty(shape=(self.initial_num_cps, data_pts.shape[1]))
        
        for j in range(self.initial_num_cps):
            # Greville Abscissa (the peak of the basis function)
            t_peak = np.mean(knot_vector[j + 1 : j + 1 + self.degree])
            
            # find closest time parameter in data
            k = np.abs(data_times - t_peak).argmin()            
            control_pts[j] = data_pts[k]
        
        return control_pts


    # ----- Utils ----- #

    def generate_uniform_open_knots(self, num_cps):
        num_internal_knots = num_cps - self.degree - 1
        num_external_knots = self.degree + 1

        head_knots = np.repeat(0, num_external_knots - 1)
        tail_knots = np.repeat(1, num_external_knots - 1)
        internal_knots = np.linspace(0, 1, num_internal_knots + 2)

        return np.concatenate([head_knots, internal_knots, tail_knots])


    def _build_matrices(self, data_times, knot_vector, control_pts):
        collocation_mat = np.array([
            BSpline.basis(data_times, j, self.degree, knot_vector)
            for j in range(control_pts.shape[0])
        ]).T

        norm_mat_diag = np.sum(collocation_mat, axis=0)
        division_mask = norm_mat_diag > 1e-10   # ensure no division by zero
        norm_mat_inv_diag = np.zeros_like(norm_mat_diag)
        norm_mat_inv_diag[division_mask] = 1.0 / norm_mat_diag[division_mask]
        
        norm_mat_inv = np.diag(norm_mat_inv_diag)
        weight_mat = norm_mat_inv @ collocation_mat.T

        return collocation_mat, weight_mat 
