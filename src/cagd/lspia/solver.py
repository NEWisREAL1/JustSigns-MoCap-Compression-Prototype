import numpy as np

from src.cagd.lspia.initializer import LSPIAInitializer
from src.cagd.lspia.knot_topo import KnotTopology
from src.cagd.splines import BSpline


class LSPIASolver:
    def __init__(
        self, 
        degree=2,
        initial_num_cps=20,
        initializer : LSPIAInitializer = None,
        knot_topo : KnotTopology = None,
        tolerance=1e-3,
        max_iters=250, 
        convergence_iters=25,
        macro_iters=20,
        ):
        self.degree = degree
        self.initial_num_cps = initial_num_cps
        self.initializer = initializer
        self.knot_topo = knot_topo
        self.tolerance = tolerance
        self.convergence_iters = convergence_iters
        self.max_iters = max_iters
        self.macro_iters = macro_iters

        # caching
        self.collocation_mat = None
        self.weight_mat = None


    def fit(self, data_points, times):
        data_points = np.array(data_points)
        abscissas, knot_vector, control_points = self.initializer.init_params(data_points, self.degree, self.initial_num_cps)
        abscissas = np.array(times) / times[-1] # rewrote the absissas with times instead
        self._build_matrices(abscissas, knot_vector, control_points)

        best_err = np.inf
        iter_w_no_improv = 0

        for i in range(self.max_iters):
            # micro step
            control_points, error_vector = self._micro_step_update(data_points, control_points)
            point_err_norm = np.linalg.norm(error_vector, axis=1)
            max_point_err_norm = np.max(point_err_norm)

            # macro step
            if i != 0 & i % self.macro_iters == 0:
                knot_vector, control_points = self._macro_step_update(
                    data_points, abscissas, knot_vector, control_points, point_err_norm, self.degree
                    )

            # stopping conditions
            if max_point_err_norm <= self.tolerance:
                break

            if max_point_err_norm < best_err - 1e-8:
                best_err = max_point_err_norm
                iter_w_no_improv = 0
            else:
                iter_w_no_improv += 1
            
            if iter_w_no_improv >= self.convergence_iters:
                break

        return BSpline(self.degree, control_points, knot_vector), abscissas


    # ----- Optimizing Steps ----- #

    def _micro_step_update(self, data_points, control_points):
        approx_pts    = self.collocation_mat @ control_points
        error_vector  = data_points - approx_pts
        local_updates = self.weight_mat @ error_vector

        new_control_pts = control_points + 1.25 * local_updates
        return new_control_pts, error_vector

    
    def _macro_step_update(self, data_points, abscissas, knot_vector, control_points, point_err_norm, degree):
        knot_vector, control_points, insertion_occurred = self.knot_topo.optimize(
            data_points, abscissas, knot_vector, control_points, point_err_norm, degree
            )

        if insertion_occurred:
            self._build_matrices(abscissas, knot_vector, control_points)

        return knot_vector, control_points



    # ----- Utils ----- #

    def _build_matrices(self, abscissas, knot_vector, control_points):
        self.collocation_mat = np.array([
            BSpline.basis(abscissas, j, self.degree, knot_vector)
            for j in range(control_points.shape[0])
        ]).T

        norm_mat_diag = np.sum(self.collocation_mat, axis=0)
        division_mask = norm_mat_diag > 1e-10   # ensure no division by zero
        norm_mat_inv_diag = np.zeros_like(norm_mat_diag)
        norm_mat_inv_diag[division_mask] = 1.0 / norm_mat_diag[division_mask]
        
        norm_mat_inv = np.diag(norm_mat_inv_diag)
        self.weight_mat = norm_mat_inv @ self.collocation_mat.T