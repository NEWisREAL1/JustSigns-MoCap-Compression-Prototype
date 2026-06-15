import numpy as np

from src.cagd.basis import bspline_basis
from src.cagd.cps_init import IInitializer
from src.cagd.knot_optim import IKnotOptimizer
from src.cagd.knot_topo import IKnotTopology
from src.cagd.manifold import IManifoldSpace
from src.cagd.param import IParameterizer
from src.cagd.splines import BSpline
from src.cagd.time_optim import ITimeOptimizer


class LSPIASolver:
    """
    Composited class for solving/optimizing LSPIA
    """

    def __init__(
        self,
        degree          : int,
        space           : IManifoldSpace,
        param           : IParameterizer,
        init            : IInitializer,
        knot_topo       : IKnotTopology,
        knot_optim      : IKnotOptimizer,
        time_optim      : ITimeOptimizer,
        ):
        self.degree = degree
        self.space = space
        self.param = param
        self.init = init
        self.knot_topo = knot_topo
        self.knot_optim = knot_optim
        self.time_optim = time_optim

        self.data_time = None
        self.knot_vector = None
        self.control_pts = None

        # caching
        self.collocation_mat = None
        self.weight_mat = None

        # logging
        self.history = dict(error=[], num_control_pts=[])

    def clear(self):
        self.data_time = None
        self.knot_vector = None
        self.control_pts = None
        self.collocation_mat = None
        self.weight_mat = None
        self.history = dict(error=[], num_control_pts=[])

    def fit(
        self, data_pts, 
        macro_step=50, max_iters=1000, converge_iter=25, 
        tolerance=1e-3, verbose=False, print_every=10,
        err_agg_func=np.max,
        ):
        # INITIALIZING
        self.data_time = self.param.parameterize(data_pts, self.space)
        self.knot_vector = self.knot_topo.generate_initial(self.data_time, self.degree, self.init.num_initial_control_pts)
        self.control_pts = self.init.initialize_control_pts(data_pts, self.data_time, self.knot_vector)

        self._build_matrices()
        approx_pts = self.space.evaluate_curve(self.control_pts, self.collocation_mat)
        global_err = self.space.calculate_error(approx_pts, data_pts)
        summarized_err = err_agg_func(np.linalg.norm(global_err, axis=1))

        delta_err = 0
        best_err = summarized_err
        iter_w_no_improvement = 0

        # OPTIMIZING LOOP
        for i in range(max_iters):
            # MICRO STEP
            global_err = self._micro_step(data_pts, global_err)

            # updating error
            new_summarized_err = err_agg_func(np.linalg.norm(global_err, axis=1))
            delta_err = new_summarized_err - summarized_err
            summarized_err = new_summarized_err 
            
            # MACRO STEP
            if delta_err <= 1e-4 and macro_step != 0 and i % macro_step == 0:
                self._macro_step(data_pts, global_err)

            # VERBOSITY
            if verbose and i % print_every == 0:
                print(f"iter {i:<5} err. = {summarized_err:.6f}")

            # LOGGING
            self.history["error"].append(summarized_err)
            self.history["num_control_pts"].append(self.control_pts.shape[0])
            
            # STOPPING CONDITIONS
            if summarized_err <= tolerance:
                if verbose:
                    print(f"Stopped due to tolerance, final err. = {summarized_err:.6f}")
                break

            if summarized_err < best_err:
                best_err = summarized_err
                iter_w_no_improvement = 0
            else:
                iter_w_no_improvement += 1

            if iter_w_no_improvement >= converge_iter:
                if verbose:
                    print(f"Stopped due to convergence, final err. = {summarized_err:.6f}")
                break


        return BSpline(self.degree, self.control_pts, self.knot_vector), self.data_time

    def _micro_step(self, data_pts, global_err):
        local_err  = self.weight_mat @ global_err
        self.control_pts = self.space.apply_update(self.control_pts, local_err)
        approx_pts = self.space.evaluate_curve(self.control_pts, self.collocation_mat)
        return self.space.calculate_error(approx_pts, data_pts)

    def _macro_step(self, data_pts, global_err):
        matrices_invalidated = False
        point_err = np.linalg.norm(global_err, axis=1)

        # 1st prioity: FREE-KNOT OPTIMIZATION
        new_knots = self.knot_optim.optimize_knots(
            self.knot_vector, 
            point_err, 
            self.data_time, 
            self.degree
            )

        shift_magnitude = np.max(np.abs(self.knot_vector - new_knots))

        if shift_magnitude >= 1e-4:
            self.knot_vector = new_knots
            matrices_invalidated = True

        # 2nd priority: ADAPTIVE KNOTS/CPS INSERTION (if no free-knot movement occur)
        else:
            new_knots, new_cps, inserted = self.knot_topo.evaluate_and_insert(
                self.knot_vector, 
                self.control_pts, 
                point_err,
                self.data_time,
                self.space,
                )

            if inserted:
                self.knot_vector = new_knots
                self.control_pts = new_cps
                matrices_invalidated = True

        # REVALIDATE MATRICES
        if matrices_invalidated:
            self._build_matrices()

    # ----- Utils ----- #

    def _build_matrices(self):
        self.collocation_mat = np.array([
            bspline_basis(self.data_time, j, self.degree, self.knot_vector)
            for j in range(self.control_pts.shape[0])
        ]).T

        norm_mat_diag = np.sum(self.collocation_mat, axis=0)
        division_mask = norm_mat_diag > 1e-10   # ensure no division by zero
        norm_mat_inv_diag = np.zeros_like(norm_mat_diag)
        norm_mat_inv_diag[division_mask] = 1.0 / norm_mat_diag[division_mask]
        
        norm_mat_inv = np.diag(norm_mat_inv_diag)
        self.weight_mat = norm_mat_inv @ self.collocation_mat.T