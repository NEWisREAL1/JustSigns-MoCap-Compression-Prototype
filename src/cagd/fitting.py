import numpy as np
from src.cagd.bspline import BSpline


class BSplineLSPIAFitter:
    """
    Optimizer for B-Spline LSPIA
    """

    ##### ----- Initialization & Validation ----- ##### 
    
    def __init__(
        self, 
        degree=2,
        initial_num_cps=10,
        time_init_method="centripetal", 
        knot_init_method="time-average", 
        cps_init_method="sampling-uniform", 
        relaxation_schedule="dynamic",
        centripetal_power=0.5,
        relaxation_factor=1.2,
        ):
        """
        Parameters
        ---
        time_init_method
            "uniform", "arc-length", or "centripetal"
        knot_init_method
            "uniform" or "time-average"
        cps_init_method
            "sampling-uniform", "sampling-basismax", or "origin"
        relaxation_schedule
            "static", "dynamic", or "origin"
        centripetal_power
            float in (0, 1]
        relaxation_factor
            float in (0, 2], exceeding 2 will cause divergance
        """
        self.data = None
        self.data_dim = None
        self.data_count = None
        self.data_time = None
        
        self.degree = degree
        self.num_cps = initial_num_cps
        self.control_pts = None
        self.knot_vector = None

        self.time_init_method = time_init_method
        self.knot_init_method = knot_init_method
        self.cps_init_method = cps_init_method
        self.relaxation_schedule = relaxation_schedule
        
        self.centripetal_power = centripetal_power
        self.relaxation_factor = relaxation_factor
        
        self.time_init_method_map = {
            "uniform"     : self._uniform_time_init,
            "arc-length"  : self._arclength_time_init,
            "centripetal" : self._centripetal_time_init,
        }

        self.knot_init_method_map = {
            "uniform"      : self._uniform_knot_init,
            "time-average" : self._timeaverage_knot_init,
        }

        self.cps_init_method_map = {
            "sampling-uniform"  : self._sampling_uniform_cps_init,
            "sampling-basismax" : self._sampling_basismax_cps_init,
            "origin"            : self._origin_cps_init,
        }

        self._validate()


    def _validate(self):
        if self.time_init_method not in self.time_init_method_map.keys():
            raise ValueError(f"Unknow time_init_method \"{self.time_init_method}\"")


    ##### ----- Time Parameterization ----- ##### 

    def _uniform_time_init(self):
        self.data_time = np.linspace(0, 1, len(self.data))


    def _arclength_time_init(self):
        diffs = np.diff(self.data, axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        cumu_dists = np.cumsum(step_dists)
        self.data_time = np.concatenate([[0], cumu_dists]) / cumu_dists[-1]


    def _centripetal_time_init(self):
        diffs = np.diff(self.data, axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        cumu_dists = np.cumsum(step_dists)
        lengths = np.concatenate([[0], cumu_dists]) / cumu_dists[-1]
        self.data_time = lengths ** self.centripetal_power


    def init_time_param(self):
        self.time_init_method_map[self.time_init_method]()


    ##### ----- Knot Vector Initialization ----- ##### 

    def _uniform_knot_init(self):
        self.knot_vector = BSpline.generate_uniform_open_knots(self.degree, self.num_cps)


    def _timeaverage_knot_init(self):
        self.knot_vector = BSpline.generate_uniform_open_knots(self.degree, self.num_cps)
        num_internal = self.num_cps - self.degree - 1
        d = self.data_count / float(self.num_cps - self.degree)

        for j in range(num_internal):
            math_j = j + 1                      # internal knot idx
            knot_idx = math_j + self.degree     # global knot idx
            start_idx = int(math_j * d)         # the starting index in the data array
            
            self.knot_vector[knot_idx] = np.mean(
                self.data_time[start_idx : start_idx + self.degree]
            )


    def init_knot_vector(self):
        self.knot_init_method_map[self.knot_init_method]()


    ##### ----- Control Points Initialization ----- ##### 

    def _sampling_uniform_cps_init(self):
        pass


    def _sampling_basismax_cps_init(self):
        pass


    def _origin_cps_init(self):
        self.control_pts = np.zeros(shape=(self.num_cps, self.data_dim))


    def init_control_points(self):
        self.cps_init_method_map[self.cps_init_method]()


    ##### ----- Relaxation Factor Scheduling (Optimal/Steepest Projection) ----- #####

    def optimal_relaxation_projection(self, global_err, local_err, colloc_mat):
        velocity = colloc_mat @ local_err
        optimal_relaxation = np.sum(global_err * velocity) / np.sum(velocity * velocity)
        return optimal_relaxation

     
    ##### ----- Dynamic Knots/Control Points Insertion (Boehm's Algorithm) ----- ##### 
    
    def boehm_cps_insertion(self, new_knot, span_idx):
        new_cps = np.zeros(shape=(self.num_cps + 1, self.data_dim))

        for i in range(self.num_cps + 1):
            if i <= span_idx - self.degree:
                new_cps[i] = self.control_pts[i]
            
            elif span_idx - self.degree + 1 <= i <= span_idx:
                # Note: When doing quaternions, replace this line with SLERP
                insertion_ratio = (new_knot - self.knot_vector[i]) / (self.knot_vector[i + self.degree] - self.knot_vector[i])
                new_cps[i] = (1 - insertion_ratio) * self.control_pts[i - 1] + insertion_ratio * self.control_pts[i]
            
            elif i >= span_idx + 1:
                new_cps[i] = self.control_pts[i - 1]
            
        return new_cps


    def knot_insertion(self, point_err, knot_insertion_tolerance):
        # group points/errs. into knot spans -> calculate max err. in each span
        knot_span_idx = np.searchsorted(self.knot_vector, self.data_time, side="right") - 1
        knot_span_idx = np.clip(knot_span_idx, self.degree, len(self.knot_vector) - self.degree - 2)
        
        unique_spans, split_indices = np.unique(knot_span_idx, return_index=True)
        
        max_span_err = np.maximum.reduceat(point_err, split_indices)

        spans_to_insert = list(zip(unique_spans, max_span_err))
        spans_to_insert.sort(key=lambda x: x[0], reverse=True) 

        insertion_occurred = False

        for span_idx, span_err in spans_to_insert:
            if span_err > knot_insertion_tolerance:
                insertion_occurred = True
                
                mid_span = (self.knot_vector[span_idx] + self.knot_vector[span_idx + 1]) / 2.0
                new_cps = self.boehm_cps_insertion(mid_span, span_idx)

                # update class state
                self.knot_vector = np.insert(self.knot_vector, span_idx + 1, mid_span)
                self.control_pts = new_cps
                self.num_cps += 1
                
        return insertion_occurred
    
    
    ##### ----- Dynamic Time Parameter Re-Estimation ----- #####

    def time_parameter_reestimation(self, global_err):
        div_colloc_mat = np.array([
            BSpline.basis_derivative(self.data_time, j, self.degree, self.knot_vector)
            for j in range(self.num_cps)
        ]).T

        vel_mat = div_colloc_mat @ self.control_pts

        dot_EV = np.sum(global_err * vel_mat, axis=1)
        dot_VV = np.sum(vel_mat * vel_mat, axis=1)
        division_mask = dot_VV > 1e-8   # avoid divide by zero

        # first order approximation
        delta_t = np.zeros_like(self.data_time)
        delta_t[division_mask] = dot_EV[division_mask] / dot_VV[division_mask]
        t_new = self.data_time + delta_t

        # maintain monotonicity
        t_lower = np.empty_like(self.data_time)
        t_upper = np.empty_like(self.data_time)
        t_lower[0] = 0.0
        t_lower[1:] = self.data_time[:-1] + 1e-6  # strictly greater than the previous t
        t_upper[-1] = 1.0
        t_upper[:-1] = self.data_time[1:] - 1e-6  # strictly less than the next t

        self.data_time = np.clip(t_new, t_lower, t_upper)
    

    ##### ----- Main LSPIA Algorithm ----- #####

    def lspia_step(self, colloc_mat, weight_matrix, global_err):
        local_err = weight_matrix @ global_err

        relax = self.relaxation_factor
        if self.relaxation_schedule == "dynamic":
            relax = self.optimal_relaxation_projection(global_err, local_err, colloc_mat)

        self.control_pts = self.control_pts + relax * local_err 
        new_global_err = self.data - colloc_mat @ self.control_pts
        return new_global_err


    def compute_matrices(self):
        colloc_mat = np.array([
            BSpline.basis(self.data_time, j, self.degree, self.knot_vector)
            for j in range(self.num_cps)
        ]).T

        norm_mat_inv_diag = 1 / np.sum(colloc_mat, axis=0)
        # avoid empty support, but can produce instability
        # norm_mat_inv_diag[np.isnan(norm_mat_inv_diag) | np.isinf(norm_mat_inv_diag)] = 0  
        norm_mat_inv = np.diag(norm_mat_inv_diag)
        weight_mat = norm_mat_inv @ colloc_mat.T
        return colloc_mat, weight_mat


    def fit(
        self, 
        data, 
        max_iter=1000, 
        tolerance=1e-5, 
        converge_iter=50,

        dynamic_knot_insertion=True,
        insert_knot_every=50, 
        knot_insertion_tolerance=None,
        
        # note: reparams tends to cause instability
        dynamic_reparams=True,
        reparams_every=10,

        err_agg_func=np.max, 
        print_every=25,
        ):
        self.data = np.asarray(data)
        self.data_count = self.data.shape[0]
        self.data_dim = self.data.shape[1]

        if knot_insertion_tolerance is None:
            knot_insertion_tolerance = tolerance
        
        self.init_time_param()
        self.init_knot_vector()
        self.init_control_points()

        colloc_mat, weight_mat = self.compute_matrices()

        global_err = self.data - colloc_mat @ self.control_pts
        err_history = [err_agg_func(np.linalg.norm(global_err, axis=1))]
        iter_w_no_improve = 0

        for iter in range(max_iter):
            
            ### optimizing step ###

            global_err = self.lspia_step(colloc_mat, weight_mat, global_err)
            point_err = np.linalg.norm(global_err, axis=1)
            
            ### dynamic parameters adjustment ###
            
            if dynamic_knot_insertion and iter % insert_knot_every == 0 and iter != 0:
                insertion_occurred = self.knot_insertion(point_err, knot_insertion_tolerance)

                # recompute matrices if an insertion happened
                if insertion_occurred:
                    colloc_mat, weight_mat = self.compute_matrices()
                    global_err = self.data - colloc_mat @ self.control_pts
                    point_err = np.linalg.norm(global_err, axis=1)
                            
            ### dynamic time parameter re-estimation ###\

            if dynamic_reparams and iter % reparams_every == 0 and iter != 0:
                self.time_parameter_reestimation(global_err)

                # recompute matrices if an insertion happened
                colloc_mat, weight_mat = self.compute_matrices()
                global_err = self.data - colloc_mat @ self.control_pts
                point_err = np.linalg.norm(global_err, axis=1)

            ### error evaluation & verbose ###

            err = err_agg_func(point_err)
            err_history.append(err)

            if (iter + 1) % print_every == 0 or iter == 0:
                print(f"iter {iter + 1:<6} | err. = {err:.10f} ({self.num_cps} cps)")

            # tolerance check
            if err <= tolerance:
                print(f"early stopping at iteration {iter + 1} due to tolerance, err. = {err} ({self.num_cps} cps)")
                break

            # convergance check
            if np.abs(err_history[-1] - err_history[-2]) < 1e-8:
                iter_w_no_improve += 1
            else:
                iter_w_no_improve = 0
            
            if iter_w_no_improve >= converge_iter:
                print(f"early stopping at iteration {iter + 1} due to convergence, err. = {err} ({self.num_cps} cps)")
                break

        return BSpline(self.degree, self.control_pts, self.knot_vector), err_history
