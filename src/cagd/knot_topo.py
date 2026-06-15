from abc import ABC, abstractmethod

import numpy as np


class IKnotTopology(ABC):
    """
    Interface for topological managers of knot vector
    """

    @abstractmethod
    def evaluate_and_insert(self, knot_vector, control_pts, point_errors, data_time, space_interface):
        pass

    def generate_initial(self, data_time, degree, num_cps):
        knot_vector = self.generate_uniform_open_knots(degree, num_cps)
        num_internal = num_cps - degree - 1
        d = data_time.shape[0] / float(num_cps - degree)

        for j in range(num_internal):
            math_j = j + 1                      # internal knot idx
            knot_idx = math_j + degree          # global knot idx
            start_idx = int(math_j * d)         # the starting index in the data array
            knot_vector[knot_idx] = np.mean(data_time[start_idx : start_idx + degree])

        return knot_vector

    def generate_uniform_open_knots(self, degree, num_cps):
        num_internal_knots = num_cps - degree - 1
        num_external_knots = degree + 1

        head_knots = np.repeat(0, num_external_knots - 1)
        tail_knots = np.repeat(1, num_external_knots - 1)
        internal_knots = np.linspace(0, 1, num_internal_knots + 2)

        return np.concatenate([head_knots, internal_knots, tail_knots])


class StaticKnots(IKnotTopology):
    """
    Encapsulator for knot vector topological logics 
    with static-sized knot vector
    """

    def evaluate_and_insert(self, knot_vector, control_pts, point_errors, data_time, space_interface):
        return knot_vector, control_pts, False


class AdaptiveBoehmKnots(IKnotTopology):
    """
    Encapsulator for knot vector topological logics with knot vector 
    with Boehm knots & control points insertion strategy
    """

    def __init__(self, knot_insertion_tolerance=1e-3):
        self.knot_insertion_tolerance = knot_insertion_tolerance


    def evaluate_and_insert(self, knot_vector, control_pts, point_errors, data_time, space_interface):
        degree = knot_vector.shape[0] - control_pts.shape[0] - 1    # assumed from input shapes

        # group points/errs. into knot spans -> calculate max err. in each span
        knot_span_idx = np.searchsorted(knot_vector, data_time, side="right") - 1
        knot_span_idx = np.clip(knot_span_idx, degree, knot_vector.shape[0] - degree - 2)
        
        unique_spans, split_indices = np.unique(knot_span_idx, return_index=True)
        
        max_span_err = np.maximum.reduceat(point_errors, split_indices)

        spans_to_insert = list(zip(unique_spans, max_span_err))
        spans_to_insert.sort(key=lambda x: x[0], reverse=True) 

        insertion_occurred = False
        new_knot_vector = np.copy(knot_vector)
        new_control_pts = np.copy(control_pts)

        for span_idx, span_err in spans_to_insert:
            if span_err > self.knot_insertion_tolerance:
                insertion_occurred = True
                
                mid_span = (new_knot_vector[span_idx] + new_knot_vector[span_idx + 1]) / 2.0
                new_cps = self.boehm_cps_insertion(new_control_pts, new_knot_vector, mid_span, span_idx, degree, space_interface)

                new_knot_vector = np.insert(new_knot_vector, span_idx + 1, mid_span)
                new_control_pts = new_cps
                
        return new_knot_vector, new_control_pts, insertion_occurred

    def boehm_cps_insertion(self, control_pts, knot_vector, new_knot, span_idx, degree, space_interface):
        new_cps = np.zeros(shape=(control_pts.shape[0] + 1, control_pts.shape[1]))

        for i in range(control_pts.shape[0] + 1):
            if i <= span_idx - degree:
                new_cps[i] = control_pts[i]
            
            elif span_idx - degree + 1 <= i <= span_idx:
                insertion_ratio = (new_knot - knot_vector[i]) / (knot_vector[i + degree] - knot_vector[i])
                new_cps[i] = space_interface.interpolate(control_pts[i - 1], control_pts[i], insertion_ratio)
            
            elif i >= span_idx + 1:
                new_cps[i] = control_pts[i - 1]
            
        return new_cps

