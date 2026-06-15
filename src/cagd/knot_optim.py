from abc import ABC, abstractmethod

import numpy as np


class IKnotOptimizer(ABC):
    """
    Interface for knot optimizers (static knot length optimizers)
    """

    @abstractmethod
    def optimize_knots(self, knot_vector, point_errors, data_time, degree):
        pass


class NoOptimization(IKnotOptimizer):
    """
    Encapsulator for knot optimization... but this one represent no optimization.
    """

    def optimize_knots(self, knot_vector, point_errors, data_time, degree):
        return knot_vector


class SpringMassKnotShifter(IKnotOptimizer):
    """
    Encapsulator for knot optimization using the manifold-safe Spring-Mass heuristic.
    """
    def __init__(self, learning_rate=0.05, min_span_ratio=0.01):
        self.lr = learning_rate
        self.min_span = min_span_ratio 

    def optimize_knots(self, knot_vector, point_errors, data_time, degree):
        knot_vector = np.copy(knot_vector)
        num_knots = len(knot_vector)
        
        internal_start = degree + 1
        internal_end = num_knots - degree - 1
        
        if internal_start >= internal_end:
            return knot_vector

        # group data points into knot spans
        span_indices = np.searchsorted(knot_vector, data_time, side="right") - 1
        span_indices = np.clip(span_indices, degree, num_knots - degree - 2)

        # calculate the maximum error in each span
        unique_spans, split_indices = np.unique(span_indices, return_index=True)
        max_span_err = np.maximum.reduceat(point_errors, split_indices)

        span_errors = np.zeros(num_knots - 1)
        span_errors[unique_spans] = max_span_err

        right_span_errors = span_errors[internal_start : internal_end]
        left_span_errors = span_errors[internal_start - 1 : internal_end - 1]
        
        forces = right_span_errors - left_span_errors

        shifts = forces * self.lr
        current_internal_knots = knot_vector[internal_start : internal_end]
        proposed_knots = current_internal_knots + shifts

        left_bounds = knot_vector[internal_start - 1 : internal_end - 1] + self.min_span
        right_bounds = knot_vector[internal_start + 1 : internal_end + 1] - self.min_span

        clamped_knots = np.maximum(left_bounds, np.minimum(right_bounds, proposed_knots))
        knot_vector[internal_start : internal_end] = clamped_knots
        
        return knot_vector