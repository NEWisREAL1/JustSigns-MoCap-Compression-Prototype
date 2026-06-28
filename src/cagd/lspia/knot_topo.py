from abc import ABC, abstractmethod

import numpy as np


class KnotTopology(ABC):

    @abstractmethod
    def optimize(self):
        pass


class BoehmKnotOptimizer(KnotTopology):

    def __init__(self, tolerance=1e-3, max_insertion_per_step=3):
        self.tolerance = tolerance
        self.max_insertion_per_step = max_insertion_per_step


    def optimize(self, data_points, abscissas, knot_vector, control_points, point_err_norm, degree):
        # group into spans
        knot_span_idx = np.searchsorted(knot_vector, abscissas, side="right") - 1
        knot_span_idx = np.clip(knot_span_idx, degree, len(knot_vector) - degree - 2)
        unique_spans, split_indices = np.unique(knot_span_idx, return_index=True)
        max_span_err = np.maximum.reduceat(point_err_norm, split_indices)

        spans_to_insert = list(zip(unique_spans, max_span_err))
        spans_to_insert.sort(key=lambda x: x[0], reverse=True) 
        insertion_occurred = False

        for span_idx, span_err in spans_to_insert[:self.max_insertion_per_step]:
            if span_err > self.tolerance:
                insertion_occurred = True
                mid_span = (knot_vector[span_idx] + knot_vector[span_idx + 1]) / 2.0
                new_control_pts = np.zeros(shape=(control_points.shape[0] + 1, control_points.shape[1]))

                for i in range(control_points.shape[0] + 1):
                    # Zone 1: Prefix
                    if i <= span_idx - degree:
                        new_control_pts[i] = control_points[i]
                    
                    # Zone 2: Insertion (Corner Cutting)
                    elif span_idx - degree + 1 <= i <= span_idx:
                        insertion_ratio = (mid_span - knot_vector[i]) / (knot_vector[i + degree] - knot_vector[i])
                        new_control_pts[i] = (1 - insertion_ratio) * control_points[i - 1] + insertion_ratio * control_points[i]
                    
                    # Zone 3: Suffix
                    elif i >= span_idx + 1:
                        new_control_pts[i] = control_points[i - 1]

                # Update the class state
                knot_vector = np.insert(knot_vector, span_idx + 1, mid_span)
                control_points = new_control_pts
        
        return knot_vector, control_points, insertion_occurred
