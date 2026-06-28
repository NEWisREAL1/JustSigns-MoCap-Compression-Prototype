from abc import ABC, abstractmethod

import numpy as np

from src.cagd.splines import BSpline


class LSPIAInitializer(ABC):

    @abstractmethod
    def init_params(self, data_points, degree, num_control_points):
        """Returns (abscissas, knot_vector, control_points)"""
        pass


class SimpleInitializer(LSPIAInitializer):

    def init_params(self, data_points, degree, num_control_points):
        abscissas = np.linspace(0, 1, data_points.shape[0])
        knot_vector = BSpline.generate_uniform_open_knots(degree, num_control_points)
        control_points = data_points[np.linspace(0, data_points.shape[0] - 1, num_control_points, dtype=int)]
        return abscissas, knot_vector, control_points  