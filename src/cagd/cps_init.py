from abc import ABC, abstractmethod

import numpy as np


class IInitializer(ABC):
    """
    Interface for control points initializers
    """

    @abstractmethod
    def initialize_control_pts(self, data_pts, data_time, knot_vector):
        pass


class UniformSampler(ABC):
    """
    Encapsulator for control points initialization 
    using uniform data sampling
    """
    
    def __init__(self, num_initial_control_pts):
        self.num_initial_control_pts = num_initial_control_pts

    def initialize_control_pts(self, data_pts, data_time, knot_vector):
        data_pts = np.atleast_2d(data_pts)
        uniform_idx = np.round(np.linspace(0, data_pts.shape[0] - 1, self.num_initial_control_pts)).astype(int)
        return data_pts[uniform_idx]


class GrevilleAbscissaSampler(ABC):
    """
    Encapsulator for control points initialization 
    using Greville Abscissa sampling (basismax sampling)
    """
    def __init__(self, num_initial_control_pts):
        self.num_initial_control_pts = num_initial_control_pts

    def initialize_control_pts(self, data_pts, data_time, knot_vector):
        data_pts = np.atleast_2d(data_pts)
        control_pts = np.empty(shape=(self.num_initial_control_pts, data_pts.shape[1]))
        degree = knot_vector.shape[0] - self.num_initial_control_pts - 1    # assumed from input shapes
        
        for j in range(self.num_initial_control_pts):
            # Greville Abscissa (the peak of the basis function)
            t_peak = np.mean(knot_vector[j + 1 : j + 1 + degree])
            
            # find closest time parameter in data
            k = np.abs(data_time - t_peak).argmin()            
            control_pts[j] = data_pts[k]
        
        return control_pts

