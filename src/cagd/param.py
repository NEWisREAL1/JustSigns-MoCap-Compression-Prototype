from abc import ABC, abstractmethod

import numpy as np


class IParameterizer(ABC):
    """
    Interface for time parameterizers
    """

    @abstractmethod
    def parameterize(self, data_pts, space_interface):
        pass


class UniformParameterizer(IParameterizer):
    """
    Encapsulator for time parameterization of data points
    using uniformly spaced parameters
    """

    def parameterize(self, data_pts, space_interface):
        data_pts = np.atleast_2d(data_pts)
        return np.linspace(0, 1, data_pts.shape[0])


class ChordLengthParameterizer(IParameterizer):
    """
    Encapsulator for time parameterization of data points 
    using chord length parameterization
    """

    def parameterize(self, data_pts, space_interface):
        data_pts = np.atleast_2d(data_pts)
        step_dists = space_interface.distance(data_pts[:-1], data_pts[1:])
        cumu_dists = np.cumsum(step_dists)
        normalized_step_dists = cumu_dists / cumu_dists[-1]
        return np.concatenate(([0], normalized_step_dists))


class CentripetalParameterizer(IParameterizer):
    """
    Encapsulator for time parameterization of data points 
    using centripetal parameterization
    """

    def __init__(self, power=0.5):
        self.power = power

    def parameterize(self, data_pts, space_interface):
        data_pts = np.atleast_2d(data_pts)
        step_dists = space_interface.distance(data_pts[:-1], data_pts[1:])
        cumu_dists = np.cumsum(step_dists)
        normalized_step_dists = cumu_dists / cumu_dists[-1]
        return np.concatenate(([0], normalized_step_dists)) ** self.power
