"""Colloection of Preprocessors and Postprocessors"""

import numpy as np


class AntipodalAlignmentProcessor:
    """
    Preprocessor for quaternions data.
    Make sure there is no sudden sign flips in the data.
    """ 

    def __call__(self, data):
        aligned_data = np.copy(data)
        for i in range(1, data.shape[0]):
            if np.dot(aligned_data[i], aligned_data[i - 1]) < 0:
                aligned_data[i] = -aligned_data[i]
        return aligned_data


class NomalizeProcessor:
    """
    General pre/postprocessor that normalize the data
    """

    def __call__(self, data):
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        return data / norms
