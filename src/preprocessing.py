from copy import deepcopy

import numpy as np


class AntipodalPreprocessor:
    """Ensure there are no sudden sign flips in the sequence of quaternions"""

    def process(self, track):
        prep_track = deepcopy(track)

        if "quaternion" in track["type"]:
            aligned_values = np.copy(prep_track["values"]).reshape(-1, 4)
            for i in range(1, aligned_values.shape[0]):
                if np.dot(aligned_values[i], aligned_values[i - 1]) < 0:
                    aligned_values[i] = -aligned_values[i]
            prep_track["values"] = aligned_values

        return prep_track