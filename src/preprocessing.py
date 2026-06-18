import numpy as np


def antipodal_alignment(quats):
    """Ensure there are no sudden sign flips in the sequence of quaternions"""
    aligned_data = np.copy(quats)
    for i in range(1, quats.shape[0]):
        if np.dot(aligned_data[i], aligned_data[i - 1]) < 0:
            aligned_data[i] = -aligned_data[i]
    return aligned_data