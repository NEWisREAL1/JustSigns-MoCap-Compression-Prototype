import numpy as np


def mean_per_joint_position_error(pos_arr1, pos_arr2):
    """
    Calculate the Mean Per-Joint Position Error (MPJPE) of the animations.

    inputs: pos_arr of shape (m, J, 3)
    """
    distances = np.linalg.norm(pos_arr1 - pos_arr2, axis=-1)
    mpjpe = np.mean(distances)
    return mpjpe
