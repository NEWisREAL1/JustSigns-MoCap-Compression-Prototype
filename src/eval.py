import os

import numpy as np
import pandas as pd

# ----- VALUES ERROR ----- #

def mean_per_joint_position_error(pos_arr1, pos_arr2):
    """
    Calculate the Mean Per-Joint Position Error (MPJPE) of the animations.

    inputs: pos_arr of shape (m, J, 3)
    """
    distances = np.linalg.norm(pos_arr1 - pos_arr2, axis=-1)
    mpjpe = np.mean(distances)
    return mpjpe

# ----- MEMORY USAGE ----- #

def memory_saving_matrix(paths, names, as_percents=True, clip_negs=False):
    n = len(paths)
    memory_used = [os.path.getsize(path) for path in paths]
    matrix = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            ratio = memory_used[i] / memory_used[j]
            saved = 1 - ratio
            matrix[i][j] = 100 * saved if as_percents else saved

    if clip_negs:
        matrix = np.clip(matrix, 0, None)

    return pd.DataFrame(data=matrix, index=names, columns=names)