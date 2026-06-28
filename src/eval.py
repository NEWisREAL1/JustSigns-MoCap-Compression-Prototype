import os

import matplotlib.pyplot as plt
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


def plot_memory_usage_matrix(mem_mat: pd.DataFrame, color_thres=10):
    matrix = mem_mat.values
    names = mem_mat.index
    n = matrix.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    # eeplace 0 with np.nan so they don't skew the color scale
    matrix_for_plot = np.where(matrix == 0, np.nan, matrix)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="black") 

    im = ax.imshow(matrix_for_plot, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(names)

    lo = np.mean(matrix[matrix != 0]) - np.std(matrix[matrix != 0])
    hi = np.mean(matrix[matrix != 0]) + np.std(matrix[matrix != 0])

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val == 0:
                ax.text(j, i, "0.00%", ha="center", va="center", color="white")
            else:
                ax.text(
                    j, i, f"{val:.2f}%", ha="center", va="center", 
                    color="black" if lo < val < hi else "white"
                    )

    # Formatting grid
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("Memory Saving Matrix\n(Saving when Using \"Row\" Over \"Column\")", fontsize=16)
    plt.tight_layout()
    plt.show()