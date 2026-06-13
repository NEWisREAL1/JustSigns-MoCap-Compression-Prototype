import numpy as np
from scipy.linalg import eigh


def check_antipodal(quats):
    """
    True = contains sudden sign flip
    """
    dots = np.einsum('ij,ij->i', quats[:-1], quats[1:])
    return np.any(dots < 0)


def antipodal_alignment(quats):
    """
    Remove sudden sign flips
    """
    n = quats.shape[0]
    aligned_quats = np.copy(quats)

    for i in range(1, n):
        if np.dot(aligned_quats[i], aligned_quats[i - 1]) < 0:
            aligned_quats[i] = -aligned_quats[i]

    return aligned_quats


def quaternions_mean(quats):
    accum_mat = (quats.T @ quats) / quats.shape[0]
    eigenvals, eigenvecs = eigh(accum_mat)
    return eigenvecs[:, np.argmax(eigenvals)]


def geodesic_distances_rad(quats1, quats2):
    quats1 = np.asarray(quats1)
    quats2 = np.asarray(quats2)

    quats1_was_vector = quats1.ndim == 1
    quats2_was_vector = quats2.ndim == 1

    quats1 = np.atleast_2d(quats1)
    quats2 = np.atleast_2d(quats2)

    distances_rad = 2 * np.arccos(np.clip(np.abs(quats1 @ quats2.T), -1, 1))

    if quats1_was_vector and quats2_was_vector:
        return distances_rad.item()
    if quats1_was_vector:
        return distances_rad[0]
    if quats2_was_vector:
        return distances_rad[:, 0]
    return distances_rad
