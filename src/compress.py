import numpy as np

from src.cagd.bspline import BSpline, StationarySpline
from src.quaternion import (
    antipodal_alignment,
    check_antipodal,
    geodesic_distances_rad,
    quaternions_mean,
)


def compress_clip(clip_quats, fitter, station_thres_deg=2.8, **fit_params):
    compressed_clip = {}
    info = {}

    info["error_history"] = {}
    info["num_moving_joints"] = 0
    info["num_nonmoving_joints"] = 0

    for joint_name, quats in clip_quats.items():
        # antipodal alignment
        if check_antipodal(quats):
            quats = antipodal_alignment(quats)

        # stationary joint detection
        q_mean = quaternions_mean(quats)
        deviations_rad = geodesic_distances_rad(quats, q_mean)
        is_stationary = np.max(deviations_rad) < np.deg2rad(station_thres_deg)

        if is_stationary:
            # collapse to singular quat
            compressed_clip[joint_name] = {}
            compressed_clip[joint_name]["compression_type"] = "singular"
            compressed_clip[joint_name]["value"] = q_mean
            compressed_clip[joint_name]["num_frame"] = quats.shape[0]
            
            info["error_history"][joint_name] = []
            info["num_nonmoving_joints"] += 1

        else:
            # curve/spline fitting
            fitter.clear()
            fitted_spl, data_time, err_hist = fitter.fit(quats, **fit_params)
            
            # data gathering
            compressed_clip[joint_name] = {}
            compressed_clip[joint_name]["compression_type"] = "bspline"
            compressed_clip[joint_name]["degree"] = fitted_spl.degree
            compressed_clip[joint_name]["control_pts"] = fitted_spl.control_pts
            compressed_clip[joint_name]["knot_vector"] = fitted_spl.knot_vector
            compressed_clip[joint_name]["data_time"] = data_time

            # save info
            info["error_history"][joint_name] = err_hist
            info["num_moving_joints"] += 1

    return compressed_clip, info


def decompress_clip(compressed_quats):
    decom_quats = {}

    for joint_name, compressed_joint in compressed_quats.items():
        compression_type = compressed_joint["compression_type"]

        if compression_type == "bspline":
            degree = compressed_joint["degree"]
            control_pts = compressed_joint["control_pts"]
            knot_vector = compressed_joint["knot_vector"]
            data_time = compressed_joint["data_time"]

            spl = BSpline(degree, control_pts, knot_vector)
            decom_quats[joint_name] = spl(data_time)

        elif compression_type == "singular":
            value = compressed_joint["value"]
            num_frame = compressed_joint["num_frame"]

            spl = StationarySpline(singularity=value)
            decom_quats[joint_name] = spl(np.arange(num_frame))

        else:
            raise ValueError(f"Unknow compression_type \"{compression_type}\"")

    return decom_quats
