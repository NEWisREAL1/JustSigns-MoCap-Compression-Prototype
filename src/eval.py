import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cagd.manifold import EuclideanSpace, QuaternionSpace
from src.kinematics import forward_kinematics
from src.model import (
    CORE_BODY_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_HAND_JOINTS,
    LOWER_BODY_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
)

# ----- Summary info extraction ----- #

def summarize_compression_info(raw_quats, compressed_quats, info):
    summary = {}
    joints = info["error"].keys()

    for joint in joints:
        summary[joint] = {}

        type = compressed_quats[joint]["compression_type"]
        summary[joint]["compression_type"] = type
        summary[joint]["final_error"]      = info["error"][joint][-1]
        summary[joint]["num_iterations"]   = len(info["error"][joint])
        summary[joint]["num_control_pts"]  = compressed_quats[joint]["fitted_spl"].control_pts.shape[0]
        summary[joint]["num_knots"]        = compressed_quats[joint]["fitted_spl"].knot_vector.shape[0]

        if type == "singular":
            summary[joint]["num_data_time"] = 1 # one integer, but assume float for simplicity
        elif type == "bspline":
            summary[joint]["num_data_time"] = compressed_quats[joint]["data_time"].shape[0]

        summary[joint]["total_used_floats"] = (
              4 * summary[joint]["num_control_pts"]
            + summary[joint]["num_knots"]
            + summary[joint]["num_data_time"]
        )

        summary[joint]["total_used_floats_raw"] = 4 * raw_quats[joint].shape[0] 
        summary[joint]["compression_ratio"] = summary[joint]["total_used_floats_raw"] / summary[joint]["total_used_floats"] 

    return pd.DataFrame(summary).T


def per_frame_errors(ground_truth_quats, approximate_quats):
    joints = ground_truth_quats.keys()
    num_frames = ground_truth_quats[list(joints)[0]].shape[0]
    errors = dict(
        positional=np.zeros(shape=(num_frames), dtype=np.float64),
        angular=np.zeros(shape=(num_frames), dtype=np.float64),
        )

    ground_truth_pos = forward_kinematics(ground_truth_quats)
    approximate_pos = forward_kinematics(approximate_quats)

    for joint in joints:
        gt_quats = ground_truth_quats[joint]
        ap_quats = approximate_quats[joint]
        gt_pos = ground_truth_pos[joint]
        ap_pos = approximate_pos[joint]

        positional_err = EuclideanSpace().distance(gt_pos, ap_pos)
        angular_err = QuaternionSpace().distance(gt_quats, ap_quats)

        errors["positional"] += positional_err / len(joints)
        errors["angular"] += angular_err / len(joints)
        
    return errors


# ----- Separate body parts plot ----- #

GRID_JOINTS = [
    [LEFT_HAND_JOINTS + LEFT_ARM_JOINTS  , CORE_BODY_JOINTS],
    [RIGHT_HAND_JOINTS + RIGHT_ARM_JOINTS, LOWER_BODY_JOINTS],
]

GRID_LABELS = [
    ["Left Hand & Arm", "Core Body"],
    ["Right Hand & Arm", "Lower Body"],
]

def body_parts_plot(series, title="PLot Title", type="bar", hline=None, hline_label=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True, width_ratios=(3.6, 1))

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):

            valid_keys = [key for key in GRID_JOINTS[i][j] if key in series.keys()]
            subset = series.loc[valid_keys]

            if type == "bar":
                ax.bar(np.arange(len(valid_keys)), subset)
            elif type == "scatter":
                ax.scatter(np.arange(len(valid_keys)), subset)

            ax.set_xticks(np.arange(len(valid_keys)), labels=valid_keys, rotation=-90)
            ax.set_title(GRID_LABELS[i][j])
            ax.grid(axis="y")

            if hline is not None:
                ax.axhline(y=hline, linestyle="--", color="k", label=hline_label)
                if hline_label is not None:
                    ax.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()