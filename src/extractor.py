import sys

import numpy as np


def extract_as_3d_tensor(clip_path, print_prefix=""):
    clip = np.load(clip_path, allow_pickle=True)

    actor = clip["actors"]
    joint_names = actor[0].item()["body"].keys()
    num_frames = len(clip["timestamps"])
    num_joints = len(joint_names)

    data = {}
    for name in joint_names:
        data[name] = np.empty(shape=(num_frames, 7))

    for frame in range(num_frames):
        body = actor[frame].item()["body"]

        for name in joint_names:
            pos = body[name]["position"]
            rot = body[name]["rotation"]
            data[name][frame] = [
                pos["x"],
                pos["y"],
                pos["z"],
                rot["x"],
                rot["y"],
                rot["z"],
                rot["w"],
            ]

        sys.stdout.write(
            f"\r{print_prefix} saving frame {frame + 1}/{num_frames} ({num_joints} joints)"
        )
        sys.stdout.flush()

    return data
