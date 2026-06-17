import json

import numpy as np


def load_clip(path):
    """Load a full json clip """
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_clip_npz(path):
    """Load a full npz clip as json/dict"""
    data = np.load(path, allow_pickle=True)
    clip = data["clip"].item()
    return clip


def print_json_structure(json_dict, level=0, indent=4):
    for key, val in json_dict.items():
        val_type = type(val)
        
        if val_type is dict:
            print(f"{" " * indent * level}{key}:")
            print_json_structure(val, level=level + 1, indent=indent)

        elif val_type is list:
            if type(val[-1]) is dict:
                print(f"{" " * indent * level}{key}: [ {len(val)} entries of")
                print_json_structure(val[-1], level=level + 1, indent=indent)
                print(f"{" " * indent * level}]")
            
            elif type(val[-1]) is list:
                pass # nested array not supported ^_^

            else:
                print(f"{" " * indent * level}{key}: [ {len(val)} entries of <{type(val[-1]).__name__}> ]")

        else:
            print(f"{" " * indent * level}{key}: <{val_type.__name__}>")


def pack_anim_to_dict(clip, includes_blendshapes=False, clean_names=True):
    """
    Extract joint tracks and parse into the dict of format {joint_name: value}.\n
    Also reshape the value according to track type, e.g., (F, 4) for quaternion, (F,) for number.
    """
    tracks = clip["animationClip"]["tracks"]
    res = {}

    for track in tracks:
        values = np.asarray(track["values"], dtype=np.float64)
        type = track["type"]
        name = track["name"]
        if clean_names:
            name = name.replace(".quaternion", "")

        if type == "quaternion":
            res[name] = np.reshape(values, shape=(-1, 4))
        elif type == "number" and includes_blendshapes:
            res[name] = values

    return res


def pack_anim_to_array(clip, kinematics_skeleton, num_frames=None, includes_blendshapes=False, clean_names=True):
    """
    Packs a dictionary of named joint animations into a 3D NumPy array.
    If a joint is missing, it seamlessly falls back to the static bind rotation.
    """
    anim_dict = pack_anim_to_dict(clip, includes_blendshapes=includes_blendshapes, clean_names=clean_names)

    if num_frames is None:
        if not anim_dict:
            raise ValueError("anim_dict is empty and num_frames not provided.")
        num_frames = next(iter(anim_dict.values())).shape[0]
        
    J = len(kinematics_skeleton.joint_names)
    anim_rotations = np.zeros((num_frames, J, 4), dtype=np.float64)
    
    for j, name in enumerate(kinematics_skeleton.joint_names):
        if name in anim_dict:
            # squeeze and shape safely handle both single-frame and multi-frame
            local_q = np.asarray(anim_dict[name])
            if local_q.ndim == 1:
                local_q = local_q.reshape(1, 4)
            anim_rotations[:, j, :] = local_q
        else:
            # FALLBACK: use the bind rotation
            anim_rotations[:, j, :] = kinematics_skeleton.bind_rotations[j]
            
    return anim_rotations