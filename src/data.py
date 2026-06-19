import json
from copy import deepcopy

import numpy as np


def load_clip(path):
    """Load a full json clip."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def load_clip_npz(path):
    """Load a full npz clip as json/dict."""
    data = np.load(path, allow_pickle=True)
    clip = data["clip"].item()
    return clip


def save_clip(data, path):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, cls=NumpyEncoder, indent=4, ensure_ascii=False)


def print_json_structure(json_dict, level=0, indent=4):
    for key, val in json_dict.items():
        val_type = type(val)
        
        if val_type is dict:
            print(f"{" " * indent * level}{key}:")
            print_json_structure(val, level=level + 1, indent=indent)

        elif val_type is list or val_type is np.ndarray:
            if type(val[-1]) is dict:
                print(f"{" " * indent * level}{key}: [ {len(val)} entries of")
                print_json_structure(val[-1], level=level + 1, indent=indent)
                print(f"{" " * indent * level}]")
            
            elif type(val[-1]) is list or type(val[-1]) is np.ndarray:
                pass # nested array not supported ^_^

            else:
                # only show len of the first entry
                print(f"{" " * indent * level}{key}: [ {len(val)} entries of <{type(val[-1]).__name__}> ]")

        else:
            print(f"{" " * indent * level}{key}: <{val_type.__name__}>")


def pack_anim_to_dict(clip, target="values", includes_blendshapes=False, clean_names=True):
    """
    Extract joint tracks and parse into the dict of format {joint_name: value}.\n
    Also reshape the value according to track type, e.g., (F, 4) for quaternion, (F,) for number.
    """
    if target not in ["values", "times", "type", "name"]:
        raise ValueError(f"Unknown target \"{target}\", (only allow \"values\", \"times\", \"type\", \"name\")")

    tracks = clip["animationClip"]["tracks"]
    res = {}

    for track in tracks:
        if target == "values" or target == "times":
            values = np.asarray(track[target], dtype=np.float64)
        else:
            values = track[target]

        type = track["type"]
        name = track["name"]
        if clean_names:
            name = name.replace(".quaternion", "")

        if target == "values":
            if type == "quaternion":
                res[name] = np.reshape(values, shape=(-1, 4))
            elif type == "number" and includes_blendshapes:
                res[name] = values
        else:
            res[name] = values

    return res


def unpack_anim_dict_to_tracks(names, **dicts):
    res = []
    for name, full_name in names.items():
        track = {}
        track["name"] = full_name
        for key, val in dicts.items():
            track[key] = val[name]
        res.append(track)
    return res


def pack_anim_to_array(clip, kinematics_skeleton, num_frames=None, includes_blendshapes=False, clean_names=True, input_as_anim_dict=False):
    """
    Packs a dictionary of named joint animations into a 3D NumPy array.\n
    If a joint is missing, it seamlessly falls back to the static bind rotation.
    """
    if input_as_anim_dict:
        anim_dict = clip
    else:
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


def extract_tracks(clip, types=None):
    """Extract a list of tracks with given type out of a clip."""
    all_tracks = clip["animationClip"]["tracks"]
    if type(types) is not list:
        types = [types]
    return [track for track in all_tracks if track["type"] in types]
