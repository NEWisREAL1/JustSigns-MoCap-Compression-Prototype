import numpy as np


def load_clip(path):
    """Load a full npz clip as json/dict"""
    data = np.load(path, allow_pickle=True)
    clip = data["clip"].item()
    return clip


def extract_tracks(clip, type=None):
    """Get list of tracks with given type (e.g. \"quaternion\" or \"number\")"""
    tracks = clip["animationClip"]["tracks"]

    if type is None:
        return tracks

    subset = [ tracks[i] for i in range(len(tracks)) if tracks[i]["type"] == type ]
    return subset


def get_joints_dict(clip, clean_names=True, include_blendshapes=False):
    """
    Extract joint tracks and parse into the dict of format {joint_name: value}.\n
    Also reshape the value according to track type, e.g., (F, 4) for quaternion, (F,) for number.
    """
    tracks = clip["animationClip"]["tracks"]
    res = {}

    for track in tracks:
        type = track["type"]
        name = track["name"]
        if clean_names:
            name = name.replace(".quaternion", "")

        if type != "number" or include_blendshapes:
            res[name] = {}
            res[name]["type"] = type
            values = np.asarray(track["values"], dtype=np.float64)

            if track["type"] == "quaternion":
                res[name]["values"] = np.reshape(values, shape=(-1, 4))
            
            elif track["type"] == "number" and include_blendshapes:
                res[name]["values"] = values

    return res


def pack_anim_dict_to_array(anim_dict, joint_names, bind_rotations, num_frames=None):
    """
    Packs a dictionary of named joint animations into a 3D NumPy array.
    If a joint is missing, it seamlessly falls back to the static bind rotation.
    """
    if num_frames is None:
        if not anim_dict:
            raise ValueError("anim_dict is empty and num_frames not provided.")
        num_frames = next(iter(anim_dict.values()))["values"].shape[0]
        
    J = len(joint_names)
    anim_rotations = np.zeros((num_frames, J, 4), dtype=np.float64)
    
    for j, name in enumerate(joint_names):
        if name in anim_dict:
            # Squeeze and shape safely handle both single-frame and multi-frame
            local_q = np.asarray(anim_dict[name]["values"])
            if local_q.ndim == 1:
                local_q = local_q.reshape(1, 4)
            anim_rotations[:, j, :] = local_q
        else:
            # PERFECT FALLBACK: Use the bind rotation from the JSON!
            anim_rotations[:, j, :] = bind_rotations[j]
            
    return anim_rotations