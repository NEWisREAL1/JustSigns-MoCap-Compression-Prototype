from copy import deepcopy

import numpy as np

#####
##### General
#####


def print_json_structure(data, level=0):
    """
    Simple json format printer
    """
    for key in data.keys():
        val_type = type(data[key])

        print(("    " * level) + f"{key}: ", end="")

        if val_type is dict:
            print("")
            print_json_structure(data[key], level=level+1)
        elif val_type is list:
            list_type = type(data[key][0]).__name__ if len(data[key]) > 0 else "???"
            print(f"list of {list_type}")
        else:
            print(f"{val_type.__name__}")


#####
##### MoCap Clip Processing
#####


def parse_clip(clip, clean_names=True):
    """
    Parse a MoCap clip to use joint names as tracks keys 
    and arrange quaternions as (F,4) array for easier navigation
    """
    new_clip = deepcopy(clip)
    tracks = new_clip["animationClip"]["tracks"]
    new_tracks = {}

    for track in tracks:
        type   = track["type"]
        times  = np.array(track["times"], dtype=float)
        values = np.array(track["values"], dtype=float)
        track_name = track["name"]
        if clean_names:
            track_name = track_name.replace(".quaternion", "")

        new_tracks[track_name] = dict(type=type, times=times, values=values)

        if type == "quaternion":
            quats = new_tracks[track_name]["values"]
            quats = np.reshape(quats, shape=(len(times), 4))
            new_tracks[track_name]["values"] = quats

    new_clip["animationClip"]["tracks"] = new_tracks
    return new_clip


def extract_quats(clip, parse=False):
    """
    Extract only quaternions of every joints from a parsed MoCap clip
    return as a dict { joint_name: np.array(F,4) }
    """
    quats = {}
    if parse:
        tracks = parse_clip(clip)["animationClip"]["tracks"]
    else:
        tracks = clip["animationClip"]["tracks"]
        

    for key, track in tracks.items():
        if track["type"].lower() == "quaternion":
            quats[key] = track["values"]
    
    return quats


def extract_frame(values, frame):
    """
    Extract a single frame out of joints dict, can be use for both quat and pos
    quants is a dict { joint_name: np.array(F,D) }
    return a dict { joint_name: np.array(D) }
    """
    frame_values = {}

    for joint, all_values in values.items():
        frame_values[joint] = all_values[frame]

    return frame_values


def pack_frame_to_matrix(frame_dict):
    """
    Pack a dict { joint_name: np.array(D) } representing single frame into np.array(J,D)
    """
    keys = list(frame_dict.keys())
    J, D = len(keys), len(frame_dict[keys[0]])
    matrix = np.empty(shape=(J, D))

    for i, key in enumerate(keys):
        matrix[i] = frame_dict[key]

    return matrix


#####
##### GLB Model Processing
#####

def glb_find_node(nodes, name):
    """
    Find and return a node of specific name
    """
    for node in nodes:
        if node.name.replace(".", "") == name:
            return node
    return None


def glb_get_name_hierarchy(root_name, nodes, hierarchy=dict(), names=[]):
    """
    Get a hierarchical dict of joint names
    """
    root_name = root_name.replace(".", "")
    names.append(root_name)
    parent_node = glb_find_node(nodes, root_name)
    hierarchy[root_name] = dict(
        translation = np.array(parent_node.translation, dtype=np.float64),
        rotation = np.array(parent_node.rotation, dtype=np.float64),
        children = dict()
    )

    for child_idx in parent_node.children:
        child_name = nodes[child_idx].name
        if not child_name.startswith("c_"):
            glb_get_name_hierarchy(child_name, nodes, hierarchy=hierarchy[root_name]["children"])

    return hierarchy, names

