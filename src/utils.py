from copy import deepcopy

import numpy as np


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
            print(f"list of {type(data[key][0]).__name__}")
        else:
            print(f"{val_type.__name__}")


def parse_clip(clip):
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

        new_tracks[track["name"]] = dict(type=type, times=times, values=values)

        if track["type"].lower() == "quaternion":
            quats = new_tracks[track["name"]]["values"]
            quats = np.reshape(quats, shape=(len(times), 4))
            new_tracks[track["name"]]["values"] = quats

    new_clip["animationClip"]["tracks"] = new_tracks
    return new_clip
