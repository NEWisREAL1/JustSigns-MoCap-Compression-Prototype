import numpy as np

BODY = [
    ("hip","spine"), ("spine","chest"), ("chest","neck"), ("neck","head"),
    ("chest","leftShoulder"), ("leftShoulder","leftUpperArm"),
    ("leftUpperArm","leftLowerArm"), ("leftLowerArm","leftHand"),
    ("chest","rightShoulder"), ("rightShoulder","rightUpperArm"),
    ("rightUpperArm","rightLowerArm"), ("rightLowerArm","rightHand"),
    ("hip","leftUpLeg"), ("leftUpLeg","leftLeg"), ("leftLeg","leftFoot"),
    ("hip","rightUpLeg"), ("rightUpLeg","rightLeg"), ("rightLeg","rightFoot"),
]

SEG = {"metacarpal":0, "proximal":1, "intermediate":2, "medial":2, "distal":3, "tip":4}

LOWER_BODY_JOINTS = ["hip", "leftUpLeg" ,"leftLeg" ,"leftFoot" ,"leftToe" ,"leftToeEnd" ,"rightUpLeg" ,"rightLeg" ,"rightFoot" ,"rightToe" ,"rightToeEnd"]
LEFT_HAND_JOINTS = [
    "leftHand",
    "leftThumbProximal", "leftThumbMedial", "leftThumbDistal", "leftThumbTip", 
    "leftIndexProximal", "leftIndexMedial", "leftIndexDistal", "leftIndexTip", 
    "leftMiddleProximal", "leftMiddleMedial", "leftMiddleDistal", "leftMiddleTip", 
    "leftRingProximal", "leftRingMedial", "leftRingDistal", "leftRingTip", 
    "leftLittleProximal", "leftLittleMedial", "leftLittleDistal", "leftLittleTip",
]
RIGHT_HAND_JOINTS = [
    "rightHand",
    "rightThumbProximal", "rightThumbMedial", "rightThumbDistal", "rightThumbTip", 
    "rightIndexProximal", "rightIndexMedial", "rightIndexDistal", "rightIndexTip", 
    "rightMiddleProximal", "rightMiddleMedial", "rightMiddleDistal", "rightMiddleTip", 
    "rightRingProximal", "rightRingMedial", "rightRingDistal", "rightRingTip", 
    "rightLittleProximal", "rightLittleMedial", "rightLittleDistal", "rightLittleTip",
]

def build_defualt_bones(names):
    have = {n.lower(): n for n in names}
    bones = [(a, b) for a, b in BODY if a.lower() in have and b.lower() in have]

    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "little", "pinky"):
            chain = sorted((have[n] for n in have if n.startswith(side + finger)),
                        key=lambda j: next((r for k, r in SEG.items() if k in j.lower()), 99))
            chain = ([have[side + "hand"]] if side + "hand" in have else []) + chain
            bones += [(chain[i], chain[i+1]) for i in range(len(chain) - 1)]

    return bones


def stack_clip_for_skeleton_anim(clip, return_qunts=False):
    if return_qunts:
        return np.stack([clip[n][:, :3] for n in clip.keys()], axis=1), np.stack([clip[n][:, 3:] for n in clip.keys()], axis=1) 
    return np.stack([clip[n][:, :3] for n in clip.keys()], axis=1)