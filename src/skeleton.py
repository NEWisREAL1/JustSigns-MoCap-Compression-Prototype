########## ---------- -------------------------- ---------- ##########
########## ---------- ARCHIVAL (Not Use Anymore) ---------- ##########
########## ---------- -------------------------- ---------- ##########

import numpy as np

JOINT_NAMES = [
    # central body (root should be hip/pelvis)
    'rootx', 'spine_01x', 'spine_02x', 'neckx', 'headx',
    
    # right arm
    'shoulderr', 'arm_stretchr', 'arm_twistr', 'forearm_stretchr', 'handr',
    
    # right fingers
    'thumb1r', 'thumb2r', 'index1r', 'index2r', 'index3r',
    'middle1r', 'middle2r', 'middle3r', 'ring1r', 'ring2r', 'ring3r',
    'pinky1r', 'pinky2r', 'pinky3r',
    
    # left arm
    'shoulderl', 'arm_stretchl', 'forearm_stretchl', 'handl',
    
    # left fingers
    'thumb1l', 'thumb2l', 'index1l', 'index2l', 'index3l',
    'middle1l', 'middle2l', 'middle3l', 'ring1l', 'ring2l', 'ring3l',
    'pinky1l', 'pinky2l', 'pinky3l',
    
    # legs & feet
    'thigh_stretchl', 'leg_stretchl', 'footl',
    'thigh_stretchr', 'leg_stretchr', 'footr'
]


PARENT_INDICES = [
         # this joint           <- parent joint
    -1,  # 0: rootx
     0,  # 1: spine_01x         <- rootx
     1,  # 2: spine_02x         <- spine_01x
     2,  # 3: neckx             <- spine_02x
     3,  # 4: headx             <- neckx
     2,  # 5: shoulderr         <- spine_02x
     5,  # 6: arm_stretchr      <- shoulderr
     6,  # 7: forearm_stretchr  <- arm_stretchr
     7,  # 8: handr             <- forearm_stretchr
     8,  # 9: thumb1r           <- handr (นิ้วแตกออกจากมือ)
     9,  # 10: thumb2r          <- thumb1r
     8,  # 11: index1r          <- handr
    11,  # 12: index2r          <- index1r
    12,  # 13: index3r          <- index2r
     8,  # 14: middle1r         <- handr
    14,  # 15: middle2r         <- middle1r
    15,  # 16: middle3r         <- middle2r
     8,  # 17: ring1r           <- handr
    17,  # 18: ring2r           <- ring1r
    18,  # 19: ring3r           <- ring2r
     8,  # 20: pinky1r          <- handr
    20,  # 21: pinky2r          <- pinky1r
    21,  # 22: pinky3r          <- pinky2r
     2,  # 23: shoulderl        <- spine_02x
    23,  # 24: arm_stretchl     <- shoulderl
    24,  # 25: forearm_stretchl <- arm_stretchl
    25,  # 26: handl            <- forearm_stretchl
    26,  # 27: thumb1l          <- handl
    27,  # 28: thumb2l          <- thumb1l
    26,  # 29: index1l          <- handl
    29,  # 30: index2l          <- index1l
    30,  # 31: index3l          <- index2l
    26,  # 32: middle1l         <- handl
    32,  # 33: middle2l         <- middle1l
    33,  # 34: middle3l         <- middle2l
    26,  # 35: ring1l           <- handl
    35,  # 36: ring2l           <- ring1l
    36,  # 37: ring3l           <- ring2l
    26,  # 38: pinky1l          <- handl
    38,  # 39: pinky2l          <- pinky1l
    39,  # 40: pinky3l          <- pinky2l
     0,  # 41: thigh_stretchl   <- rootx (hip)
    41,  # 42: leg_stretchl     <- thigh_stretchl
    42,  # 43: footl            <- leg_stretchl
     0,  # 44: thigh_stretchr   <- rootx (hip)
    44,  # 45: leg_stretchr     <- thigh_stretchr
    45,  # 46: footr            <- leg_stretchr
]

# Standardized 1.70m - 1.75m Human T-Pose (1.0 = 1 Meter)
# Standardized 1.70m - 1.75m Human T-Pose (1.0 = 1 Meter)
BONES_OFFSET = np.array([
    # Central Axis (Y-Up)
    [0.0,   0.0,   0.0],    # 0: rootx (Pelvis center)
    [0.0,   0.15,  0.0],    # 1: spine_01 (Lower back)
    [0.0,   0.20,  0.0],    # 2: spine_02 (Chest)
    [0.0,   0.15,  0.0],    # 3: neck (Base of neck)
    [0.0,   0.12,  0.0],    # 4: head (Center of head)
    
    # Right Arm (-X axis)
    [-0.18, 0.05,  0.0],    # 5: shoulderr (Collarbone to shoulder joint)
    [-0.28, 0.0,   0.0],    # 6: arm_stretchr (Upper arm)
    [-0.26, 0.0,   0.0],    # 7: forearm_stretchr (Lower arm)
    [-0.10, 0.0,   0.0],    # 8: handr (Wrist)
    
    # Right Hand Fingers (Palms down)
    # The first joint (e.g., index1r) spans the entire palm (metacarpals)
    [-0.04,  -0.03,  0.04], # 9: thumb1r  (Wrist to base of thumb)
    [-0.035, -0.01,  0.02], # 10: thumb2r (Thumb proximal to distal)
    
    [-0.09,   0.0,   0.03], # 11: index1r (Wrist to index knuckle - THE PALM)
    [-0.04,   0.0,   0.0],  # 12: index2r (Index proximal)
    [-0.025,  0.0,   0.0],  # 13: index3r (Index middle/distal)
    
    [-0.095,  0.0,   0.0],  # 14: middle1r(Wrist to middle knuckle - THE PALM)
    [-0.045,  0.0,   0.0],  # 15: middle2r(Middle proximal)
    [-0.03,   0.0,   0.0],  # 16: middle3r(Middle middle/distal)
    
    [-0.09,   0.0,  -0.02], # 17: ring1r  (Wrist to ring knuckle - THE PALM)
    [-0.04,   0.0,   0.0],  # 18: ring2r  (Ring proximal)
    [-0.025,  0.0,   0.0],  # 19: ring3r  (Ring middle/distal)
    
    [-0.08,   0.0,  -0.04], # 20: pinky1r (Wrist to pinky knuckle - THE PALM)
    [-0.035,  0.0,   0.0],  # 21: pinky2r (Pinky proximal)
    [-0.02,   0.0,   0.0],  # 22: pinky3r (Pinky middle/distal)

    # Left Arm (+X axis)
    [ 0.18,  0.05, 0.0],    # 23: shoulderl
    [ 0.28,  0.0,  0.0],    # 24: arm_stretchl
    [ 0.26,  0.0,  0.0],    # 25: forearm_stretchl
    [ 0.10,  0.0,  0.0],    # 26: handl
    
    # Left Hand Fingers (Palms down)
    [ 0.04,  -0.03,  0.04], # 27: thumb1l 
    [ 0.035, -0.01,  0.02], # 28: thumb2l 
    
    [ 0.09,   0.0,   0.03], # 29: index1l 
    [ 0.04,   0.0,   0.0],  # 30: index2l 
    [ 0.025,  0.0,   0.0],  # 31: index3l 
    
    [ 0.095,  0.0,   0.0],  # 32: middle1l
    [ 0.045,  0.0,   0.0],  # 33: middle2l
    [ 0.03,   0.0,   0.0],  # 34: middle3l
    
    [ 0.09,   0.0,  -0.02], # 35: ring1l  
    [ 0.04,   0.0,   0.0],  # 36: ring2l  
    [ 0.025,  0.0,   0.0],  # 37: ring3l  
    
    [ 0.08,   0.0,  -0.04], # 38: pinky1l 
    [ 0.035,  0.0,   0.0],  # 39: pinky2l 
    [ 0.02,   0.0,   0.0],  # 40: pinky3l 

    # Left Leg (+X offset for hip, straight down -Y)
    [ 0.12, -0.05, 0.0],    # 41: thigh_stretchl
    [ 0.0,  -0.42, 0.0],    # 42: leg_stretchl
    [ 0.0,  -0.40, 0.0],    # 43: footl

    # Right Leg (-X offset for hip, straight down -Y)
    [-0.12, -0.05, 0.0],    # 44: thigh_stretchr
    [ 0.0,  -0.42, 0.0],    # 45: leg_stretchr
    [ 0.0,  -0.40, 0.0],    # 46: footr
])