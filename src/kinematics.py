import numpy as np
from scipy.spatial.transform import Rotation as R

from src.model import get_alex_bind_model


def forward_kinematics(anim_quats, bind_model=None):
    """
    Calculate global positions using a nested dictionary bind model.
    Supports both single-frame (1D) and multi-frame (2D) quaternion inputs.
    
    bind_model: dict (The parsed JSON tree)
    anim_quats: dict { 'joint_name.quaternion': np.array (F, 4) OR (4,) }
    returns: dict { 'joint_name': np.array (F, 3) OR (3,) }
    """
    if bind_model is None:
        bind_model = get_alex_bind_model()

    if not anim_quats:
        return {}
        
    global_positions = {}
    global_rotations = {}

    first_key = list(anim_quats.keys())[0]
    first_val = np.asarray(anim_quats[first_key])
    
    is_single_frame = (first_val.ndim == 1)
    F = 1 if is_single_frame else first_val.shape[0]

    root_name = list(bind_model.keys())[0]
    stack = [(root_name, bind_model[root_name], None)]

    while stack:
        joint, data, parent = stack.pop()
        base_trans = np.array(data['translation'])

        parent_is_nan = False
        if parent is not None:
            parent_pos = global_positions[parent]
            parent_is_nan = np.isnan(parent_pos).any()

        # anim_key = f"{joint}.quaternion"
        if joint in anim_quats:
            local_q = np.asarray(anim_quats[joint])
            
            if is_single_frame:
                local_q = local_q.reshape(1, 4) 
            has_nan_quat = np.isnan(local_q).any()

            if has_nan_quat:
                local_rot = None
            else:
                local_rot = R.from_quat(local_q)
        else:
            local_rot = R.from_quat(np.tile(data['rotation'], (F, 1)))

        if parent is None:
            if local_rot is None:
                global_positions[joint] = np.full((F, 3), np.nan)
                global_rotations[joint] = None
            else:
                global_positions[joint] = np.tile(base_trans, (F, 1))
                global_rotations[joint] = local_rot
        else:
            parent_rot = global_rotations[parent]

            if parent_is_nan or parent_rot is None or local_rot is None:
                global_positions[joint] = np.full((F, 3), np.nan)
                global_rotations[joint] = None
            else:
                global_rotations[joint] = parent_rot * local_rot
                rotated_offset = parent_rot.apply(base_trans)
                global_positions[joint] = parent_pos + rotated_offset

        children = data.get('children', {})
        for child_name in reversed(list(children.keys())):
            stack.append((child_name, children[child_name], joint))

    if is_single_frame:
        for key in global_positions:
            global_positions[key] = global_positions[key][0]

    return global_positions