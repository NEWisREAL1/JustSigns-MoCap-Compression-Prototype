import numpy as np


class KinematicsSkeleton:
    def __init__(self, parents, offsets):
        self.num_joints = len(parents)
        self.parents = np.array(parents, dtype=np.int32)
        self.offsets = np.atleast_2d(offsets).astype(np.float64)

    def forward_kinematics(self, anim_rotations, root_positions=None):
        m = anim_rotations.shape[0]
        J = self.num_joints
        
        global_rotations = np.empty((m, J, 4), dtype=np.float64)
        global_positions = np.empty((m, J, 3), dtype=np.float64)
        
        # Root is just the absolute animated rotation
        global_rotations[:, 0, :] = anim_rotations[:, 0, :]
        
        if root_positions is not None:
            global_positions[:, 0, :] = root_positions
        else:
            # FIX: Ensure root uses its bind translation, not just [0,0,0]
            global_positions[:, 0, :] = self.offsets[0]

        for j in range(1, J):
            p = self.parents[j]
            # Since anim_rotations are absolute local rotations, we just multiply by parent global
            global_rotations[:, j, :] = self._quat_mult_batch(global_rotations[:, p, :], anim_rotations[:, j, :])
            
            rotated_offset = self._rotate_vector_batch(global_rotations[:, p, :], self.offsets[j])
            global_positions[:, j, :] = global_positions[:, p, :] + rotated_offset
            
        return global_positions, global_rotations

    @staticmethod
    def _quat_mult_batch(q1, q2):
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        res = np.empty_like(q1)
        res[:, 0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        res[:, 1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        res[:, 2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        res[:, 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return res

    @staticmethod
    def _rotate_vector_batch(q, v):
        u = q[:, :3]
        w = q[:, 3:4]
        v_broadcast = np.tile(v, (q.shape[0], 1))
        uv = np.cross(u, v_broadcast)
        uuv = np.cross(u, uv)
        return v_broadcast + 2.0 * w * uv + 2.0 * uuv