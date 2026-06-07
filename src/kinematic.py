"""Forward kinematics for the quaternion (Three.js) clip format.

Local quaternions carry no position; a skeleton only exists after composing
each joint's local rotation down the hierarchy onto a rest skeleton. This
module supplies the hierarchy (read off the joint names), a synthetic T-pose
of rest offsets, and the FK that turns {joint: (T,4) quaternions} into world
positions ready for AnimatedPlotter.add_skeleton.

The hierarchy is exact. The rest offsets here are a plausible humanoid guess,
NOT the rig's real values -- see note in default_rest_offsets().
"""
import numpy as np

# right-side + centre parents; left side is mirrored from this
_RIGHT_AND_CENTRE = {
    "rootx": None,
    "spine_01x": "rootx", "spine_02x": "spine_01x", "neckx": "spine_02x", "headx": "neckx",
    "shoulderr": "spine_02x", "arm_stretchr": "shoulderr",
    "forearm_stretchr": "arm_stretchr", "handr": "forearm_stretchr",
    "thumb1r": "handr", "thumb2r": "thumb1r",
    "index1r": "handr", "index2r": "index1r", "index3r": "index2r",
    "middle1r": "handr", "middle2r": "middle1r", "middle3r": "middle2r",
    "ring1r": "handr", "ring2r": "ring1r", "ring3r": "ring2r",
    "pinky1r": "handr", "pinky2r": "pinky1r", "pinky3r": "pinky2r",
    "thigh_stretchr": "rootx", "leg_stretchr": "thigh_stretchr", "footr": "leg_stretchr",
}


def _mirror_name(n):
    return n[:-1] + "l" if n and n.endswith("r") else n


def _build_parents():
    parents = dict(_RIGHT_AND_CENTRE)
    for child, par in _RIGHT_AND_CENTRE.items():
        if child.endswith("r"):
            parents[_mirror_name(child)] = _mirror_name(par)
    return parents


PARENTS = _build_parents()       # all 47 joints


def default_rest_offsets():
    """A synthetic T-pose (Y-up, metres-ish), mirrored R->L.

    NOTE: these are a generic humanoid guess, not your rig's real offsets, so
    the absolute figure will look approximate (limbs may point slightly wrong,
    since the true offsets live in the rig's local frames). It's enough to
    validate the pipeline and to *see motion*; for an anatomically faithful
    figure, read the offsets from the GLB (each bone's rest position) or view
    in Three.js. Crucially, for codec QA the offsets cancel -- GT and recon use
    the same FK -- so the error you see between two skeletons is still exact.
    """
    base = {
        "rootx": (0, 0, 0),
        "spine_01x": (0, 0.16, 0), "spine_02x": (0, 0.18, 0),
        "neckx": (0, 0.20, 0), "headx": (0, 0.12, 0),
        "shoulderr": (0.05, 0.14, 0), "arm_stretchr": (0.27, 0, 0),
        "forearm_stretchr": (0.25, 0, 0), "handr": (0.08, 0, 0),
        "thumb1r": (0.03, 0, 0.025), "thumb2r": (0.03, 0, 0.015),
        "index1r": (0.08, 0, 0.02), "index2r": (0.025, 0, 0), "index3r": (0.02, 0, 0),
        "middle1r": (0.08, 0, 0.005), "middle2r": (0.03, 0, 0), "middle3r": (0.022, 0, 0),
        "ring1r": (0.075, 0, -0.01), "ring2r": (0.028, 0, 0), "ring3r": (0.02, 0, 0),
        "pinky1r": (0.07, 0, -0.025), "pinky2r": (0.022, 0, 0), "pinky3r": (0.018, 0, 0),
        "thigh_stretchr": (0.09, -0.06, 0), "leg_stretchr": (0, -0.42, 0), "footr": (0, -0.40, 0.06),
    }
    offsets = dict(base)
    for k, (x, y, z) in base.items():
        if k.endswith("r"):
            offsets[_mirror_name(k)] = (-x, y, z)
    return offsets


def topo_order(parents):
    """Joints ordered so every parent precedes its children."""
    order, seen = [], set()

    def visit(b):
        if b in seen:
            return
        p = parents.get(b)
        if p is not None and p in parents:
            visit(p)
        seen.add(b)
        order.append(b)

    for b in parents:
        visit(b)
    return order


def forward_kinematics(quats, parents=PARENTS, offsets=None, root_pos=None):
    """Compose local quaternions into world joint positions.

    quats    : {joint: (T, 4)} local quaternions [x, y, z, w]
    offsets  : {joint: (3,)} rest translation in parent's frame (defaults to
               the synthetic T-pose above)
    root_pos : optional (T, 3) world translation for the root

    Returns (order, P, bones): joint order, (T, J, 3) world positions, and the
    bone list (parent, child) ready for AnimatedPlotter.add_skeleton.
    """
    from scipy.spatial.transform import Rotation as Rot
    offsets = offsets or default_rest_offsets()
    order = [b for b in topo_order(parents) if b in quats]
    T = len(np.asarray(next(iter(quats.values()))))

    M = {}
    for b in order:
        L = np.broadcast_to(np.eye(4), (T, 4, 4)).copy()
        L[:, :3, :3] = Rot.from_quat(np.asarray(quats[b], dtype=float)).as_matrix()
        L[:, :3, 3] = offsets.get(b, (0.0, 0.0, 0.0))
        p = parents.get(b)
        if p is None or p not in M:
            if p is None and root_pos is not None:
                L[:, :3, 3] += np.asarray(root_pos, dtype=float)
            M[b] = L
        else:
            M[b] = M[p] @ L

    P = np.stack([M[b][:, :3, 3] for b in order], axis=1)
    bones = [(parents[b], b) for b in order if parents.get(b) in M]
    return order, P, bones