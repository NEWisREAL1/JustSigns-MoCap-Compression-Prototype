import numpy as np
import plotly.graph_objects as go
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp as S

from src.utils import glb_get_name_hierarchy, pack_frame_to_matrix

ALEX_MODEL_PATH = "model/Alex_Rig_v2.4_rokoko_wface_nov30.glb"

DEFAULT_ROTATION_PLOT_LAYOUT = dict(
    title="This is the Title",
    template="seaborn",
    width=1000, height=600,
    margin=dict(t=80, b=0, l=0, r=0),
)

DEFAULT_SKELETON_PLOT_LAYOUT = dict(
    title="This is the Title",
    template="simple_white",
    width=1000, height=600,
    margin=dict(t=80, b=0, l=0, r=0),
    scene=dict(
        aspectmode="data",
        yaxis=dict(title="Z"),
        zaxis=dict(title="Y"),    
    ),
    scene_camera=dict(
        # up=dict(x=0, y=1, z=0),
        eye=dict(x=1.75, y=1.75, z=1.75),
    ),
)


###
### ----- Rotation Plotting ----- ###
###


def _build_xyplane(fig, resolution=10):
    x = np.linspace(-1.2, 1.2, resolution)
    y = np.linspace(-1.2, 1.2, resolution)
    x_plane, y_plane = np.meshgrid(x, y)
    z_plane = np.zeros_like(x_plane)

    fig.add_trace(go.Surface(
        x=x_plane, y=y_plane, z=z_plane,
        colorscale=[(0, "#ababab"), (1, "#ababab")],
        opacity=0.06,
        showscale=False,
        contours={
            "x": {"show": True, "color": "grey", "width": 1},
            "y": {"show": True, "color": "grey", "width": 1},
            "z": {"show": True, "color": "grey", "width": 1}
        },
        hoverinfo="skip"
    ))


def _build_vector_arrow(fig, pos, color="red"):
    pos = pos / np.linalg.norm(pos)

    fig.add_trace(go.Cone(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        u=[pos[0]], v=[pos[1]], w=[pos[2]],
        anchor="tip",
        sizeref=0.3,
        sizemode="absolute",
        colorscale=[(0, color), (1, color)],
        showscale=False,
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0.7 * pos[0]], y=[0, 0.7 * pos[1]], z=[0, 0.7 * pos[2]],
        mode="lines",
        line=dict(width=15, color=color),
        showlegend=False,
    ))


def _build_path(fig, path, color="red"):
    fig.add_trace(go.Scatter3d(
        x=path[:,0], y=path[:,1], z=path[:,2],
        mode="lines",
        line=dict(width=6, color=color),
        showlegend=False,
    ))



def _build_sphere(fig, center=(0,0,0), r=1, resolution=50):
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi   = np.linspace(0, np.pi, resolution)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = r * np.sin(phi_grid) * np.cos(theta_grid) + center[0]
    y = r * np.sin(phi_grid) * np.sin(theta_grid) + center[1]
    z = r * np.cos(phi_grid) + center[2]

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[(0, "#4a90d9"), (1, "#4a90d9")],
        opacity=0.06,
        showscale=False,
        contours={
            "x": {"show": True, "color": "grey", "width": 1},
            "y": {"show": True, "color": "grey", "width": 1},
            "z": {"show": True, "color": "grey", "width": 1}
        },
        hoverinfo="skip"
    ))


def plot_static_quat(quats, colors, reference=[0, 0, 1], layout_options=None):
    """
    Visualization for rotation represented by a quaternion
    """
    reference = np.array(reference) / np.linalg.norm(reference)
    fig = go.Figure(layout=layout_options)

    _build_sphere(fig)
    _build_xyplane(fig)
    _build_vector_arrow(fig, reference, color="grey")

    times = np.linspace(0, 1, 50)
    for i, quat in enumerate(quats):
        rotation = R.from_quat(quat)
        identity = R.identity()
        slerp = S(times=[0,1], rotations=R.concatenate([identity, rotation]))
        path = slerp(times).apply(reference)
        _build_vector_arrow(fig, rotation.apply(reference), color=colors[i])
        _build_path(fig, path, color=colors[i])

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1,1]),
            yaxis=dict(range=[-1,1]),
            zaxis=dict(range=[-1,1]),
            aspectmode="cube",
        ),
    )
    fig.show()

###
### ----- Static Skeleton Plotting ----- ###
###


def _build_bones(pos_dict, bind_model):
    """
    Build bones for lines plotting
    """
    bones = []
    parent_name = list(bind_model.keys())[0]
    stack = [(parent_name, bind_model[parent_name])]

    while stack:
        parent_name, data = stack.pop()
        for child_name in data["children"]:
            bones.append(pos_dict[parent_name])
            bones.append(pos_dict[child_name])
            bones.append(np.array([np.nan, np.nan, np.nan]))
            stack.append((child_name, data["children"][child_name]))

    return np.array(bones)


def plot_static_skeleton(pos_dict, bind_model=None, joints_color="blue", bones_color="red", layout_options=None):
    """
    Plot a single static skeleton 🦴 ...
    """
    fig = go.Figure(layout=layout_options)
    points = pack_frame_to_matrix(pos_dict)
    # bones = _build_bones(
    #     pos_dict, 
    #     joint_names=joint_names, parent_indices=parent_indices, bones_offset=bones_offset
    # )

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 2],
            z=points[:, 1],
            marker=dict(size=5, color=joints_color),
            mode="markers",
            text=list(pos_dict.keys()),
            name="joint",
            showlegend=True,
        )
    )

    if bind_model is None:
        gltf = GLTF2().load(ALEX_MODEL_PATH)
        bind_model, _ = glb_get_name_hierarchy("rootx", gltf.nodes)

    bones = _build_bones(pos_dict, bind_model)

    fig.add_trace(
        go.Scatter3d(
            x=bones[:, 0],
            y=bones[:, 2],
            z=bones[:, 1],
            line=dict(width=10, color=bones_color),
            mode="lines",
            connectgaps=False,
            name="bone",
            hoverinfo="skip",
        )
    )

    fig.show()