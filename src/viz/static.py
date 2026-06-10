import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp as S

from src.model import get_alex_bind_model, get_paper_plane_mesh
from src.utils import pack_frame_to_matrix

ALEX_MODEL_PATH = "model/Alex_Rig_v2.4_rokoko_wface_nov30.glb"

DEFAULT_ROTATION_PLOT_LAYOUT = dict(
    # title="This is the Title",
    template="seaborn",
    width=1000, height=600,
    margin=dict(t=80, b=10, l=10, r=10),
    scene=dict(aspectmode="data"),
    scene_camera=dict(
        # up=dict(x=0, y=1, z=0),
        eye=dict(x=1.1, y=1.1, z=1.1),
    ),
)

DEFAULT_SKELETON_PLOT_LAYOUT = dict(
    # title="This is the Title",
    template="simple_white",
    width=1000, height=600,
    margin=dict(t=80, b=10, l=10, r=10),
    scene=dict(
        aspectmode="data",
        yaxis=dict(title="Z"),
        zaxis=dict(title="Y"),    
    ),
    scene_camera=dict(
        # up=dict(x=0, y=1, z=0),
        eye=dict(x=1.85, y=1.85, z=1.85),
    ),
)


class BaseStaticPlotter:
    """
    Base class for common static plotter logics
    """

    def __init__(self, **fig_layout):
        self.fig = go.Figure(layout=fig_layout)

        
    def update_layout(self, layout_dict):
        self.fig.update_layout(layout_dict)
        return self


    def show(self):
        self.fig.show()
        # return self


class StaticSkeletonsPlotter(BaseStaticPlotter):
    """
    Plotter wrapper for static skeleton plots
    """

    def __init__(self, **fig_layout):
        super().__init__(**fig_layout)
        self.bind_model = get_alex_bind_model()


    def _build_bones(self, pos_dict):
        """
        Build bones for lines plotting
        """
        bones = []
        parent_name = list(self.bind_model.keys())[0]
        stack = [(parent_name, self.bind_model[parent_name])]

        while stack:
            parent_name, data = stack.pop()
            for child_name in data["children"]:
                bones.append(pos_dict[parent_name])
                bones.append(pos_dict[child_name])
                bones.append(np.array([np.nan, np.nan, np.nan]))
                stack.append((child_name, data["children"][child_name]))

        return np.array(bones)


    def add_skeleton(
        self, pos_dict, 
        joints_size=5, bones_width=10,
        offset=None, name=None, joints_color=None, bones_color=None
        ):
        points = pack_frame_to_matrix(pos_dict)
        bones = self._build_bones(pos_dict)

        if offset is not None:
            offset = np.array(offset)
            points += offset
            bones  += offset

        self.fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 2], z=points[:, 1],
            marker=dict(size=joints_size, color=joints_color),
            mode="markers",
            text=list(pos_dict.keys()),
            name=f"{name} (joints)",
            showlegend=name is not None,
        ))


        self.fig.add_trace(go.Scatter3d(
            x=bones[:, 0], y=bones[:, 2], z=bones[:, 1],
            line=dict(width=bones_width, color=bones_color),
            mode="lines",
            connectgaps=False,
            name=f"{name} (bones)",
            showlegend=name is not None,
            hoverinfo="skip",
        ))

        return self


    def apply_defualt_layout(self):
        self.fig.update_layout(DEFAULT_SKELETON_PLOT_LAYOUT)
        return self


class StaticRotationsPlotter(BaseStaticPlotter):
    """
    Plotter wrapper for static rotation plots
    """

    def __init__(self, ref_model_opcaity=0.2, **fig_layout):
        super().__init__(**fig_layout)
        self.model_vertices, self.model_faces = get_paper_plane_mesh(recenter=True, scale=0.25)
        self.model_vertices = R.from_euler(('xz'), (-90,-90), degrees=True).apply(self.model_vertices)
        self.ref_vector = np.array([[-1, 0, 0], [1, 0, 0]])

        self.add_quat([0, 0, 0, 1], color="tan", model_opacity=ref_model_opcaity, name="ref")
        self._build_sphere()


    def _build_sphere(self, resolution=25):
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi   = np.linspace(0, np.pi, resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)

        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[(0, "#4a90d9"), (1, "#4a90d9")],
            opacity=0.0,
            showscale=False,
            contours=dict(
                x=dict(show=True, color="grey", width=1),
                y=dict(show=True, color="grey", width=1),
                z=dict(show=True, color="grey", width=1)
            ),
            hoverinfo="skip"
        ))


    def add_quat(self, quat, color=None, model_opacity=1, name=None):
        rotation = R.from_quat(quat)
        rotated_vertices = rotation.apply(self.model_vertices)
        rotated_axis = rotation.apply(self.ref_vector)

        self.fig.add_trace(go.Mesh3d(
            x=rotated_vertices[:, 0],
            y=rotated_vertices[:, 1],
            z=rotated_vertices[:, 2],
            i=self.model_faces[:, 0],
            j=self.model_faces[:, 1],
            k=self.model_faces[:, 2],
            color=color,
            opacity=model_opacity,
            showlegend=False,
        ))

        self.fig.add_trace(go.Scatter3d(
            x=rotated_axis[:, 0],
            y=rotated_axis[:, 1],
            z=rotated_axis[:, 2],
            mode="lines",
            line=dict(color=color, width=6),
            name=name,
            showlegend=name is not None,
        ))
        self.fig.add_trace(go.Scatter3d(
            x=[rotated_axis[1, 0]],
            y=[rotated_axis[1, 1]],
            z=[rotated_axis[1, 2]],
            mode="markers",
            marker=dict(color=color, size=6),
            showlegend=False,
        ))


    def apply_defualt_layout(self):
        self.fig.update_layout(DEFAULT_ROTATION_PLOT_LAYOUT)
        return self

