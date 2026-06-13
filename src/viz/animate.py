import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

from src.model import get_alex_bind_model, get_paper_plane_mesh
from src.utils import extract_frame, pack_frame_to_matrix
from src.viz.static import DEFAULT_ROTATION_PLOT_LAYOUT, DEFAULT_SKELETON_PLOT_LAYOUT


class BaseAnimatedPlotter:
    """
    Base class for common animated plotter logics
    """

    def __init__(self, fps=30, stride=2, **fig_layout):
        self.fig = go.Figure(layout=fig_layout)
        self.static_traces = []
        self.frames_traces = []    # frames = [[trace, ...], ...]
        self.fps = fps
        self.stride = stride

        self.play_frame_settings = {
            "duration": 1000/self.fps, # ms
            "redraw": True
        }

        self.play_transition_settings = {
            "duration": 0, # ms
            # "easing": "linear"
        }

    
    def update_layout(self, layout_dict):
        self.fig.update_layout(layout_dict)
        return self


    def _build_fig(self):
        if len(self.static_traces) > 0:
            self.fig.add_traces(data=self.static_traces)


        if len(self.frames_traces) > 0:
            self.fig.add_traces(data=self.frames_traces[0])

            animated_start = len(self.static_traces)
            animated_count = len(self.frames_traces[0])
            frame_trace_indices = list(range(animated_start, animated_start + animated_count))

            self.fig.frames = [go.Frame(
                data=[trace for trace in self.frames_traces[f]],
                traces=frame_trace_indices,
                name=str(f),
            ) for f in range(0, len(self.frames_traces), self.stride)]


    def _build_buttons(self):
        play_button = {
            "label": "▶ Play",
            "method": "animate",
            "args": [
                None,  # None means play all frames in order
                {
                    "frame": self.play_frame_settings,
                    "transition": self.play_transition_settings,
                    "fromcurrent": True,  # Resume from current position if paused
                }
            ]
        }

        pause_button = {
            "label": "⏸ Pause",
            "method": "animate",
            "args": [
                [None],  # An empty list breaks the animation loop sequence
                {
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                    "mode": "immediate"  # Halt the active frame sequence instantly
                }
            ]
        }

        animation_menu = {
            "type": "buttons",
            "buttons": [play_button, pause_button],
            "direction": "left",        # Arrange buttons horizontally
            "pad": {"r": 10, "t": 10},  # Padding
            "showactive": False,        # Don't keep the button visually "pressed"
            "x": 0.5, "y": -0.05,
            "xanchor": "center",
            "yanchor": "top"
        }

        self.fig.update_layout(updatemenus=[animation_menu])


    def _build_slider(self):
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "center",
            # "currentvalue": {
            #     "font": {"size": 16},
            #     "prefix": "Frame: ",
            #     "visible": True,
            #     "xanchor": "center"
            # },
            "transition": self.play_transition_settings,
            "pad": {"b": 10, "t": 10},
            "len": 0.8,
            "x": 0.5, "y": -0.15,
            "steps": []
        }

        for f in range(0, len(self.frames_traces), self.stride):
            slider_step = {
                "args": [
                    [str(f)],  # This targets the specific frame name we set earlier
                    {
                        "frame": self.play_frame_settings,
                        "transition": self.play_transition_settings,
                        "mode": "immediate",
                    }
                ],
                "label": str(f),
                "method": "animate"
            }
            sliders_dict["steps"].append(slider_step)

        self.fig.update_layout(sliders=[sliders_dict])


    def build(self):
        self._build_fig()
        self._build_buttons()
        self._build_slider()


    def save_html(self, path, auto_build=True):
        if auto_build:
            self.build()
        self.fig.write_html(path, include_plotlyjs="cdn")


    def show(self, auto_build=True):
        if auto_build:
            self.build()
        self.fig.show()
        # return self


class AnimatedSkeletonsPlotter(BaseAnimatedPlotter):
    """
    Plotter wrapper for animated skeleton plots
    """

    def __init__(self, fps=30, stride=2, **fig_layout):
        super().__init__(fps=fps, stride=stride, **fig_layout)
        self.bind_model = get_alex_bind_model()
        self._bone_edges = self._precompute_bone_edges()


    def _precompute_bone_edges(self):
        """
        Precompute parent-child joint pairs once so frame rendering only packs coordinates.
        """
        edges = []
        root_name = list(self.bind_model.keys())[0]
        stack = [(root_name, self.bind_model[root_name])]

        while stack:
            parent_name, data = stack.pop()
            children = data["children"]

            for child_name, child_data in children.items():
                edges.append((parent_name, child_name))
                stack.append((child_name, child_data))

        return edges


    def _build_bones(self, pos_dict):
        """
        Build bones for lines plotting
        """
        bones = []

        for parent_name, child_name in self._bone_edges:
            if parent_name in pos_dict.keys() and child_name in pos_dict.keys():
                bones.append(pos_dict[parent_name])
                bones.append(pos_dict[child_name])
                bones.append(np.array([np.nan, np.nan, np.nan]))

        return np.array(bones)


    def add_skeleton_frames(
        self, pos_dict,
        joints_size=5, bones_width=10,
        offset=None, name=None, joints_color=None, bones_color=None
        ):
        first_key = list(pos_dict.keys())[0]
        n_frames = pos_dict[first_key].shape[0]
        
        if offset is not None:
            offset = np.array(offset)

        for _ in range(0, n_frames - len(self.frames_traces)):
            self.frames_traces.append([])

        for f in range(n_frames):
            f_pos_dict = extract_frame(pos_dict, f)      
            f_points = pack_frame_to_matrix(f_pos_dict)
            f_bones = self._build_bones(f_pos_dict)

            if offset is not None:
                f_points += offset
                f_bones  += offset

            f_joints_trace = go.Scatter3d(
                x=f_points[:, 0], y=f_points[:, 2], z=f_points[:, 1],
                marker=dict(size=joints_size, color=joints_color),
                mode="markers",
                text=list(pos_dict.keys()),
                name=f"{name} (joints)",
                showlegend=name is not None,
            )

            f_bones_trace = go.Scatter3d(
                x=f_bones[:, 0], y=f_bones[:, 2], z=f_bones[:, 1],
                line=dict(width=bones_width, color=bones_color),
                mode="lines",
                connectgaps=False,
                name=f"{name} (bones)",
                showlegend=name is not None,
                hoverinfo="skip",
            )

            self.frames_traces[f].append(f_joints_trace)
            self.frames_traces[f].append(f_bones_trace)

        return self


    def apply_defualt_layout(self):
        self.fig.update_layout(DEFAULT_SKELETON_PLOT_LAYOUT)
        return self


class AnimatedRotationsPlotter(BaseAnimatedPlotter):
    """
    Plotter wrapper for animated rotation plots
    """

    def __init__(self, fps=30, stride=2, **fig_layout):
        super().__init__(fps=fps, stride=stride, **fig_layout)
        self.model_vertices, self.model_faces = get_paper_plane_mesh(recenter=True, scale=0.25)
        self.model_vertices = R.from_euler(('xz'), (-90,-90), degrees=True).apply(self.model_vertices)
        self.ref_vector = np.array([[-1, 0, 0], [1, 0, 0]])
        # self.add_quat([0, 0, 0, 1], color="tan", model_opacity=ref_model_opcaity, name="ref")
        self._build_sphere()


    def _build_sphere(self, resolution=25):
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi   = np.linspace(0, np.pi, resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)

        self.static_traces.append(go.Surface(
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


    def add_quat_frames(self, quats, color=None, model_opacity=1, name=None):
        n_frames = np.array(quats).shape[0]

        for _ in range(0, n_frames - len(self.frames_traces)):
            self.frames_traces.append([])

        for f, quat in enumerate(quats):
            f_rotation = R.from_quat(quat)
            f_rotated_vertices = f_rotation.apply(self.model_vertices)
            f_rotated_axis = f_rotation.apply(self.ref_vector)

            f_model = go.Mesh3d(
                x=f_rotated_vertices[:, 0],
                y=f_rotated_vertices[:, 1],
                z=f_rotated_vertices[:, 2],
                i=self.model_faces[:, 0],
                j=self.model_faces[:, 1],
                k=self.model_faces[:, 2],
                color=color,
                opacity=model_opacity,
                showlegend=False,
            )

            f_axis = go.Scatter3d(
                x=f_rotated_axis[:, 0],
                y=f_rotated_axis[:, 1],
                z=f_rotated_axis[:, 2],
                mode="lines",
                line=dict(color=color, width=6),
                name=name,
                showlegend=name is not None,
            )

            f_axis_head = go.Scatter3d(
                x=[f_rotated_axis[1, 0]],
                y=[f_rotated_axis[1, 1]],
                z=[f_rotated_axis[1, 2]],
                mode="markers",
                marker=dict(color=color, size=6),
                showlegend=False,
            )

            self.frames_traces[f].append(f_model)
            self.frames_traces[f].append(f_axis)
            self.frames_traces[f].append(f_axis_head)

        return self


    def apply_defualt_layout(self):
        self.fig.update_layout(DEFAULT_ROTATION_PLOT_LAYOUT)
        return self


