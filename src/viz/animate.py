import numpy as np
import plotly.graph_objects as go

from src.model import get_alex_bind_model
from src.utils import extract_frame, pack_frame_to_matrix
from src.viz.static import DEFAULT_SKELETON_PLOT_LAYOUT


class AnimatedSkeletonsPlotter:
    """
    Plotter wrapper for animated skeleton plots
    """

    def __init__(self, fps=30, stride=2, **fig_layout):
        self.fig = go.Figure(layout=fig_layout)
        self.bind_model = get_alex_bind_model()
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


    def add_skeleton_frames(
        self, pos_dict,
        joints_size=5, bones_width=10,
        offset=None, name=None, joints_color=None, bones_color=None
        ):
        first_key = list(pos_dict.keys())[0]
        n_frames = pos_dict[first_key].shape[0]

        for _ in range(0, n_frames - len(self.frames_traces)):
            self.frames_traces.append([])

        for f in range(n_frames):
            f_pos_dict = extract_frame(pos_dict, f)      
            f_points = pack_frame_to_matrix(f_pos_dict)
            f_bones = self._build_bones(f_pos_dict)

            if offset is not None:
                offset = np.array(offset)
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


    def update_layout(self, layout_dict):
        self.fig.update_layout(layout_dict)
        return self


    def apply_defualt_layout(self):
        self.fig.update_layout(DEFAULT_SKELETON_PLOT_LAYOUT)
        return self


    def _build_fig(self):
        self.fig.add_traces(data=self.frames_traces[0])
        self.fig.frames = [go.Frame(
            data=[trace for trace in self.frames_traces[f]],
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


    def show(self, auto_build=True):
        if auto_build:
            self.build()
        self.fig.show()
        # return self
