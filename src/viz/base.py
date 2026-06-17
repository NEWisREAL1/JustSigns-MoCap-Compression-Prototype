import plotly.graph_objects as go

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