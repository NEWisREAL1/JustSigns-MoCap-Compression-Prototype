import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def signals_compare_plot(gt, approx, figsize=(16, 4), sharey=False):
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=sharey)
    axes_label=["X", "Y", "Z"]

    for i, ax in enumerate(axs):
        ax.plot(gt[:,i], label="ground truth")
        ax.plot(approx[:,i], label="approximate")
        ax.set_title(f"{axes_label[i]} Signal")
        ax.set_xlabel("frame")
        ax.set_ylabel(f"{axes_label[i]} value")
        ax.legend()
        ax.grid()

    plt.show()


def signals_error_plot(gt, approx, abs=True, figsize=(16, 4), sharey=True):
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=sharey)
    axes_label=["X", "Y", "Z"]

    for i, ax in enumerate(axs):
        err = gt[:,i] - approx[:,i]
        ax.plot(err if not abs else np.abs(err), 'r-')
        ax.set_title(f"{axes_label[i]} Error")
        ax.set_xlabel("frame")
        ax.set_ylabel("error")
        ax.grid()

    plt.show()


class Plotter:
    """A warpper class for data visualizations"""

    def __init__(self, title="Plot", template="seaborn", **layout_kwargs):
        self.fig = go.Figure()
        self.fig.update_layout(title=title, template=template, **layout_kwargs)


    def _parse_points(self, pts):
        """Helper method to determine if points are 2D or 3D and extract axes."""
        pts = np.asarray(pts)
        if pts.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (N, 2) or (N, 3), got shape {pts.shape}")
            
        dims = pts.shape[1]
        if dims == 2:
            return go.Scatter, dict(x=pts[:, 0], y=pts[:, 1])
        elif dims == 3:
            return go.Scatter3d, dict(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])
        else:
            raise ValueError(f"Only 2D or 3D points are supported. Got {dims} dimensions.")
        
        
    def add_points(self, pts, name="Points", color=None, point_size=8, **kwargs):
        """Adds scatter points to the plot."""
        TraceClass, coords = self._parse_points(pts)
        
        marker_opts = {"size": point_size}
        if color:
            marker_opts["color"] = color
            
        trace = TraceClass(
            **coords,
            mode="markers",
            name=name,
            marker=marker_opts,
            **kwargs
        )
        self.fig.add_trace(trace)
        return self


    def add_trajectory(self, pts, name="Trajectory", color=None, line_width=3, dash="solid", **kwargs):
        """Adds a line trajectory to the plot."""
        TraceClass, coords = self._parse_points(pts)
        
        line_opts = {"width": line_width, "dash": dash}
        if color:
            line_opts["color"] = color
            
        trace = TraceClass(
            **coords,
            mode="lines",
            name=name,
            line=line_opts,
            **kwargs
        )
        self.fig.add_trace(trace)
        return self
        

    def clear(self):
        """Clear all traces."""
        self.fig.data = []
    

    def show(self):
        """Render!"""
        self.fig.show()


class AnimatedPlotter:
    """A wrapper class for animated data visualizations (companion to Plotter).

    Register time-indexed elements, then call show(). Static context (a faint
    full path) is drawn once; only the moving parts are rebuilt per frame, and
    axis ranges are fixed from the data so the view doesn't lurch while playing.

        ap = AnimatedPlotter(title="GT vs approx", scene={"aspectmode": "data"},
                             scene_camera={"up": {"x": 0, "y": 1, "z": 0}})
        ap.add_trajectory(traj, name="ground truth", color="blue")
        ap.add_trajectory(approx_traj, name="approx", color="red")
        ap.show()
    """

    def __init__(self, title="Animation", template="seaborn", fps=30, stride=1, **layout_kwargs):
        self.fig = go.Figure()
        self.fig.update_layout(title=title, template=template, **layout_kwargs)
        self.fps = fps
        self.stride = stride
        self._static = []      # traces drawn once
        self._dynamic = []     # (builder(frame) -> trace, n_frames)
        self._bounds = []      # (M, D) arrays for fixing axis ranges


    @staticmethod
    def _axes(dims):
        """Pick the trace class and axis keys for 2D or 3D data."""
        if dims == 2:
            return go.Scatter, ("x", "y")
        elif dims == 3:
            return go.Scatter3d, ("x", "y", "z")
        else:
            raise ValueError(f"Only 2D or 3D points are supported. Got {dims} dimensions.")


    @staticmethod
    def _coords(arr, keys):
        """Map an (N, D) array to a {x:..., y:..., (z:...)} dict."""
        return {k: arr[:, i] for i, k in enumerate(keys)}


    def add_trajectory(self, pts, name="Trajectory", color=None, line_width=3, dash="solid",
                       trail=True, trail_length=None, show_path=True, head_size=6, **kwargs):
        """Animate a point travelling along a path, leaving a trail behind it.

        trail_length: number of frames the trail spans behind the head. None
        (default) grows the trail from the start; an int shows a sliding window
        of that many recent frames."""
        pts = np.asarray(pts)
        if pts.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (N, 2) or (N, 3), got shape {pts.shape}")
        TraceClass, keys = self._axes(pts.shape[1])
        self._bounds.append(pts)

        line_opts = {"width": line_width, "dash": dash}
        if color:
            line_opts["color"] = color

        if show_path:
            path_line = {"width": max(line_width - 1, 1)}
            if color:
                path_line["color"] = color
            self._static.append(TraceClass(
                **self._coords(pts, keys), mode="lines", name=name,
                line=path_line, opacity=0.25, showlegend=True))

        if trail:
            def _trail(f):
                start = 0 if trail_length is None else max(0, f - trail_length + 1)
                seg = pts[start:max(f, 0) + 1]
                return TraceClass(**self._coords(seg, keys), mode="lines",
                                  name=name, line=line_opts, showlegend=False, **kwargs)
            self._dynamic.append((_trail, len(pts)))

        def _head(f):
            i = min(max(f, 0), len(pts) - 1)
            marker_opts = {"size": head_size}
            if color:
                marker_opts["color"] = color
            return TraceClass(**self._coords(pts[i:i + 1], keys), mode="markers",
                              name=name, marker=marker_opts, showlegend=False, **kwargs)
        self._dynamic.append((_head, len(pts)))
        return self


    def add_points(self, pts, name="Points", color=None, point_size=8, **kwargs):
        """Animate a set of points over time. Expects (F, K, D) or (F, D)."""
        pts = np.asarray(pts)
        if pts.ndim == 2:                  # (F, D) single moving point -> (F, 1, D)
            pts = pts[:, None, :]
        if pts.ndim != 3:
            raise ValueError(f"Expected (F, K, 2|3) or (F, 2|3), got shape {pts.shape}")
        F, _, dims = pts.shape
        TraceClass, keys = self._axes(dims)
        self._bounds.append(pts.reshape(-1, dims))

        def _builder(f):
            i = min(max(f, 0), F - 1)
            marker_opts = {"size": point_size}
            if color:
                marker_opts["color"] = color
            return TraceClass(**self._coords(pts[i], keys), mode="markers",
                              name=name, marker=marker_opts, **kwargs)
        self._dynamic.append((_builder, F))
        return self


    def add_skeleton(self, P, bones, names=None, idx=None, name="Skeleton",
                     joint_color=None, bone_color=None, joint_size=4, bone_width=3, **kwargs):
        """Animate a skeleton. P is (F, J, D); bones is a list of (parent, child)
        name pairs; pass names (len J) or an idx {name: row} mapping."""
        P = np.asarray(P)
        if P.ndim != 3:
            raise ValueError(f"Expected skeleton array of shape (F, J, 2|3), got {P.shape}")
        F, _, dims = P.shape
        TraceClass, keys = self._axes(dims)
        if idx is None:
            if names is None:
                raise ValueError("Provide names (len J) or an idx mapping.")
            idx = {n: i for i, n in enumerate(names)}
        valid = [(a, b) for a, b in bones if a in idx and b in idx]
        bone_color = bone_color or joint_color
        self._bounds.append(P.reshape(-1, dims))

        def _joints(f):
            i = min(max(f, 0), F - 1)
            marker_opts = {"size": joint_size}
            if joint_color:
                marker_opts["color"] = joint_color
            return TraceClass(**self._coords(P[i], keys), mode="markers", name=f"{name} (joint)", marker=marker_opts, **kwargs)

        def _bones(f):
            i = min(max(f, 0), F - 1)
            pose = P[i]
            segs = {k: [] for k in keys}
            for a, b in valid:
                pa, pb = pose[idx[a]], pose[idx[b]]
                for d, k in enumerate(keys):
                    segs[k] += [pa[d], pb[d], None]      # None breaks the line between bones
            line_opts = {"width": bone_width}
            if bone_color:
                line_opts["color"] = bone_color
            return TraceClass(**segs, mode="lines", name=f"{name} (bones)", line=line_opts)

        self._dynamic.append((_joints, F))
        self._dynamic.append((_bones, F))
        return self
    

    def add_orientations(self, positions, quats, scale=0.05, width=4,
                         colors=("red", "green", "blue"), name="frames", **kwargs):
        """Animate coordinate triads (local X/Y/Z axes) at each joint, showing
        orientation that position can't (e.g. a bone's axial twist).
 
        positions: (F, J, 3); quats: (F, J, 4) in [x, y, z, w] order. Each axis
        is one multi-segment line trace over all joints, RGB = XYZ by default."""
        from scipy.spatial.transform import Rotation as Rot  # xyzw == scalar-last
        positions = np.asarray(positions)
        quats = np.asarray(quats)
        F, J, _ = positions.shape
        R = Rot.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(F, J, 3, 3)
        self._bounds.append(positions.reshape(-1, 3))
 
        def make_axis(axis):
            def _builder(f):
                i = min(max(f, 0), F - 1)
                p, r = positions[i], R[i]
                xs, ys, zs = [], [], []
                for j in range(J):
                    tip = p[j] + scale * r[j][:, axis]    # column = rotated basis vector
                    xs += [p[j, 0], tip[0], None]
                    ys += [p[j, 1], tip[1], None]
                    zs += [p[j, 2], tip[2], None]
                return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                    line=dict(width=width, color=colors[axis]),
                                    name=f"{name} {'XYZ'[axis]}", showlegend=False, **kwargs)
            return _builder
 
        for axis in range(3):
            self._dynamic.append((make_axis(axis), F))
        return self


    def _build(self):
        """Assemble static traces, per-frame data, ranges, and playback controls."""
        if not self._dynamic:
            raise ValueError("Nothing to animate; add a trajectory, points, or skeleton first.")

        n_frames = max(n for _, n in self._dynamic)
        builders = [b for b, _ in self._dynamic]

        frame_ids = list(range(0, n_frames, self.stride))
        if frame_ids[-1] != n_frames - 1:
            frame_ids.append(n_frames - 1)

        dims, lo, hi = self._compute_bounds()

        # static context, then an invisible anchor that pins the bounds in every
        # frame (otherwise an all-dynamic scene, e.g. a skeleton, lets the view
        # rescale per frame), then frame-0 of each dynamic element
        self.fig.data = ()
        for tr in self._static:
            self.fig.add_trace(tr)
        self.fig.add_trace(self._anchor(dims, lo, hi))
        n_fixed = len(self._static) + 1
        for b in builders:
            self.fig.add_trace(b(0))

        dyn_idx = list(range(n_fixed, n_fixed + len(builders)))
        frame_layout = self._range_layout(dims, lo, hi)   # re-asserted every frame
        self.fig.frames = [
            go.Frame(data=[b(f) for b in builders], traces=dyn_idx,
                     name=str(f), layout=frame_layout)
            for f in frame_ids
        ]

        self._apply_ranges(dims, lo, hi)
        self._apply_controls(frame_ids)
        return self.fig


    def _compute_bounds(self):
        """Global padded (lo, hi) corners across all registered data."""
        allpts = np.concatenate(self._bounds, axis=0)
        dims = allpts.shape[1]
        lo, hi = allpts.min(0), allpts.max(0)
        pad = (hi - lo).max() * 0.05 or 1.0
        return dims, lo - pad, hi + pad


    def _anchor(self, dims, lo, hi):
        """Invisible 2-point trace at the bounding-box corners; keeps the scene
        extent constant even though every frame's data changes."""
        TraceClass, keys = self._axes(dims)
        corners = np.vstack([lo, hi])
        return TraceClass(**self._coords(corners, keys), mode="markers",
                          marker=dict(size=0.1, opacity=0), hoverinfo="skip", showlegend=False)


    def _range_layout(self, dims, lo, hi):
        """Fixed-range layout (no camera, so orbiting is preserved)."""
        if dims == 3:
            return go.Layout(scene=dict(
                xaxis=dict(range=[lo[0], hi[0]], autorange=False),
                yaxis=dict(range=[lo[1], hi[1]], autorange=False),
                zaxis=dict(range=[lo[2], hi[2]], autorange=False)))
        return go.Layout(
            xaxis=dict(range=[lo[0], hi[0]], autorange=False),
            yaxis=dict(range=[lo[1], hi[1]], autorange=False))


    def _apply_ranges(self, dims, lo, hi):
        """Fix axis ranges on the base layout (merges into existing scene)."""
        self.fig.update_layout(self._range_layout(dims, lo, hi))


    def _apply_controls(self, frame_ids):
        """Add Play/Pause buttons and a frame scrubber."""
        dur = int(1000 / self.fps) * self.stride
        play = dict(label="Play", method="animate",
                    args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True)])
        pause = dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
        steps = [dict(method="animate", label=str(f),
                      args=[[str(f)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                 for f in frame_ids]
        self.fig.update_layout(
            updatemenus=[dict(type="buttons", x=0.05, y=0.05, xanchor="left", yanchor="bottom",
                              buttons=[play, pause])],
            sliders=[dict(active=0, x=0.05, len=0.9, currentvalue=dict(prefix="frame "), steps=steps)],
        )


    def clear(self):
        """Clear all traces, frames, and registered elements."""
        self.fig.data = ()
        self.fig.frames = ()
        self._static = []
        self._dynamic = []
        self._bounds = []


    def show(self):
        """Render!"""
        self._build()
        self.fig.show()


    def save_html(self, path):
        """Write a standalone HTML file (keeps the animation and controls)."""
        self._build()
        self.fig.write_html(path)
        return path