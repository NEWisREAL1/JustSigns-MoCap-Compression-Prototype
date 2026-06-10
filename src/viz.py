import numpy as np
import plotly.graph_objects as go


class RotationPlotter:
    """A wrapper class for visualizing a rotation as an oriented triad.

    The reference (identity) frame is drawn as thin neutral guide lines with
    colored X/Y/Z labels; each rotation added with add_rotation() is drawn as
    bold colored arrows on top, so the two are never confused. A wireframe
    globe (axis tips ride on it) and an origin plane with a grid give the eye
    something to track tilt and twist against, and the rotation's own axis is
    drawn through the sphere with its angle.

        rp = RotationPlotter(up="y")
        rp.add_rotation([0, 0, 0, 1])          # quaternion [x, y, z, w]
        rp.show()
    """

    _colors = ("#d62728", "#2ca02c", "#1f77b4")     # X, Y, Z (softer than pure RGB)
    _axis_names = ("X", "Y", "Z")

    def __init__(self, title="Rotation", template="plotly_white", axis_len=1.0, up="y",
                 show_sphere=True, show_plane=True, show_reference=True, **layout_kwargs):
        self.fig = go.Figure()
        self.axis_len = axis_len
        self.up = up
        self.show_sphere = show_sphere
        self.show_plane = show_plane
        self.show_reference = show_reference

        rng = 1.45 * axis_len
        axis_style = dict(showbackground=False, showgrid=True, gridcolor="#f0f0f0",
                          zeroline=False, range=[-rng, rng])
        self.fig.update_layout(
            title=title, template=template, showlegend=True,
            scene=dict(
                xaxis=dict(title="X", **axis_style),
                yaxis=dict(title="Y", **axis_style),
                zaxis=dict(title="Z", **axis_style),
                aspectmode="cube",
                camera=dict(up=self._up_vector(), eye=self._default_eye())),
            margin=dict(l=0, r=0, b=0, t=40))
        self.fig.update_layout(**layout_kwargs)         # user overrides win
        self._draw_scaffold()


    # -- frame helpers ----------------------------------------------------
    def _up_vector(self):
        return {"y": dict(x=0, y=1, z=0),
                "z": dict(x=0, y=0, z=1),
                "x": dict(x=1, y=0, z=0)}[self.up]

    def _default_eye(self):
        return {"y": dict(x=1.5, y=1.0, z=1.6),
                "z": dict(x=1.6, y=1.5, z=1.1),
                "x": dict(x=1.1, y=1.6, z=1.5)}[self.up]

    def _to_plane(self, a, b):
        """Map in-plane (a, b) to 3D so the plane passes through the origin
        perpendicular to the up axis (xz for y-up, xy for z-up)."""
        zeros = np.zeros_like(a)
        if self.up == "y":
            return a, zeros, b
        if self.up == "z":
            return a, b, zeros
        return zeros, a, b

    def _on_sphere(self, t, lon):
        """Point(s) on the sphere with the pole aligned to the up axis, so the
        equator lies in the origin plane."""
        up_c = np.cos(t)
        a = np.sin(t) * np.cos(lon)
        b = np.sin(t) * np.sin(lon)
        if self.up == "y":
            return a, up_c, b
        if self.up == "z":
            return a, b, up_c
        return up_c, a, b


    @staticmethod
    def _to_matrix(rotation):
        """Accept a quaternion [x, y, z, w], a 3x3 matrix, or any object with
        .as_matrix() (e.g. scipy Rotation). Columns are the rotated axes."""
        if hasattr(rotation, "as_matrix"):
            return np.asarray(rotation.as_matrix(), dtype=float)
        arr = np.asarray(rotation, dtype=float)
        if arr.shape == (3, 3):
            return arr
        if arr.shape == (4,):
            x, y, z, w = arr / np.linalg.norm(arr)
            return np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
        raise ValueError("rotation must be a quaternion [x,y,z,w], a 3x3 matrix, "
                         "or have an .as_matrix() method")

    @staticmethod
    def _axis_angle(R):
        """Rotation axis (unit) and angle (rad) from a rotation matrix."""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        n = np.linalg.norm(ax)
        if n < 1e-8:                                    # angle ~0 or ~pi
            vals, vecs = np.linalg.eig(R)
            ax = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
            n = np.linalg.norm(ax)
        return (ax / n if n else np.array([0, 0, 1.0])), angle


    # -- drawing ----------------------------------------------------------
    def _triad(self, R, opacity=1.0, colors=None, prefix="", width=8, head=0.22):
        colors = colors or self._colors
        for i in range(3):
            vec = R[:, i] * self.axis_len
            self.fig.add_trace(go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines",
                line=dict(color=colors[i], width=width), opacity=opacity, showlegend=False))
            self.fig.add_trace(go.Cone(
                x=[vec[0]], y=[vec[1]], z=[vec[2]], u=[vec[0]], v=[vec[1]], w=[vec[2]],
                colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False,
                sizemode="absolute", sizeref=head * self.axis_len, anchor="tip",
                opacity=opacity, showlegend=True, name=f"{prefix}{self._axis_names[i]}"))

    def _reference(self):
        L = self.axis_len
        for i in range(3):
            vec = np.eye(3)[:, i] * L
            self.fig.add_trace(go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines",
                line=dict(color="#888", width=2), opacity=0.55,
                hoverinfo="skip", showlegend=False))
            self.fig.add_trace(go.Scatter3d(
                x=[vec[0] * 1.13], y=[vec[1] * 1.13], z=[vec[2] * 1.13], mode="text",
                text=[self._axis_names[i]], textfont=dict(color=self._colors[i], size=14),
                hoverinfo="skip", showlegend=False))

    def _sphere(self):
        r = self.axis_len
        t = np.linspace(0, np.pi, 30)
        lon = np.linspace(0, 2 * np.pi, 60)
        T, Lo = np.meshgrid(t, lon)
        sx, sy, sz = self._on_sphere(T, Lo)
        self.fig.add_trace(go.Surface(
            x=r * sx, y=r * sy, z=r * sz, opacity=0.06,
            colorscale=[[0, "#4a90d9"], [1, "#4a90d9"]], showscale=False, hoverinfo="skip"))
        xs, ys, zs = [], [], []
        for L0 in np.linspace(0, 2 * np.pi, 13)[:-1]:           # meridians
            tt = np.linspace(0, np.pi, 30)
            mx, my, mz = self._on_sphere(tt, np.full_like(tt, L0))
            xs += list(r * mx) + [None]; ys += list(r * my) + [None]; zs += list(r * mz) + [None]
        for t0 in np.linspace(0, np.pi, 7)[1:-1]:               # parallels
            pp = np.linspace(0, 2 * np.pi, 48)
            px, py, pz = self._on_sphere(np.full_like(pp, t0), pp)
            xs += list(r * px) + [None]; ys += list(r * py) + [None]; zs += list(r * pz) + [None]
        self.fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#9ec5e8", width=1), opacity=0.55, hoverinfo="skip", showlegend=False))

    def _plane(self):
        ext = 1.3 * self.axis_len
        A, B = np.meshgrid([-ext, ext], [-ext, ext])
        x, y, z = self._to_plane(A, B)
        self.fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.07,
            colorscale=[[0, "#888"], [1, "#888"]], showscale=False, hoverinfo="skip"))
        xs, ys, zs = [], [], []
        for c in np.linspace(-ext, ext, 9):
            cc, line = np.array([c, c]), np.array([-ext, ext])
            for a, b in ((cc, line), (line, cc)):
                gx, gy, gz = self._to_plane(a, b)
                xs += [gx[0], gx[1], None]; ys += [gy[0], gy[1], None]; zs += [gz[0], gz[1], None]
        self.fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#999", width=1), opacity=0.6, hoverinfo="skip", showlegend=False))

    def _draw_scaffold(self):
        if self.show_plane:
            self._plane()
        if self.show_sphere:
            self._sphere()
        # origin marker
        self.fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
            marker=dict(size=3, color="#444"), hoverinfo="skip", showlegend=False))
        if self.show_reference:
            self._reference()


    # -- public API -------------------------------------------------------
    def add_rotation(self, rotation, name="rotated", colors=None, opacity=1.0,
                     show_axis=True, axis_color="#ff8c00"):
        """Draw a rotation (quaternion [x,y,z,w], 3x3 matrix, or scipy Rotation)
        as a solid triad. With show_axis, also draw its axis-angle line."""
        R = self._to_matrix(rotation)
        self._triad(R, opacity=opacity, colors=colors, prefix=f"{name} ")
        if show_axis:
            ax, angle = self._axis_angle(R)
            p = ax * self.axis_len * 1.25
            self.fig.add_trace(go.Scatter3d(
                x=[-p[0], p[0]], y=[-p[1], p[1]], z=[-p[2], p[2]], mode="lines+markers",
                line=dict(color=axis_color, width=4), marker=dict(size=3, color=axis_color),
                opacity=0.9, name=f"{name} axis ({np.degrees(angle):.0f}°)"))
        return self

    def clear(self):
        """Clear added rotations and redraw the reference scaffold."""
        self.fig.data = []
        self._draw_scaffold()

    def show(self):
        """Render!"""
        self.fig.show()


class RotationPathPlotter(RotationPlotter):
    """Trace the path swept on the unit sphere by rotated reference axes over a
    sequence of rotations (static path; animation handled by a later class).

    Inherits the globe / origin-plane / reference scaffold from RotationPlotter.

        rpp = RotationPathPlotter(title="RightHand path", up="y")
        rpp.add_rotation_path(quats["RightHand"])     # (T, 4) [x,y,z,w]
        rpp.show()
    """

    _AXIS_IDX = {"x": 0, "y": 1, "z": 2}

    def _stack_matrices(self, rotations):
        """Normalize input to (T, 3, 3). Accepts (T,4) quats, (T,3,3) matrices,
        a single scipy Rotation holding the sequence, or an iterable of any
        single-rotation form."""
        if hasattr(rotations, "as_matrix"):                 # scipy Rotation(seq)
            m = np.asarray(rotations.as_matrix(), dtype=float)
            return m if m.ndim == 3 else m[None]
        if isinstance(rotations, np.ndarray):
            if rotations.ndim == 3:
                return rotations
            if rotations.ndim == 2 and rotations.shape[1] == 4:
                return np.stack([self._to_matrix(q) for q in rotations])
        return np.stack([self._to_matrix(r) for r in rotations])

    def _axis_path(self, M, axis_idx, color, name, color_by_time, width, show_endpoints):
        V = M[:, :, axis_idx] * self.axis_len               # (T, 3), rides on the globe
        T = len(V)
        if color_by_time:
            line = dict(color=np.arange(T), colorscale="Viridis", width=width,
                        colorbar=dict(title="frame", thickness=12))
        else:
            line = dict(color=color, width=width)
        self.fig.add_trace(go.Scatter3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2], mode="lines", line=line,
            name=name, showlegend=not color_by_time,
            text=[f"frame {i}" for i in range(T)], hoverinfo="text"))
        if show_endpoints:
            self.fig.add_trace(go.Scatter3d(
                x=[V[0, 0]], y=[V[0, 1]], z=[V[0, 2]], mode="markers",
                marker=dict(size=6, color=color, symbol="circle-open", line=dict(width=2)),
                hoverinfo="text", text=["start"], showlegend=False))
            self.fig.add_trace(go.Scatter3d(
                x=[V[-1, 0]], y=[V[-1, 1]], z=[V[-1, 2]], mode="markers",
                marker=dict(size=5, color=color, symbol="diamond"),
                hoverinfo="text", text=["end"], showlegend=False))

    def add_rotation_path(self, rotations, axes=("x", "y", "z"), name="path",
                          colors=None, color_by_time=False, width=5,
                          show_endpoints=True, show_end_triad=False):
        """Draw the sphere-surface path(s) traced by the chosen reference axes.

        rotations    : (T,4) quaternions [x,y,z,w], (T,3,3) matrices, a scipy
                       Rotation sequence, or an iterable of single rotations.
        axes         : any of 'x','y','z' to trace (default all three; one axis
                       alone can't show twist about itself, hence three).
        color_by_time: gradient the path by frame instead of axis color (best
                       with a single axis); adds a frame colorbar.
        show_end_triad: also draw the final orientation as a faint triad.
        """
        M = self._stack_matrices(rotations)
        colors = colors or self._colors
        for ax in axes:
            i = self._AXIS_IDX[ax]
            self._axis_path(M, i, colors[i], f"{name} {ax.upper()}",
                            color_by_time, width, show_endpoints)
        if show_end_triad:
            self._triad(M[-1], opacity=0.5, prefix=f"{name} end ")
        return self