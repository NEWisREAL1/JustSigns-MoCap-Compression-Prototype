import numpy as np
import plotly.graph_objects as go

# from src.model.skeleton import KinematicsSkeleton


class AnimatedSkeletonsPlotter:
    def __init__(self, title="MoCap Animation", width=900, height=500, aspectmode="data"):
        self.fig = go.Figure()
        self.title = title
        self.width = width
        self.height = height
        self.aspectmode = aspectmode
        
        # Store state for multiple skeletons to sync frames later
        self.skeletons_data = []
        self.max_frames = 0
        
        # Expandable global bounding box
        self.bounds = {
            'x': [float('inf'), float('-inf')],
            'y': [float('inf'), float('-inf')],
            'z': [float('inf'), float('-inf')]
        }
        
        # Color palette to distinguish different skeletons
        self.colors = [
            'rgb(0, 150, 255)',   # Blue
            'rgb(255, 50, 50)',   # Red
            'rgb(50, 200, 50)',   # Green
            'rgb(255, 150, 0)',   # Orange
            'rgb(200, 0, 255)'    # Purple
        ]
        
    def _build_lines(self, positions, parents):
        x_lines, y_lines, z_lines = [], [], []
        for j in range(1, len(parents)):
            p = parents[j]
            x_lines.extend([positions[p, 0], positions[j, 0], None])
            y_lines.extend([positions[p, 1], positions[j, 1], None])
            z_lines.extend([positions[p, 2], positions[j, 2], None])
        return x_lines, y_lines, z_lines

    def add_skeleton(self, anim_rotations, kinematics_skeleton, name=None, offset=None, stride=1):
        global_positions, _ = kinematics_skeleton.forward_kinematics(anim_rotations)

        if offset is not None:
            global_positions += np.asarray(offset)
        
        global_positions = global_positions[::stride]
        num_frames = global_positions.shape[0]
        
        # Update max frames tracking
        self.max_frames = max(self.max_frames, num_frames)
        
        # Cycle through colors for distinct skeletons
        color_idx = len(self.skeletons_data) % len(self.colors)
        base_color = self.colors[color_idx]
        
        # Update overall bounding box incrementally
        self.bounds['x'][0] = min(self.bounds['x'][0], np.min(global_positions[:, :, 0]) - 0.25)
        self.bounds['x'][1] = max(self.bounds['x'][1], np.max(global_positions[:, :, 0]) + 0.25)
        
        self.bounds['y'][0] = min(self.bounds['y'][0], np.min(global_positions[:, :, 2]) - 0.25)
        self.bounds['y'][1] = max(self.bounds['y'][1], np.max(global_positions[:, :, 2]) + 0.25)
        
        self.bounds['z'][0] = min(self.bounds['z'][0], np.min(global_positions[:, :, 1]) - 0.25)
        self.bounds['z'][1] = max(self.bounds['z'][1], np.max(global_positions[:, :, 1]) + 0.25)

        # Store for synchronous frame rendering later
        self.skeletons_data.append({
            'positions': global_positions,
            'parents': kinematics_skeleton.parents
        })
        
        # Plot initial (frame 0) traces
        pos_0 = global_positions[0]
        x_lines, y_lines, z_lines = self._build_lines(pos_0, kinematics_skeleton.parents)
        
        # Base Bones
        self.fig.add_trace(go.Scatter3d(
            x=x_lines, y=z_lines, z=y_lines,
            mode='lines',
            line=dict(color=base_color, width=10),
            name=f"{name} (Bones)" if name is not None else "Bones",
            hoverinfo="skip",
        ))
        
        # Base Joints
        self.fig.add_trace(go.Scatter3d(
            x=pos_0[:, 0], y=pos_0[:, 2], z=pos_0[:, 1],
            mode='markers',
            marker=dict(size=4, color="black"),
            # name=f"{name} (Joints)" if name is not None else "Joints",
            showlegend=False,
            text=kinematics_skeleton.joint_names,
        ))

    def _build_all_frames(self):
        """Assembles frames to animate all traces simultaneously."""
        frames = []
        for f in range(self.max_frames):
            frame_data = []
            
            # Update each skeleton in the exact order they were added
            for skel in self.skeletons_data:
                # If clips have different lengths, freeze the shorter one at its final frame
                safe_f = min(f, skel['positions'].shape[0] - 1)
                pos_f = skel['positions'][safe_f]
                
                xl, yl, zl = self._build_lines(pos_f, skel['parents'])
                
                frame_data.append(go.Scatter3d(x=xl, y=zl, z=yl))
                frame_data.append(go.Scatter3d(x=pos_f[:, 0], y=pos_f[:, 2], z=pos_f[:, 1]))
                
            frames.append(go.Frame(data=frame_data, name=f"frame_{f}"))
        return frames

    def show(self):
        # Build all synchronized frames
        self.fig.frames = self._build_all_frames()

        sliders = [{
            "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                       "label": str(k), "method": "animate"} for k, f in enumerate(self.fig.frames)],
            "x": 0.1, "y": 0, "len": 0.9, "xanchor": "left", "yanchor": "top"
        }]

        self.fig.update_layout(
            template="simple_white",
            title=self.title,
            width=self.width,
            height=self.height,
            scene=dict(
                xaxis=dict(range=self.bounds['x'], autorange=False),
                yaxis=dict(range=self.bounds['y'], autorange=False),
                zaxis=dict(range=self.bounds['z'], autorange=False),
                aspectmode=self.aspectmode,
            ),
            scene_camera=dict(
                eye=dict(x=0, y=1.25, z=0.5),
                center=dict(x=0, y=0, z=0.5),
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 1000/30, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ],
                x=0, y=0, xanchor="right", yanchor="top"
            )],
            sliders=sliders,
        )
        self.fig.show()