#!/usr/bin/env python3
"""
Enhanced Trajectory to Video Conversion with Clutter and Multiple Objects
=========================================================================
Extends trajectory_to_video.py to support:
- Transient objects (appear and disappear)
- Persistent objects (full trajectory)
- Background clutter/noise
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import base function
from basic.trajectory_to_video import trajectory_to_video, render_trajectory_frame


class ClutterObject:
    """Represents a clutter object (transient or persistent)."""
    
    def __init__(
        self,
        position: np.ndarray,
        shape: str = 'circle',  # 'circle', 'square', 'triangle', 'star'
        size: float = 0.02,
        color: Tuple[int, int, int] = (128, 128, 128),
        appearance_frame: int = 0,
        disappearance_frame: Optional[int] = None,  # None = persistent
        trajectory: Optional[np.ndarray] = None  # For moving clutter
    ):
        """
        Initialize clutter object.
        
        Args:
            position: (3,) or (2,) initial position
            shape: Shape type
            size: Size of object (normalized 0-1)
            color: RGB color tuple (0-255)
            appearance_frame: Frame when object appears
            disappearance_frame: Frame when object disappears (None = persistent)
            trajectory: Optional trajectory for moving clutter (N, 3) or (N, 2)
        """
        self.position = np.array(position)
        self.shape = shape
        self.size = size
        self.color = np.array(color, dtype=np.uint8) / 255.0  # Normalize to 0-1
        self.appearance_frame = appearance_frame
        self.disappearance_frame = disappearance_frame
        self.trajectory = trajectory
        self.is_persistent = (disappearance_frame is None)
    
    def get_position_at_frame(self, frame: int) -> np.ndarray:
        """Get object position at given frame."""
        if self.trajectory is not None:
            # Moving clutter - interpolate trajectory
            if frame < len(self.trajectory):
                return self.trajectory[frame]
            else:
                return self.trajectory[-1]
        else:
            # Static clutter
            return self.position
    
    def is_visible_at_frame(self, frame: int) -> bool:
        """Check if object is visible at given frame."""
        if frame < self.appearance_frame:
            return False
        if self.disappearance_frame is not None and frame >= self.disappearance_frame:
            return False
        return True


def trajectory_to_video_with_clutter(
    trajectory: np.ndarray,
    resolution: Tuple[int, int] = (128, 128),
    num_frames: Optional[int] = None,
    camera_view: str = '3d_projection',
    fps: int = 10,
    line_width: float = 2.0,
    point_size: float = 50.0,
    # Clutter parameters
    transient_objects: Optional[List[ClutterObject]] = None,
    persistent_objects: Optional[List[ClutterObject]] = None,
    background_clutter: bool = False,
    clutter_density: float = 0.1,  # 0-1, fraction of frame area
    noise_level: float = 0.0  # 0-1, amount of random noise
) -> np.ndarray:
    """
    Convert 3D trajectory to video with clutter, transient, and persistent objects.
    
    Args:
        trajectory: (N, 3) array of (x, y, z) points
        resolution: (height, width) of output frames
        num_frames: Number of frames (default: N, or min(N, 100))
        camera_view: '3d_projection', 'xy', 'xz', 'yz', or 'all_views'
        fps: Frames per second
        line_width: Width of trajectory line
        point_size: Size of current point marker
        transient_objects: List of ClutterObject that appear/disappear
        persistent_objects: List of ClutterObject that persist through full video
        background_clutter: Whether to add random background clutter
        clutter_density: Density of background clutter (0-1)
        noise_level: Amount of random noise to add (0-1)
    
    Returns:
        video: (T, H, W, 3) array of RGB frames (uint8, 0-255)
    """
    if trajectory.shape[1] != 3:
        raise ValueError(f"Trajectory must be (N, 3), got shape {trajectory.shape}")
    
    N = len(trajectory)
    
    # Determine number of frames
    if num_frames is None:
        num_frames = min(N, 100)
    
    # Interpolate trajectory to desired frame count
    if N != num_frames:
        t_original = np.linspace(0, 1, N)
        t_new = np.linspace(0, 1, num_frames)
        trajectory_interp = np.array([
            np.interp(t_new, t_original, trajectory[:, i])
            for i in range(3)
        ]).T
    else:
        trajectory_interp = trajectory.copy()
    
    # Normalize trajectory to fit in frame
    traj_min = trajectory_interp.min(axis=0)
    traj_max = trajectory_interp.max(axis=0)
    traj_range = traj_max - traj_min
    traj_range[traj_range == 0] = 1
    
    trajectory_normalized = (trajectory_interp - traj_min) / traj_range
    trajectory_normalized = trajectory_normalized * 0.8 + 0.1
    
    # Generate background clutter if requested
    if background_clutter:
        clutter_objects = _generate_background_clutter(
            num_frames, clutter_density, traj_min, traj_max, traj_range
        )
        if transient_objects is None:
            transient_objects = []
        transient_objects.extend(clutter_objects)
    
    # Normalize clutter object positions
    if transient_objects:
        for obj in transient_objects:
            if obj.trajectory is not None:
                obj.trajectory = (obj.trajectory - traj_min) / traj_range * 0.8 + 0.1
            else:
                obj.position = (obj.position - traj_min) / traj_range * 0.8 + 0.1
    
    if persistent_objects:
        for obj in persistent_objects:
            if obj.trajectory is not None:
                obj.trajectory = (obj.trajectory - traj_min) / traj_range * 0.8 + 0.1
            else:
                obj.position = (obj.position - traj_min) / traj_range * 0.8 + 0.1
    
    # Create frames
    frames = []
    for i in range(num_frames):
        frame = render_trajectory_frame_with_clutter(
            trajectory_normalized[:i+1],
            resolution,
            camera_view,
            line_width,
            point_size,
            transient_objects if transient_objects else [],
            persistent_objects if persistent_objects else [],
            frame_idx=i,
            noise_level=noise_level
        )
        frames.append(frame)
    
    return np.array(frames, dtype=np.uint8)


def _generate_background_clutter(
    num_frames: int,
    density: float,
    traj_min: np.ndarray,
    traj_max: np.ndarray,
    traj_range: np.ndarray
) -> List[ClutterObject]:
    """
    Generate random background clutter objects.
    
    Args:
        num_frames: Total number of frames
        density: Clutter density (0-1)
        traj_min: Minimum trajectory bounds
        traj_max: Maximum trajectory bounds
        traj_range: Trajectory range
    
    Returns:
        List of ClutterObject instances
    """
    clutter_objects = []
    num_clutter = int(density * 20)  # Scale density to object count
    
    shapes = ['circle', 'square', 'triangle']
    colors = [
        (100, 100, 100),  # Gray
        (150, 150, 150),  # Light gray
        (80, 80, 80),     # Dark gray
        (120, 120, 120),  # Medium gray
    ]
    
    for _ in range(num_clutter):
        # Random position within trajectory bounds (with some margin)
        margin = 0.1
        position = np.array([
            np.random.uniform(traj_min[0] - margin, traj_max[0] + margin),
            np.random.uniform(traj_min[1] - margin, traj_max[1] + margin),
            np.random.uniform(traj_min[2] - margin, traj_max[2] + margin)
        ])
        
        # Random appearance/disappearance
        appearance = np.random.randint(0, num_frames // 2)
        disappearance = np.random.randint(num_frames // 2, num_frames)
        
        # Some objects are persistent
        if np.random.random() < 0.3:
            disappearance = None
        
        obj = ClutterObject(
            position=position,
            shape=np.random.choice(shapes),
            size=np.random.uniform(0.01, 0.03),
            color=colors[np.random.randint(len(colors))],
            appearance_frame=appearance,
            disappearance_frame=disappearance
        )
        clutter_objects.append(obj)
    
    return clutter_objects


def render_trajectory_frame_with_clutter(
    trajectory_segment: np.ndarray,
    resolution: Tuple[int, int],
    camera_view: str,
    line_width: float = 2.0,
    point_size: float = 50.0,
    transient_objects: List[ClutterObject] = None,
    persistent_objects: List[ClutterObject] = None,
    frame_idx: int = 0,
    noise_level: float = 0.0
) -> np.ndarray:
    """
    Render a single frame with trajectory, transient, and persistent objects.
    
    Args:
        trajectory_segment: (M, 3) trajectory points up to current frame
        resolution: (height, width) of output frame
        camera_view: '3d_projection', 'xy', 'xz', or 'yz'
        line_width: Width of trajectory line
        point_size: Size of current point marker
        transient_objects: List of transient clutter objects
        persistent_objects: List of persistent clutter objects
        frame_idx: Current frame index
        noise_level: Amount of random noise (0-1)
    
    Returns:
        frame: (H, W, 3) RGB array (uint8)
    """
    height, width = resolution
    
    # Create figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('black')
    
    if camera_view == '3d_projection':
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        # Draw trajectory
        if len(trajectory_segment) > 1:
            ax.plot(
                trajectory_segment[:, 0],
                trajectory_segment[:, 1],
                trajectory_segment[:, 2],
                'b-',
                linewidth=line_width,
                alpha=0.8
            )
        
        # Current point
        if len(trajectory_segment) > 0:
            ax.scatter(
                trajectory_segment[-1, 0],
                trajectory_segment[-1, 1],
                trajectory_segment[-1, 2],
                c='red',
                s=point_size,
                alpha=1.0
            )
        
        # Draw persistent objects
        if persistent_objects:
            for obj in persistent_objects:
                pos = obj.get_position_at_frame(frame_idx)
                _draw_clutter_object_3d(ax, pos, obj.shape, obj.size, obj.color)
        
        # Draw transient objects (if visible)
        if transient_objects:
            for obj in transient_objects:
                if obj.is_visible_at_frame(frame_idx):
                    pos = obj.get_position_at_frame(frame_idx)
                    _draw_clutter_object_3d(ax, pos, obj.shape, obj.size, obj.color)
        
        # Set equal aspect ratio
        max_range = np.array([
            trajectory_segment[:, 0].max() - trajectory_segment[:, 0].min(),
            trajectory_segment[:, 1].max() - trajectory_segment[:, 1].min(),
            trajectory_segment[:, 2].max() - trajectory_segment[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (trajectory_segment[:, 0].max() + trajectory_segment[:, 0].min()) * 0.5
        mid_y = (trajectory_segment[:, 1].max() + trajectory_segment[:, 1].min()) * 0.5
        mid_z = (trajectory_segment[:, 2].max() + trajectory_segment[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
    
    else:
        # 2D projection
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        # Draw trajectory
        if len(trajectory_segment) > 1:
            if camera_view == 'xy':
                ax.plot(
                    trajectory_segment[:, 0],
                    trajectory_segment[:, 1],
                    'b-',
                    linewidth=line_width,
                    alpha=0.8
                )
                if len(trajectory_segment) > 0:
                    ax.scatter(
                        trajectory_segment[-1, 0],
                        trajectory_segment[-1, 1],
                        c='red',
                        s=point_size,
                        alpha=1.0
                    )
            elif camera_view == 'xz':
                ax.plot(
                    trajectory_segment[:, 0],
                    trajectory_segment[:, 2],
                    'b-',
                    linewidth=line_width,
                    alpha=0.8
                )
                if len(trajectory_segment) > 0:
                    ax.scatter(
                        trajectory_segment[-1, 0],
                        trajectory_segment[-1, 2],
                        c='red',
                        s=point_size,
                        alpha=1.0
                    )
            elif camera_view == 'yz':
                ax.plot(
                    trajectory_segment[:, 1],
                    trajectory_segment[:, 2],
                    'b-',
                    linewidth=line_width,
                    alpha=0.8
                )
                if len(trajectory_segment) > 0:
                    ax.scatter(
                        trajectory_segment[-1, 1],
                        trajectory_segment[-1, 2],
                        c='red',
                        s=point_size,
                        alpha=1.0
                    )
        
        # Draw persistent objects
        if persistent_objects:
            for obj in persistent_objects:
                pos = obj.get_position_at_frame(frame_idx)
                _draw_clutter_object_2d(ax, pos, obj.shape, obj.size, obj.color, camera_view)
        
        # Draw transient objects
        if transient_objects:
            for obj in transient_objects:
                if obj.is_visible_at_frame(frame_idx):
                    pos = obj.get_position_at_frame(frame_idx)
                    _draw_clutter_object_2d(ax, pos, obj.shape, obj.size, obj.color, camera_view)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    
    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(
        fig.canvas.tostring_rgb() if hasattr(fig.canvas, 'tostring_rgb') 
        else fig.canvas.buffer_rgba(), 
        dtype=np.uint8
    )
    if len(buf) == height * width * 4:  # RGBA
        buf = buf.reshape(height, width, 4)
        frame = buf[:, :, :3]  # Take RGB, drop Alpha
    else:  # RGB
        frame = buf.reshape(height, width, 3)
    plt.close(fig)
    
    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame


def _draw_clutter_object_3d(
    ax,
    position: np.ndarray,
    shape: str,
    size: float,
    color: np.ndarray
):
    """Draw a clutter object in 3D plot."""
    if shape == 'circle':
        # Draw sphere (approximate with scatter)
        ax.scatter(
            position[0], position[1], position[2],
            c=[color],
            s=size * 1000,
            alpha=0.6,
            edgecolors='none'
        )
    elif shape == 'square':
        # Draw cube (approximate with scatter)
        ax.scatter(
            position[0], position[1], position[2],
            c=[color],
            s=size * 1000,
            alpha=0.6,
            marker='s',
            edgecolors='none'
        )
    elif shape == 'triangle':
        # Draw triangle (approximate with scatter)
        ax.scatter(
            position[0], position[1], position[2],
            c=[color],
            s=size * 1000,
            alpha=0.6,
            marker='^',
            edgecolors='none'
        )


def _draw_clutter_object_2d(
    ax,
    position: np.ndarray,
    shape: str,
    size: float,
    color: np.ndarray,
    camera_view: str
):
    """Draw a clutter object in 2D plot."""
    # Extract 2D coordinates based on camera view
    if camera_view == 'xy':
        x, y = position[0], position[1]
    elif camera_view == 'xz':
        x, y = position[0], position[2]
    elif camera_view == 'yz':
        x, y = position[1], position[2]
    else:
        x, y = position[0], position[1]
    
    if shape == 'circle':
        circle = plt.Circle((x, y), size, color=color, alpha=0.6)
        ax.add_patch(circle)
    elif shape == 'square':
        square = plt.Rectangle(
            (x - size, y - size), 2*size, 2*size,
            color=color, alpha=0.6
        )
        ax.add_patch(square)
    elif shape == 'triangle':
        triangle = plt.Polygon(
            [(x, y + size), (x - size, y - size), (x + size, y - size)],
            color=color, alpha=0.6
        )
        ax.add_patch(triangle)


# Example usage and helper functions
def create_simple_transient_object(
    position: np.ndarray,
    appearance_frame: int,
    disappearance_frame: int,
    shape: str = 'circle',
    size: float = 0.02
) -> ClutterObject:
    """Helper to create a simple transient object."""
    return ClutterObject(
        position=position,
        shape=shape,
        size=size,
        color=(150, 150, 150),
        appearance_frame=appearance_frame,
        disappearance_frame=disappearance_frame
    )


def create_simple_persistent_object(
    position: np.ndarray,
    shape: str = 'square',
    size: float = 0.02
) -> ClutterObject:
    """Helper to create a simple persistent object."""
    return ClutterObject(
        position=position,
        shape=shape,
        size=size,
        color=(100, 100, 100),
        appearance_frame=0,
        disappearance_frame=None  # Persistent
    )


if __name__ == '__main__':
    # Example: Create trajectory with clutter
    print("Testing enhanced trajectory to video with clutter...")
    
    # Create sample trajectory
    t = np.linspace(0, 4 * np.pi, 100)
    trajectory = np.array([
        np.cos(t) * t,
        np.sin(t) * t,
        t * 0.5
    ]).T
    
    # Create some transient objects
    transient_objects = [
        create_simple_transient_object(
            position=np.array([2.0, 1.0, 1.0]),
            appearance_frame=10,
            disappearance_frame=30,
            shape='circle',
            size=0.03
        ),
        create_simple_transient_object(
            position=np.array([-1.0, 2.0, 2.0]),
            appearance_frame=20,
            disappearance_frame=40,
            shape='square',
            size=0.025
        )
    ]
    
    # Create some persistent objects
    persistent_objects = [
        create_simple_persistent_object(
            position=np.array([0.5, 0.5, 1.5]),
            shape='triangle',
            size=0.02
        ),
        create_simple_persistent_object(
            position=np.array([-0.5, 1.0, 2.0]),
            shape='square',
            size=0.02
        )
    ]
    
    # Generate video with clutter
    video = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        transient_objects=transient_objects,
        persistent_objects=persistent_objects,
        background_clutter=True,
        clutter_density=0.1,
        noise_level=0.05
    )
    
    print(f"✅ Generated video with clutter")
    print(f"   Video shape: {video.shape}")
    print(f"   Transient objects: {len(transient_objects)}")
    print(f"   Persistent objects: {len(persistent_objects)}")

