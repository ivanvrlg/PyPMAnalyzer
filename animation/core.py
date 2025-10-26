"""
Core animation rendering components - shared logic for all animation types
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class SkeletonRenderer:
    """
    Renders bones and markers for a single skeleton
    Handles the matplotlib artist creation and updates
    """
    
    def __init__(
        self,
        ax,
        n_markers: int,
        bone_connections: List,
        marker_colors: List[str],
        bone_color: str,
        marker_edge_color: str,
        bone_width: float = 2.0,
        marker_size: int = 6,
        show_bones: bool = True
    ):
        """
        Initialize skeleton renderer
        
        Args:
            ax: Matplotlib axis to draw on
            n_markers: Number of markers
            bone_connections: List of bone connections
            marker_colors: List of marker colors
            bone_color: Color for bones
            marker_edge_color: Color for marker edges
            bone_width: Width of bone lines
            marker_size: Size of marker dots
            show_bones: Whether to show skeleton connections
        """
        self.ax = ax
        self.n_markers = n_markers
        self.bone_connections = bone_connections
        self.show_bones = show_bones
        
        # Create matplotlib artists
        self.bone_lines = []
        self.marker_dots = []
        
        # Initialize bones
        if show_bones:
            for connection in bone_connections:
                line, = ax.plot([], [], '-', color=bone_color,
                               linewidth=bone_width, alpha=0.8, zorder=1)
                self.bone_lines.append(line)
        
        # Initialize markers
        for i in range(n_markers):
            color = marker_colors[i] if i < len(marker_colors) else '#808080'
            marker, = ax.plot([], [], 'o', color=color,
                             markersize=marker_size,
                             markeredgecolor=marker_edge_color,
                             markeredgewidth=0.5, zorder=2)
            self.marker_dots.append(marker)
    
    def update_sagittal(self, frame_data: np.ndarray):
        """
        Update skeleton for sagittal view (x-z plane)
        
        Args:
            frame_data: Flat array of coordinates [x1, y1, z1, x2, y2, z2, ...]
        """
        # Update bones
        if self.show_bones:
            for line, connection in zip(self.bone_lines, self.bone_connections):
                if isinstance(connection, int):
                    connection = [connection]
                
                valid_indices = [idx for idx in connection if idx < self.n_markers]
                if len(valid_indices) >= 2:
                    x_vals = frame_data[np.array(valid_indices) * 3]
                    z_vals = frame_data[np.array(valid_indices) * 3 + 2]
                    line.set_data(x_vals, z_vals)
                else:
                    line.set_data([], [])
        
        # Update markers
        for i, marker in enumerate(self.marker_dots):
            if i < self.n_markers:
                x_val = frame_data[i * 3]
                z_val = frame_data[i * 3 + 2]
                marker.set_data([x_val], [z_val])
    
    def update_frontal(self, frame_data: np.ndarray):
        """
        Update skeleton for frontal view (y-z plane)
        
        Args:
            frame_data: Flat array of coordinates [x1, y1, z1, x2, y2, z2, ...]
        """
        # Update bones
        if self.show_bones:
            for line, connection in zip(self.bone_lines, self.bone_connections):
                if isinstance(connection, int):
                    connection = [connection]
                
                valid_indices = [idx for idx in connection if idx < self.n_markers]
                if len(valid_indices) >= 2:
                    y_vals = frame_data[np.array(valid_indices) * 3 + 1]
                    z_vals = frame_data[np.array(valid_indices) * 3 + 2]
                    line.set_data(y_vals, z_vals)
                else:
                    line.set_data([], [])
        
        # Update markers
        for i, marker in enumerate(self.marker_dots):
            if i < self.n_markers:
                y_val = frame_data[i * 3 + 1]
                z_val = frame_data[i * 3 + 2]
                marker.set_data([y_val], [z_val])
    
    def get_artists(self) -> List:
        """Return all matplotlib artists for blitting"""
        return self.bone_lines + self.marker_dots


class AnimationManager:
    """
    Manages the animation loop and frame updates
    """
    
    def __init__(
        self,
        data: np.ndarray,
        skip_frames: int = 1
    ):
        """
        Args:
            data: Motion data (n_frames, n_coords)
            skip_frames: Frame skip for faster playback
        """
        self.data = data
        self.skip_frames = skip_frames
        self.n_frames = data.shape[0]
        self.n_anim_frames = self.n_frames // skip_frames
    
    def get_frame_data(self, frame_idx: int) -> np.ndarray:
        """
        Get frame data for animation index
        
        Args:
            frame_idx: Animation frame index
            
        Returns:
            Frame data array
        """
        actual_frame = frame_idx * self.skip_frames
        if actual_frame >= self.n_frames:
            actual_frame = self.n_frames - 1
        return self.data[actual_frame]