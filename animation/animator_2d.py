"""
2D Motion animations - single motion visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import Optional, List
from pathlib import Path

from .marker_sets import QualisysMarkerSet
from .styles import AnimationStyle
from .plot_utils import get_axis_limits, setup_2d_axis
from .core import SkeletonRenderer, AnimationManager


class Animator2D:
    """
    Create 2D animations of motion data
    """
    
    def __init__(self, bone_connections: Optional[List] = None):
        """
        Args:
            bone_connections: List of bone connections (default: Qualisys 53-marker)
        """
        if bone_connections is None:
            self.bone_connections = QualisysMarkerSet.BONE_CONNECTIONS
        else:
            self.bone_connections = bone_connections
    
    def animate(
        self,
        data: np.ndarray,
        title: str = "Motion",
        skip_frames: int = 1,
        show_sagittal: bool = True,
        show_frontal: bool = True,
        save_path: Optional[str] = None,
        fps: int = 30,
        show_bones: bool = True,
        style: str = 'qtm',
        bone_width: float = 2.0,
        marker_size: int = 6
    ) -> Optional[FuncAnimation]:
        """
        Create 2D animation with simple presets
        
        Args:
            data: (n_frames, n_coords) array
            title: Animation title
            skip_frames: Frame skip for faster animation
            show_sagittal: Show sagittal (x-z) view
            show_frontal: Show frontal (y-z) view
            save_path: Path to save video (None = display only)
            fps: Frames per second
            show_bones: Show skeleton connections
            style: 'qtm' (colored + yellow) or 'black' (black + black)
            bone_width: Line width for bones
            marker_size: Size of marker dots
            
        Returns:
            FuncAnimation object (if not saving)
        """
        n_frames = data.shape[0]
        n_markers = data.shape[1] // 3
        
        # Get colors based on style
        marker_colors = AnimationStyle.get_marker_colors(n_markers, style)
        bone_color = AnimationStyle.get_bone_color(style)
        marker_edge_color = AnimationStyle.get_marker_edge_color(style)
        
        # Get axis limits
        xlim, ylim, zlim = get_axis_limits(data)
        
        # Determine subplot layout
        n_views = int(show_sagittal) + int(show_frontal)
        if n_views == 0:
            raise ValueError("Must show at least one view")
        
        fig, axes = plt.subplots(1, n_views, figsize=(7*n_views, 6))
        if n_views == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        fig.patch.set_facecolor('white')
        
        # Setup axes and renderers
        renderers = []
        view_idx = 0
        
        if show_sagittal:
            ax = axes[view_idx]
            setup_2d_axis(ax, xlim, zlim, 'X (AP direction)', 'Z (Vertical)', 'Sagittal View')
            renderer = SkeletonRenderer(
                ax, n_markers, self.bone_connections,
                marker_colors, bone_color, marker_edge_color,
                bone_width, marker_size, show_bones
            )
            renderers.append(('sagittal', renderer))
            view_idx += 1
        
        if show_frontal:
            ax = axes[view_idx]
            setup_2d_axis(ax, ylim, zlim, 'Y (ML direction)', 'Z (Vertical)', 'Frontal View')
            renderer = SkeletonRenderer(
                ax, n_markers, self.bone_connections,
                marker_colors, bone_color, marker_edge_color,
                bone_width, marker_size, show_bones
            )
            renderers.append(('frontal', renderer))
        
        # Create animation manager
        anim_mgr = AnimationManager(data, skip_frames)
        
        def init():
            all_artists = []
            for _, renderer in renderers:
                all_artists.extend(renderer.get_artists())
            return all_artists
        
        def update(frame_idx):
            frame_data = anim_mgr.get_frame_data(frame_idx)
            
            for view_type, renderer in renderers:
                if view_type == 'sagittal':
                    renderer.update_sagittal(frame_data)
                else:  # frontal
                    renderer.update_frontal(frame_data)
            
            all_artists = []
            for _, renderer in renderers:
                all_artists.extend(renderer.get_artists())
            return all_artists
        
        # Create animation
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=anim_mgr.n_anim_frames, interval=1000/fps,
            blit=True, repeat=True
        )
        
        # Save or display
        if save_path:
            print(f"Saving animation to {save_path}...")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            writer = FFMpegWriter(fps=fps, bitrate=3000)
            anim.save(save_path, writer=writer)
            plt.close(fig)
            print(f"✓ Saved to {save_path}")
            return None
        else:
            plt.tight_layout()
            return anim