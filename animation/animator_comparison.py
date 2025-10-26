"""
Multi-PM comparison animations - side-by-side principal movement visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import Dict, Optional, List
from pathlib import Path

from .marker_sets import QualisysMarkerSet
from .styles import AnimationStyle
from .plot_utils import get_axis_limits, setup_2d_axis
from .core import SkeletonRenderer, AnimationManager


class AnimatorComparison:
    """
    Create side-by-side animations comparing multiple PMs
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
        pm_data_dict: Dict[int, np.ndarray],
        title: str = "Principal Movements",
        skip_frames: int = 1,
        save_path: Optional[str] = None,
        fps: int = 30,
        show_bones: bool = True,
        style: str = 'qtm',
        bone_width: float = 2.0,
        marker_size: int = 4,
        show_pp_activity: bool = False,
        pp_values: Optional[Dict[int, np.ndarray]] = None
    ):
        """
        Animate multiple PMs side-by-side
        
        Args:
            pm_data_dict: {pm_index: data} for each PM
            title: Animation title
            skip_frames: Frame skip
            save_path: Save path
            fps: Frames per second
            show_bones: Show skeleton
            style: 'qtm' or 'black'
            bone_width: Bone line width
            marker_size: Marker size
            show_pp_activity: Show PP bar plot
            pp_values: PP values for bar plot
        """
        n_pms = len(pm_data_dict)
        pm_indices = sorted(pm_data_dict.keys())
        
        # Get dimensions from first PM
        first_data = pm_data_dict[pm_indices[0]]
        n_frames = first_data.shape[0]
        n_markers = first_data.shape[1] // 3
        
        # Get colors
        marker_colors = AnimationStyle.get_marker_colors(n_markers, style)
        bone_color = AnimationStyle.get_bone_color(style)
        marker_edge_color = AnimationStyle.get_marker_edge_color(style)
        
        # Calculate combined axis limits
        all_data = np.vstack([pm_data_dict[pm] for pm in pm_indices])
        xlim, ylim, zlim = get_axis_limits(all_data)
        
        # Create figure
        n_rows = 3 if show_pp_activity else 2
        fig = plt.figure(figsize=(5*n_pms, 8 if show_pp_activity else 6))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        fig.patch.set_facecolor('white')
        
        # Create renderers for each PM (sagittal and frontal)
        renderers_sag = []
        renderers_front = []
        anim_managers = {}
        
        for i, pm_idx in enumerate(pm_indices):
            # Sagittal view
            ax_sag = plt.subplot(n_rows, n_pms, i + 1)
            setup_2d_axis(ax_sag, xlim, zlim, '', '', f'Sagittal PM{pm_idx+1}')
            renderer_sag = SkeletonRenderer(
                ax_sag, n_markers, self.bone_connections,
                marker_colors, bone_color, marker_edge_color,
                bone_width, marker_size, show_bones
            )
            renderers_sag.append(renderer_sag)
            
            # Frontal view
            ax_front = plt.subplot(n_rows, n_pms, i + 1 + n_pms)
            setup_2d_axis(ax_front, ylim, zlim, '', '', f'Frontal PM{pm_idx+1}')
            renderer_front = SkeletonRenderer(
                ax_front, n_markers, self.bone_connections,
                marker_colors, bone_color, marker_edge_color,
                bone_width, marker_size, show_bones
            )
            renderers_front.append(renderer_front)
            
            # Create animation manager for this PM
            anim_managers[pm_idx] = AnimationManager(pm_data_dict[pm_idx], skip_frames)
        
        # PP activity bar plot
        ax_pp = None
        bars = None
        if show_pp_activity and pp_values:
            ax_pp = plt.subplot(n_rows, 1, n_rows)
            ax_pp.set_facecolor('white')
            bar_width = 0.6
            bar_positions = np.array(pm_indices) + 1
            bars = ax_pp.bar(bar_positions, np.zeros(len(pm_indices)), 
                            bar_width, color='steelblue', edgecolor='black')
            ax_pp.set_xlim([0.5, max(pm_indices) + 1.5])
            ax_pp.set_ylim([-100, 100])
            ax_pp.set_xlabel('Principal Position', fontsize=11)
            ax_pp.set_ylabel('Activity (%)', fontsize=11)
            ax_pp.set_title('PP Activity', fontsize=12, fontweight='bold')
            ax_pp.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_pp.grid(True, alpha=0.3, axis='y')
        
        def init():
            all_artists = []
            for renderer in renderers_sag:
                all_artists.extend(renderer.get_artists())
            for renderer in renderers_front:
                all_artists.extend(renderer.get_artists())
            return all_artists
        
        def update(frame_idx):
            # Update each PM
            for i, pm_idx in enumerate(pm_indices):
                frame_data = anim_managers[pm_idx].get_frame_data(frame_idx)
                renderers_sag[i].update_sagittal(frame_data)
                renderers_front[i].update_frontal(frame_data)
            
            # Update PP bars
            if bars and pp_values:
                for i, pm_idx in enumerate(pm_indices):
                    if pm_idx in pp_values:
                        actual_frame = frame_idx * skip_frames
                        if actual_frame < len(pp_values[pm_idx]):
                            bars[i].set_height(pp_values[pm_idx][actual_frame])
            
            all_artists = []
            for renderer in renderers_sag:
                all_artists.extend(renderer.get_artists())
            for renderer in renderers_front:
                all_artists.extend(renderer.get_artists())
            return all_artists
        
        # Create animation
        n_anim_frames = n_frames // skip_frames
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=n_anim_frames, interval=1000/fps,
            blit=True, repeat=True
        )
        
        # Save or display
        if save_path:
            print(f"Saving animation to {save_path}...")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            writer = FFMpegWriter(fps=fps, bitrate=5000)
            anim.save(save_path, writer=writer)
            plt.close(fig)
            print(f"✓ Saved to {save_path}")
            return None
        else:
            plt.tight_layout()
            return anim