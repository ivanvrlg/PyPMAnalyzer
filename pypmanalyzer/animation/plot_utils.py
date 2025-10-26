"""
Plotting utilities for motion animations
"""

import numpy as np
from typing import Tuple


def get_axis_limits(data: np.ndarray, padding: float = 0.1) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate axis limits with equal aspect ratio
    
    Args:
        data: Motion data array of shape (n_frames, n_coords)
        padding: Padding factor (0.1 = 10% extra space)
        
    Returns:
        Tuple of (xlim, ylim, zlim) where each is (min, max)
    """
    n_markers = data.shape[1] // 3
    
    # Extract x, y, z coordinates
    x_coords = data[:, 0::3].flatten()
    y_coords = data[:, 1::3].flatten()
    z_coords = data[:, 2::3].flatten()
    
    # Remove NaNs
    x_coords = x_coords[~np.isnan(x_coords)]
    y_coords = y_coords[~np.isnan(y_coords)]
    z_coords = z_coords[~np.isnan(z_coords)]
    
    if len(x_coords) == 0:
        return (-1, 1), (-1, 1), (-1, 1)
    
    # Get ranges
    x_range = (np.min(x_coords), np.max(x_coords))
    y_range = (np.min(y_coords), np.max(y_coords))
    z_range = (np.min(z_coords), np.max(z_coords))
    
    # Ensure equal aspect ratio
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    )
    
    if max_range == 0:
        max_range = 1
    
    x_center = (x_range[0] + x_range[1]) / 2
    y_center = (y_range[0] + y_range[1]) / 2
    z_center = (z_range[0] + z_range[1]) / 2
    
    half_range = max_range * (1 + padding) / 2
    
    xlim = (x_center - half_range, x_center + half_range)
    ylim = (y_center - half_range, y_center + half_range)
    zlim = (z_center - half_range, z_center + half_range)
    
    return xlim, ylim, zlim


def setup_2d_axis(ax, xlim, ylim, xlabel: str, ylabel: str, title: str):
    """
    Configure a 2D axis for motion animation
    
    Args:
        ax: Matplotlib axis
        xlim: X-axis limits
        ylim: Y-axis limits
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Axis title
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_facecolor('white')