"""
High-level visualization functions - convenient wrappers for common use cases
"""

import numpy as np
from typing import List, Optional

from ppma_core import PrincipalMovementAnalysis, TrialKey
from animation import Animator2D, AnimatorComparison, PMReconstructor


def visualize_original_motion(
    data: np.ndarray,
    title: str = "Original Motion",
    save_path: Optional[str] = None,
    fps: int = 30,
    skip_frames: int = 1,
    show_bones: bool = True,
    style: str = 'qtm',
    capture_fps: Optional[float] = None,
    playback_speed: float = 1.0
):
    """
    Quick visualization of original motion data
    
    Args:
        data: (n_frames, n_coords) array
        title: Title
        save_path: Save path (None = display)
        fps: OUTPUT video frames per second (playback rate)
        skip_frames: Frame skip (1=use all frames, 2=use every 2nd frame, etc.)
        show_bones: Show skeleton
        style: 'qtm' (colored + yellow) or 'black' (black + black)
        capture_fps: Optional - if provided, will auto-calculate fps for real-time
        playback_speed: Speed multiplier (1.0=real-time, 2.0=2x speed, 0.5=half speed)
        
    Note:
        For REAL-TIME playback: output_fps = capture_fps / skip_frames
        
        Example: Data captured at 100 Hz
        - Real-time, all frames: fps=100, skip_frames=1
        - Real-time, half frames: fps=50, skip_frames=2  
        - Real-time, 1/3 frames: fps=33, skip_frames=3
        - 2x speed, all frames: fps=200, skip_frames=1
        - Slow motion, all frames: fps=50, skip_frames=1
    """
    # Auto-calculate fps for real-time if capture_fps provided
    if capture_fps is not None:
        fps = int((capture_fps / skip_frames) * playback_speed)
        print(f"Auto-calculated output FPS: {fps} "
              f"(capture: {capture_fps} Hz, skip: {skip_frames}, speed: {playback_speed}x)")
    
    animator = Animator2D()
    return animator.animate(
        data, title=title, save_path=save_path,
        fps=fps, skip_frames=skip_frames,
        show_bones=show_bones, style=style
    )


def visualize_principal_movements(
    pma: PrincipalMovementAnalysis,
    pm_indices: List[int],
    amplifications: Optional[List[float]] = None,
    preprocessor=None,
    trial_key: Optional[TrialKey] = None,
    n_frames: int = 60,
    title: str = "Principal Movements",
    save_path: Optional[str] = None,
    fps: int = 20,
    show_bones: bool = True,
    style: str = 'qtm'
):
    """
    Visualize specific principal movements
    
    Args:
        pma: Fitted PMA object
        pm_indices: Which PMs to show (0-indexed)
        amplifications: Amplification factors
        preprocessor: PreProcessor for inverse transforms
        trial_key: Reference trial
        n_frames: Frames per oscillation
        title: Title
        save_path: Save path
        fps: Frames per second
        show_bones: Show skeleton
        style: 'qtm' or 'black'
    """
    if amplifications is None:
        amplifications = [2.0] * len(pm_indices)
    
    # Reconstruct PMs
    reconstructor = PMReconstructor(pma, preprocessor)
    pm_data_dict = reconstructor.reconstruct_multiple_pms(
        pm_indices, amplifications, n_frames, trial_key
    )
    
    # Animate
    animator = AnimatorComparison()
    return animator.animate(
        pm_data_dict,
        title=title,
        save_path=save_path,
        fps=fps,
        skip_frames=1,
        show_bones=show_bones,
        style=style
    )