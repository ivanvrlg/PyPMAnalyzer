"""
Integration script: Load Qualisys data → PManalyzer pipeline
Demonstrates the complete workflow from data loading to PM analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import matplotlib.pyplot as plt


# PManalyzer imports
from ppma_load_trials import (
    load_multiple_trials,
    load_trials_from_specs,
    load_trials_from_paths,
    load_trials_from_config,
    create_subject_info)

from ppma_core import (
    PreProcessor,
    PrincipalMovementAnalysis,
    TrialKey,
    SubjectInfo,
    compute_rVAR,
    compute_zero_crossings,
    compute_zc_per_second,
    compute_pv_std,
    compute_rms
)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_eigenvalue_spectrum(pma: PrincipalMovementAnalysis, max_pms: int = 15):
    """Plot eigenvalue (explained variance) spectrum"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    n_show = min(max_pms, len(pma.explained_variance_ratio_))
    axes[0].bar(range(1, n_show + 1), pma.explained_variance_ratio_[:n_show],
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Principal Movement', fontsize=12)
    axes[0].set_ylabel('Explained Variance (%)', fontsize=12)
    axes[0].set_title('Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative plot
    cumsum = np.cumsum(pma.explained_variance_ratio_[:n_show])
    axes[1].plot(range(1, n_show + 1), cumsum, 'o-', linewidth=2, 
                 markersize=8, color='darkred', markerfacecolor='red')
    axes[1].axhline(80, color='gray', linestyle='--', alpha=0.5, label='80%')
    axes[1].axhline(90, color='gray', linestyle='--', alpha=0.5, label='90%')
    axes[1].set_xlabel('Principal Movement', fontsize=12)
    axes[1].set_ylabel('Cumulative Variance (%)', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.show()


def plot_rVAR_comparison(rvar_dict: Dict[TrialKey, np.ndarray], max_pms: int = 10):
    """Compare relative variance (movement structure) across trials"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_pms = min(max_pms, min(len(v) for v in rvar_dict.values()))
    x = np.arange(1, n_pms + 1)
    width = 0.8 / len(rvar_dict)
    
    for i, (key, rvar) in enumerate(rvar_dict.items()):
        offset = (i - len(rvar_dict)/2 + 0.5) * width
        ax.bar(x + offset, rvar[:n_pms], width, label=str(key), alpha=0.8)
    
    ax.set_xlabel('Principal Movement', fontsize=12)
    ax.set_ylabel('Relative Variance (%)', fontsize=12)
    ax.set_title('Movement Structure Comparison (rVAR)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_pp_time_series(
    pp_dict: Dict[TrialKey, np.ndarray],
    pm_indices: List[int] = [0, 1, 2, 3]
):
    """Plot PP time series for selected PMs"""
    n_pms = len(pm_indices)
    fig, axes = plt.subplots(n_pms, 1, figsize=(14, 3*n_pms), sharex=True)
    
    if n_pms == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pp_dict)))
    
    for pm_idx, ax in zip(pm_indices, axes):
        for (key, pp), color in zip(pp_dict.items(), colors):
            if pm_idx < pp.shape[1]:
                ax.plot(pp[:, pm_idx], label=str(key), color=color, alpha=0.7)
        
        ax.set_ylabel(f'PM{pm_idx+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        if pm_idx == pm_indices[0]:
            ax.set_title('Principal Positions (PP) Time Series', 
                        fontsize=14, fontweight='bold')
    
    axes[-1].set_xlabel('Frame', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_control_metrics(
    pa_dict: Dict[TrialKey, np.ndarray],
    fs: float = 100.0,
    max_pms: int = 10
):
    """Plot control metrics (NoZC, NoZCps) from PA"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute metrics
    nozc_dict = {}
    nozc_ps_dict = {}
    
    for key, pa in pa_dict.items():
        nozc_dict[key] = compute_zero_crossings(pa)
        nozc_ps_dict[key] = compute_zc_per_second(pa, fs)
    
    # Plot NoZC
    n_pms = min(max_pms, min(len(v) for v in nozc_dict.values()))
    x = np.arange(1, n_pms + 1)
    width = 0.8 / len(nozc_dict)
    
    for i, (key, nozc) in enumerate(nozc_dict.items()):
        offset = (i - len(nozc_dict)/2 + 0.5) * width
        axes[0].bar(x + offset, nozc[:n_pms], width, label=str(key), alpha=0.8)
    
    axes[0].set_xlabel('Principal Movement', fontsize=12)
    axes[0].set_ylabel('Number of Zero Crossings', fontsize=12)
    axes[0].set_title('Control Interventions (NoZC)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot NoZCps
    for i, (key, nozc_ps) in enumerate(nozc_ps_dict.items()):
        offset = (i - len(nozc_ps_dict)/2 + 0.5) * width
        axes[1].bar(x + offset, nozc_ps[:n_pms], width, label=str(key), alpha=0.8)
    
    axes[1].set_xlabel('Principal Movement', fontsize=12)
    axes[1].set_ylabel('Zero Crossings per Second', fontsize=12)
    axes[1].set_title('Control Frequency (NoZCps)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pm_analysis_pipeline(
    session_path: Optional[str] = None,
    trial_names: Optional[List[str]] = None,
    subject_ids: Optional[List[str]] = None,
    subject_info: Optional[Dict[TrialKey, SubjectInfo]] = None,
    normalize_method: str = 'height',
    filter_cutoff: float = 7.0,
    n_components: int = 12,
    fs: float = 100.0,
    trim_start: float = 0.0,
    trim_end: Optional[float] = None,
    config_path: Optional[str] = None,
    data_dict: Optional[Dict[TrialKey, np.ndarray]] = None
):
    """
    Complete PM analysis pipeline

    This function supports two modes:
    1. Load data from a single session (original API)
    2. Use pre-loaded data from multiple sessions (new flexible API)

    Args:
        session_path: Path to Qualisys session directory (required if data_dict not provided)
        trial_names: List of trial names to analyze (required if data_dict not provided)
        subject_ids: Optional subject IDs (only used if data_dict not provided)
        subject_info: Optional subject metadata (height, mass, gender)
        normalize_method: 'height', 'MED', 'max_range', 'none'
        filter_cutoff: Low-pass filter cutoff in Hz (None for no filtering)
        n_components: Number of PMs to extract
        fs: Sampling frequency in Hz
        trim_start: Time in seconds to trim from start of each trial
        trim_end: Time in seconds to trim from end of each trial
        data_dict: Pre-loaded data dictionary (use this to bypass loading step)

    Returns:
        Tuple of (preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics)

    Examples:
        >>> # Mode 1: Traditional usage (single session)
        >>> results = run_pm_analysis_pipeline(
        ...     session_path="/data/subject01/session1",
        ...     trial_names=["Walk_01", "Walk_02"],
        ...     n_components=12
        ... )

        >>> # Mode 2: Pre-loaded data from multiple sessions
        >>> data_dict = load_trials_from_paths([
        ...     ("/data/subject01/session1", "Walk_01"),
        ...     ("/data/subject02/session1", "Walk_01"),
        ...     ("/data/subject03/session2", "Walk_02"),
        ... ], subject_ids=["S01", "S02", "S03"])
        >>> results = run_pm_analysis_pipeline(
        ...     data_dict=data_dict,
        ...     n_components=12
        ... )
    """

    print("\n" + "="*70)
    print("PRINCIPAL MOVEMENT ANALYSIS PIPELINE")
    print("="*70)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================

    if data_dict is None:
        # Mode 1: Load from session_path
        if config_path is not None:
            data_dict = load_trials_from_config("examples/trials_config.json", skip_errors=True)
            print(f"\nLoaded {len(data_dict)} trials from config file")
            return data_dict
        
        elif session_path is None or trial_names is None:
            raise ValueError(
                "Either provide (session_path + trial_names) OR data_dict. "
                "Got neither session_path nor data_dict."
            )

        print("\n### STEP 1: LOADING DATA ###\n")
        data_dict = load_multiple_trials(session_path, trial_names, subject_ids)
    else:

        # Mode 2: Use pre-loaded data
        print("\n### STEP 1: USING PRE-LOADED DATA ###\n")
        print(f"  Using {len(data_dict)} pre-loaded trials")
        for key in data_dict.keys():
            coords = data_dict[key]
            print(f"  {key}: {coords.shape[0]} frames, {coords.shape[1]//3} markers")
    
    # TRIM DATA
    if trim_start > 0 or trim_end is not None:
        print(f"\nTrimming data: start={trim_start}s, end={trim_end}s")
        data_dict_trimmed = {}
        for key, data in data_dict.items():
            start_frame = int(trim_start * fs)
            end_frame = int(trim_end * fs) if trim_end else data.shape[0]
            data_dict_trimmed[key] = data[start_frame:end_frame]
            print(f"  {key}: {data.shape[0]} → {data_dict_trimmed[key].shape[0]} frames")
        data_dict = data_dict_trimmed
    
    trial_keys = list(data_dict.keys())
    
    # Create subject info if not provided
    if subject_info is None:
        subject_info = create_subject_info(trial_keys)
    
    # ========================================================================
    # STEP 2: Preprocess
    # ========================================================================
    print("\n" + "="*70)
    print("### STEP 2: PREPROCESSING ###")
    print("="*70)
    
    preprocessor = PreProcessor(
        center=True,
        normalize_method=normalize_method,
        weight_method='equal',  # TODO: Implement body_segments with your marker set
        order='center_weight_normalize',
        filter_cutoff=filter_cutoff,
        filter_order=3
    )
    
    data_preprocessed = preprocessor.fit_transform(data_dict, subject_info, fs)
    
    # ========================================================================
    # STEP 3: PCA
    # ========================================================================
    print("\n" + "="*70)
    print("### STEP 3: PRINCIPAL COMPONENT ANALYSIS ###")
    print("="*70)
    
    pma = PrincipalMovementAnalysis(n_components=n_components)
    pma.fit(data_preprocessed)
    
    # ========================================================================
    # STEP 4: Compute Derivatives
    # ========================================================================
    print("\n" + "="*70)
    print("### STEP 4: COMPUTING DERIVATIVES ###")
    print("="*70)
    
    print("Computing principal velocities (PV)...")
    pv_dict = pma.compute_pv(fs)
    
    print("Computing principal accelerations (PA)...")
    pa_dict = pma.compute_pa(fs)
    
    print("✓ Derivatives computed")
    
    # ========================================================================
    # STEP 5: Compute Metrics
    # ========================================================================
    print("\n" + "="*70)
    print("### STEP 5: COMPUTING METRICS ###")
    print("="*70)
    
    metrics = {}
    
    print("Computing rVAR (relative variance)...")
    metrics['rVAR'] = compute_rVAR(pma.pp_)
    
    print("Computing PV-STD (movement speed variability)...")
    metrics['PV_STD'] = compute_pv_std(pv_dict)
    
    print("Computing RMS...")
    metrics['PP_RMS'] = compute_rms(pma.pp_)
    metrics['PV_RMS'] = compute_rms(pv_dict)
    
    print("Computing control metrics (NoZC, NoZCps)...")
    for key, pa in pa_dict.items():
        if key not in metrics:
            metrics[key] = {}
        metrics[key]['NoZC'] = compute_zero_crossings(pa)
        metrics[key]['NoZCps'] = compute_zc_per_second(pa, fs)
    
    print("✓ All metrics computed")
    
    # ========================================================================
    # STEP 6: Display Results
    # ========================================================================
    print("\n" + "="*70)
    print("### RESULTS SUMMARY ###")
    print("="*70)
    
    print(f"\n✓ Analyzed {len(trial_keys)} trials")
    print(f"✓ Extracted {pma.n_components_fitted_} principal movements")
    print(f"✓ Total variance explained: {np.sum(pma.explained_variance_ratio_):.2f}%")
    
    print("\n### Movement Structure (rVAR) ###")
    for key in trial_keys:
        rvar = metrics['rVAR'][key]
        print(f"\n{key}:")
        for i in range(min(5, len(rvar))):
            print(f"  PM{i+1}: {rvar[i]:.2f}%")
    
    return preprocessor, pma, pma.pp_, pv_dict, pa_dict, metrics


# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def animate_original_motion(
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
    Create animation of original motion data
    
    Args:
        data: (n_frames, n_coords) marker coordinate array
        title: Animation title
        save_path: Path to save video (None = display)
        fps: OUTPUT video frames per second (ignored if capture_fps provided)
        skip_frames: Frame skip (1=all frames, 2=every other, etc.)
        show_bones: Show skeleton connections
        style: 'qtm' (colored + yellow) or 'black' (black + black)
        capture_fps: Capture rate in Hz (e.g., 100 for 100Hz mocap)
        playback_speed: Speed multiplier (1.0=real-time, 2.0=double speed)
        
    Returns:
        Animation object (if save_path is None)
        
    Example:
        >>> # Real-time playback of 100Hz data
        >>> animate_original_motion(
        ...     data=marker_data,
        ...     capture_fps=100,
        ...     skip_frames=2,
        ...     playback_speed=1.0,  # Results in 50fps video
        ...     save_path="walk.mp4"
        ... )
    """
    from visualize import visualize_original_motion
    
    return visualize_original_motion(
        data=data,
        title=title,
        save_path=save_path,
        fps=fps,
        skip_frames=skip_frames,
        show_bones=show_bones,
        style=style,
        capture_fps=capture_fps,
        playback_speed=playback_speed
    )


def animate_principal_movements(
    pma: PrincipalMovementAnalysis,
    preprocessor: PreProcessor,
    pm_indices: List[int],
    amplifications: Optional[List[float]] = None,
    trial_key: Optional[TrialKey] = None,
    n_frames: int = 60,
    title: str = "Principal Movements",
    save_path: Optional[str] = None,
    fps: int = 20,
    show_bones: bool = True,
    style: str = 'qtm'
):
    """
    Create animation of principal movements
    
    Args:
        pma: Fitted PrincipalMovementAnalysis object
        preprocessor: PreProcessor used for data
        pm_indices: Which PMs to visualize (0-indexed)
        amplifications: Amplification factors (default: 3.0 for each)
        trial_key: Reference trial for mean posture
        n_frames: Frames per PM oscillation
        title: Animation title
        save_path: Path to save video (None = display)
        fps: Frames per second
        show_bones: Show skeleton
        style: 'qtm' or 'black'
        
    Returns:
        Animation object (if save_path is None)
        
    Example:
        >>> # After running pipeline
        >>> animate_principal_movements(
        ...     pma=pma,
        ...     preprocessor=preprocessor,
        ...     pm_indices=[0, 1, 2, 3],
        ...     amplifications=[3.0, 3.0, 3.0, 3.0],
        ...     trial_key=list(pp_dict.keys())[0],
        ...     save_path="output/PMs_1-4.mp4",
        ...     style='qtm'
        ... )
    """
    from visualize import visualize_principal_movements
    
    if amplifications is None:
        amplifications = [3.0] * len(pm_indices)  # Higher default for visibility
    
    return visualize_principal_movements(
        pma=pma,
        pm_indices=pm_indices,
        amplifications=amplifications,
        preprocessor=preprocessor,
        trial_key=trial_key,
        n_frames=n_frames,
        title=title,
        save_path=save_path,
        fps=fps,
        show_bones=show_bones,
        style=style
    )


def batch_export_pms(
    pma: PrincipalMovementAnalysis,
    preprocessor: PreProcessor,
    trial_key: TrialKey,
    output_dir: str = "output/pms",
    n_pms: int = 8,
    amplification: float = 3.0,
    n_frames: int = 60,
    fps: int = 20,
    styles: List[str] = ['qtm', 'black']
):
    """
    Batch export multiple PMs in different styles
    
    Args:
        pma: Fitted PrincipalMovementAnalysis
        preprocessor: PreProcessor
        trial_key: Reference trial
        output_dir: Output directory
        n_pms: Number of PMs to export
        amplification: Amplification factor
        n_frames: Frames per oscillation
        fps: Frames per second
        styles: List of styles to export
        
    Example:
        >>> # After running pipeline
        >>> ref_key = list(pp_dict.keys())[0]
        >>> batch_export_pms(
        ...     pma=pma,
        ...     preprocessor=preprocessor,
        ...     trial_key=ref_key,
        ...     output_dir="output/all_pms",
        ...     n_pms=8,
        ...     styles=['qtm', 'black']
        ... )
    """
    from visualize import visualize_principal_movements
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting {n_pms} PMs in {len(styles)} style(s)...")
    print(f"Output directory: {output_path}")
    
    for pm_idx in range(n_pms):
        print(f"\nExporting PM{pm_idx+1}...")
        
        for style in styles:
            save_path = output_path / f"PM{pm_idx+1:02d}_{style}.mp4"
            
            visualize_principal_movements(
                pma=pma,
                pm_indices=[pm_idx],
                amplifications=[amplification],
                preprocessor=preprocessor,
                trial_key=trial_key,
                n_frames=n_frames,
                title=f"PM{pm_idx+1} - {style.upper()}",
                save_path=str(save_path),
                fps=fps,
                show_bones=True,
                style=style
            )
            
            print(f"  ✓ {style} style")
    
    print(f"\n✓ Exported {n_pms} PMs to {output_path}")