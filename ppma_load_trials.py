

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
import yaml
import json

# Your imports
from qualisys_loader import QualisysLoader

from ppma_core import (TrialKey, SubjectInfo)

# ============================================================================
# DATA STRUCTURES FOR TRIAL SPECIFICATION
# ============================================================================

@dataclass
class TrialSpec:
    """
    Specification for a single trial with flexible path handling

    Attributes:
        session_path: Path to session directory
        trial_name: Name of the trial
        subject_id: Subject identifier (optional, will be auto-generated if None)
    """
    session_path: str
    trial_name: str
    subject_id: Optional[str] = None

    def __post_init__(self):
        """Convert paths to Path objects"""
        self.session_path = Path(self.session_path)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_multiple_trials(
    session_path: str,
    trial_names: List[str],
    subject_ids: List[str] = None
) -> Dict[TrialKey, np.ndarray]:
    """
    Load multiple trials from Qualisys
    
    Args:
        session_path: Path to session directory
        trial_names: List of trial names
        subject_ids: Optional list of subject IDs (if None, uses trial index)
        
    Returns:
        Dictionary mapping TrialKey to marker coordinates
    """
    data_dict = {}
    
    if subject_ids is None:
        subject_ids = [f"{i:02d}" for i in range(len(trial_names))]
    
    for trial_name, subj_id in zip(trial_names, subject_ids):
        print(f"\n{'='*60}")
        print(f"Loading: {trial_name} (Subject {subj_id})")
        print(f"{'='*60}")
        
        # Load with QualisysLoader
        loader = QualisysLoader(session_path, trial_name)
        loader.load_marker_file()
        
        # Get coordinates (shape: n_frames x n_coords)
        coords = loader.get_marker_coordinates()  # Already in meters
        
        # Create trial key
        key = TrialKey(subject=subj_id, trial=trial_name)
        data_dict[key] = coords
        
        print(f"  ✓ Loaded: {coords.shape[0]} frames, {coords.shape[1]//3} markers")
        
        # Check for NaNs
        nan_count = np.sum(np.isnan(coords))
        if nan_count > 0:
            nan_pct = 100 * nan_count / coords.size
            print(f"  ⚠ Warning: {nan_count} NaNs ({nan_pct:.2f}%)")
    
    return data_dict


def load_trials_from_specs(
    trial_specs: List[TrialSpec],
    skip_errors: bool = False
) -> Dict[TrialKey, np.ndarray]:
    """
    Load multiple trials from different directories using TrialSpec objects

    This function allows loading trials from different session directories,
    making it easy to combine data from multiple subjects or recording sessions.

    Args:
        trial_specs: List of TrialSpec objects specifying trials to load
        skip_errors: If True, skip trials that fail to load instead of raising error

    Returns:
        Dictionary mapping TrialKey to marker coordinates

    Example:
        >>> # Load 5 walking trials from 5 different subjects
        >>> specs = [
        ...     TrialSpec("/data/subject01/session1", "Walk_01", "S01"),
        ...     TrialSpec("/data/subject02/session1", "Walk_01", "S02"),
        ...     TrialSpec("/data/subject03/session2", "Walk_02", "S03"),
        ...     TrialSpec("/data/subject04/session1", "Walk_01", "S04"),
        ...     TrialSpec("/data/subject05/session3", "Walk_01", "S05"),
        ... ]
        >>> data_dict = load_trials_from_specs(specs)
    """
    data_dict = {}
    errors = []

    for i, spec in enumerate(trial_specs):
        # Auto-generate subject ID if not provided
        subject_id = spec.subject_id if spec.subject_id else f"{i:02d}"

        print(f"\n{'='*60}")
        print(f"Loading: {spec.trial_name} (Subject {subject_id})")
        print(f"  Path: {spec.session_path}")
        print(f"{'='*60}")

        try:
            # Load with QualisysLoader
            loader = QualisysLoader(str(spec.session_path), spec.trial_name)
            loader.load_marker_file()

            # Get coordinates (shape: n_frames x n_coords)
            coords = loader.get_marker_coordinates()  # Already in meters

            # Create trial key
            key = TrialKey(subject=subject_id, trial=spec.trial_name)
            data_dict[key] = coords

            print(f"  ✓ Loaded: {coords.shape[0]} frames, {coords.shape[1]//3} markers")

            # Check for NaNs
            nan_count = np.sum(np.isnan(coords))
            if nan_count > 0:
                nan_pct = 100 * nan_count / coords.size
                print(f"  ⚠ Warning: {nan_count} NaNs ({nan_pct:.2f}%)")

        except Exception as e:
            error_msg = f"Failed to load {spec.trial_name} from {spec.session_path}: {str(e)}"
            errors.append(error_msg)

            if skip_errors:
                print(f"  ✗ ERROR: {str(e)}")
                print(f"  → Skipping this trial...")
            else:
                raise RuntimeError(error_msg) from e

    # Report summary
    print(f"\n{'='*60}")
    print(f"LOADING SUMMARY")
    print(f"{'='*60}")
    print(f"  Successfully loaded: {len(data_dict)}/{len(trial_specs)} trials")
    if errors:
        print(f"  Failed: {len(errors)} trials")
        for error in errors:
            print(f"    - {error}")
    print(f"{'='*60}\n")

    return data_dict


def load_trials_from_paths(
    trial_paths: List[Tuple[str, str]],
    subject_ids: Optional[List[str]] = None,
    skip_errors: bool = False
) -> Dict[TrialKey, np.ndarray]:
    """
    Load multiple trials from different directories using (session_path, trial_name) tuples

    Convenience function for loading trials from different locations without
    creating TrialSpec objects explicitly.

    Args:
        trial_paths: List of (session_path, trial_name) tuples
        subject_ids: Optional list of subject IDs (auto-generated if None)
        skip_errors: If True, skip trials that fail to load

    Returns:
        Dictionary mapping TrialKey to marker coordinates

    Example:
        >>> # Load walking trials from different subjects
        >>> trial_paths = [
        ...     ("/data/subject01/session1", "Walk_01"),
        ...     ("/data/subject02/session1", "Walk_01"),
        ...     ("/data/subject03/session2", "Walk_02"),
        ... ]
        >>> subject_ids = ["S01", "S02", "S03"]
        >>> data_dict = load_trials_from_paths(trial_paths, subject_ids)
    """
    # Convert to TrialSpec objects
    specs = []
    for i, (session_path, trial_name) in enumerate(trial_paths):
        subject_id = subject_ids[i] if subject_ids else None
        specs.append(TrialSpec(session_path, trial_name, subject_id))

    return load_trials_from_specs(specs, skip_errors=skip_errors)


def load_trials_from_config(
    config_path: str,
    skip_errors: bool = False
) -> Dict[TrialKey, np.ndarray]:
    """
    Load multiple trials from a YAML or JSON configuration file

    This is the most convenient way to load multiple trials from different
    directories. Simply create a config file listing all trials to load.

    Args:
        config_path: Path to YAML or JSON configuration file
        skip_errors: If True, skip trials that fail to load

    Returns:
        Dictionary mapping TrialKey to marker coordinates

    Config file format (YAML):
        trials:
          - session_path: /data/subject01/session1
            trial_name: Walk_01
            subject_id: S01
          - session_path: /data/subject02/session1
            trial_name: Walk_01
            subject_id: S02
          - session_path: /data/subject03/session2
            trial_name: Walk_02
            subject_id: S03

    Config file format (JSON):
        {
          "trials": [
            {
              "session_path": "/data/subject01/session1",
              "trial_name": "Walk_01",
              "subject_id": "S01"
            },
            {
              "session_path": "/data/subject02/session1",
              "trial_name": "Walk_01",
              "subject_id": "S02"
            }
          ]
        }

    Example:
        >>> # Create a config file (trials_config.yaml)
        >>> data_dict = load_trials_from_config("trials_config.yaml")
    """
    config_path = Path(config_path)

    # Load config file
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .yaml, .yml, or .json")

    # Parse trial specifications
    if 'trials' not in config:
        raise ValueError("Config file must contain 'trials' key")

    specs = []
    for trial_config in config['trials']:
        spec = TrialSpec(
            session_path=trial_config['session_path'],
            trial_name=trial_config['trial_name'],
            subject_id=trial_config.get('subject_id', None)
        )
        specs.append(spec)

    print(f"\n{'='*60}")
    print(f"Loading {len(specs)} trials from config: {config_path}")
    print(f"{'='*60}")

    return load_trials_from_specs(specs, skip_errors=skip_errors)


def create_subject_info(
    trial_keys: List[TrialKey],
    heights: List[float] = None,
    masses: List[float] = None,
    genders: List[str] = None
) -> Dict[TrialKey, SubjectInfo]:
    """
    Create subject info dictionary
    
    Args:
        trial_keys: List of TrialKey objects
        heights: List of heights in meters (default: 1.75)
        masses: List of masses in kg (default: 70.0)
        genders: List of genders 'M' or 'F' (default: 'M')
        
    Returns:
        Dictionary mapping TrialKey to SubjectInfo
    """
    n_trials = len(trial_keys)
    
    if heights is None:
        heights = [1.75] * n_trials
    if masses is None:
        masses = [70.0] * n_trials
    if genders is None:
        genders = ['M'] * n_trials
    
    info_dict = {}
    for key, h, m, g in zip(trial_keys, heights, masses, genders):
        info_dict[key] = SubjectInfo(height=h, mass=m, gender=g)
    
    return info_dict