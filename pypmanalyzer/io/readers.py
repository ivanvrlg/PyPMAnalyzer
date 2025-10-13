"""
Input/Output functions for reading kinematic data.

Supports multiple formats: CSV, TSV, and Qualisys-specific formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
import warnings


def read_csv(filepath, delimiter=',', skip_rows=0, skip_cols=0):
    """
    Read CSV/TSV file with kinematic data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to file
    delimiter : str, default=','
        Column delimiter (use '\t' for TSV)
    skip_rows : int, default=0
        Number of header rows to skip
    skip_cols : int, default=0
        Number of initial columns to skip (e.g., frame number, time)
        
    Returns
    -------
    data : ndarray
        Kinematic data (n_frames, n_markers*3)
    """
    df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skip_rows, header=None)
    
    # Skip initial columns if specified
    if skip_cols > 0:
        df = df.iloc[:, skip_cols:]
    
    return df.values


def read_qualisys_tsv(filepath, skip_rows=11):
    """
    Read Qualisys TSV export.
    
    Qualisys exports typically have 11 header rows by default.
    
    Parameters
    ----------
    filepath : str or Path
        Path to Qualisys TSV file
    skip_rows : int, default=11
        Number of header rows
        
    Returns
    -------
    data : ndarray
        Kinematic data (n_frames, n_markers*3)
    marker_names : list
        Names of markers (if available in header)
    """
    # Try to read marker names from header
    try:
        with open(filepath, 'r') as f:
            # Qualisys typically has marker names in a specific row
            # This is a simplified parser - adjust based on your exports
            lines = [f.readline() for _ in range(skip_rows)]
            # Marker names are usually in one of the header lines
            # You may need to adjust this based on your Qualisys version
            marker_names = None
    except:
        marker_names = None
    
    # Read data (skip initial columns: frame number, time)
    data = read_csv(filepath, delimiter='\t', skip_rows=skip_rows, skip_cols=2)
    
    return data, marker_names



def load_directory(directory, pattern='*.csv', subject_trial_pattern=r'subject(\d+)_trial(\d+)',
                   reader_func=None, **reader_kwargs):
    r"""
    Load all files matching pattern from directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing data files
    pattern : str, default='*.csv'
        Glob pattern for matching files
    subject_trial_pattern : str, default=r'subject(\d+)_trial(\d+)'
        Regex pattern to extract subject and trial IDs from filename
    reader_func : callable, optional
        Function to read individual files. If None, infers from file extension.
    **reader_kwargs
        Additional arguments passed to reader function
        
    Returns
    -------
    data_dict : dict
        {(subject_id, trial_id): array}
    metadata : dict
        {(subject_id, trial_id): {'filename': str, ...}}
        
    Examples
    --------
    >>> data_dict, meta = load_directory(
    ...     'data/pilot/',
    ...     pattern='*.tsv',
    ...     subject_trial_pattern=r'S(\d+)_T(\d+)',
    ...     reader_func=read_qualisys_tsv
    ... )
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    
    if len(files) == 0:
        raise ValueError(f"No files found matching {pattern} in {directory}")
    
    data_dict = {}
    metadata = {}
    pattern_regex = re.compile(subject_trial_pattern)
    
    for filepath in files:
        # Extract subject and trial IDs from filename
        match = pattern_regex.search(filepath.stem)
        
        if match:
            subject_id = int(match.group(1))
            trial_id = int(match.group(2))
        else:
            warnings.warn(f"Could not parse subject/trial from {filepath.name}, skipping")
            continue
        
        # Determine reader function
        if reader_func is None:
            ext = filepath.suffix.lower()
            if ext in ['.csv']:
                reader_func = read_csv
            elif ext in ['.tsv', '.txt']:
                reader_func = lambda f: read_csv(f, delimiter='\t')
            else:
                warnings.warn(f"Unknown extension {ext} for {filepath.name}, skipping")
                continue
        
        # Read data
        try:
            result = reader_func(filepath, **reader_kwargs)
            
            # Handle different return types
            if isinstance(result, tuple):
                data = result[0]  # First element is always data
            else:
                data = result
            
            key = (subject_id, trial_id)
            data_dict[key] = data
            metadata[key] = {'filename': filepath.name, 'filepath': str(filepath)}
            
            print(f"Loaded {filepath.name}: shape {data.shape}, key {key}")
            
        except Exception as e:
            warnings.warn(f"Error reading {filepath.name}: {e}")
            continue
    
    print(f"\nLoaded {len(data_dict)} trials from {len(files)} files")
    
    return data_dict, metadata


def parse_subject_info_from_filename(filename, pattern=r'S(\d+)_H(\d+)_([MF])'):
    r"""
    Parse subject information from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
    pattern : str, default=r'S(\d+)_H(\d+)_([MF])'
        Regex pattern. Default extracts: Subject ID, Height (cm), Gender
        
    Returns
    -------
    info : dict
        {'subject_id': int, 'height': float, 'gender': str}
        
    Examples
    --------
    >>> parse_subject_info_from_filename('S01_H175_M_trial1.csv')
    {'subject_id': 1, 'height': 1.75, 'gender': 'male'}
    """
    match = re.search(pattern, filename)
    if not match:
        return {}
    
    info = {
        'subject_id': int(match.group(1)),
        'height': int(match.group(2)) / 100.0,  # Convert cm to m
        'gender': 'male' if match.group(3) == 'M' else 'female'
    }
    
    return info