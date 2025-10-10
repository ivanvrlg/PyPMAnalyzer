"""
Motor control metrics from principal accelerations.

Implements control-related variables from:
- Haid et al. (2018) Front. Aging Neurosci. 10:22
- Promsri et al. (2018) Hum. Mov. Sci. 58:165-174
"""

import numpy as np


def compute_zero_crossings(pa_dict):
    """
    Count zero-crossings in PA time series (N_k).
    
    Zero-crossings indicate changes in the direction of postural
    acceleration, reflecting interventions by the control system.
    
    Exactly matches the NoZC function from optionsVariablesComp.m:
    y_sign = sign(timeSeries)
    locations_ZC = find(y_sign(1:end-1) ~= y_sign(2:end))
    Result = length(locations_ZC)
    
    Parameters
    ----------
    pa_dict : dict
        Principal accelerations {key: array(n_frames, n_components)}
        
    Returns
    -------
    n_dict : dict
        {key: array(n_components,)} with zero-crossing counts
        
    Examples
    --------
    >>> pa = np.array([1, 2, -1, -2, 3, 1, -1])
    >>> # Sign changes at indices: 1→2 and 3→4 and 5→6
    >>> # So 3 zero-crossings
    
    References
    ----------
    Haid et al. (2018) Front. Aging Neurosci. 10:22
    """
    n_dict = {}
    
    for key, pa in pa_dict.items():
        n_components = pa.shape[1]
        n_k = np.zeros(n_components)
        
        for k in range(n_components):
            # Get signs
            signs = np.sign(pa[:, k])
            
            # Find where sign changes
            sign_changes = signs[:-1] != signs[1:]
            
            # Count changes
            n_k[k] = np.sum(sign_changes)
        
        n_dict[key] = n_k
    
    return n_dict


def compute_zc_variability(pa_dict, fs=100):
    """
    Compute variability of time between zero-crossings (σ_k).
    
    This quantifies the temporal regularity of control interventions.
    Higher σ indicates more variable control timing.
    
    Matches stdTbZC from optionsVariablesComp.m:
    y_sign = sign(timeSeries)
    locations_ZC = find(y_sign(1:end-1) ~= y_sign(2:end))
    frames_between_ZC = locations_ZC(2:end) - locations_ZC(1:end-1)
    Result = std(frames_between_ZC / frequency)
    
    Parameters
    ----------
    pa_dict : dict
        Principal accelerations {key: array(n_frames, n_components)}
    fs : float, default=100
        Sampling frequency in Hz
        
    Returns
    -------
    sigma_dict : dict
        {key: array(n_components,)} with σ values in seconds
        
    Notes
    -----
    Returns NaN for components with fewer than 2 zero-crossings.
    
    References
    ----------
    Haid et al. (2018) Front. Aging Neurosci. 10:22
    Promsri et al. (2018) Hum. Mov. Sci. 58:165-174
    """
    sigma_dict = {}
    
    for key, pa in pa_dict.items():
        n_components = pa.shape[1]
        sigma_k = np.zeros(n_components)
        
        for k in range(n_components):
            # Get signs
            signs = np.sign(pa[:, k])
            
            # Find zero-crossing locations (frame indices)
            zc_locations = np.where(signs[:-1] != signs[1:])[0]
            
            if len(zc_locations) < 2:
                sigma_k[k] = np.nan
                continue
            
            # Intervals between zero-crossings (in frames)
            intervals = np.diff(zc_locations)
            
            # Convert to seconds and compute std
            intervals_sec = intervals / fs
            sigma_k[k] = np.std(intervals_sec, ddof=1)
        
        sigma_dict[key] = sigma_k
    
    return sigma_dict


def compute_mean_zc_interval(pa_dict, fs=100):
    """
    Compute mean time between zero-crossings.
    
    Complements σ by providing the average control intervention interval.
    
    Parameters
    ----------
    pa_dict : dict
        Principal accelerations
    fs : float, default=100
        Sampling frequency in Hz
        
    Returns
    -------
    mean_interval_dict : dict
        {key: array(n_components,)} with mean intervals in seconds
    """
    mean_interval_dict = {}
    
    for key, pa in pa_dict.items():
        n_components = pa.shape[1]
        mean_intervals = np.zeros(n_components)
        
        for k in range(n_components):
            signs = np.sign(pa[:, k])
            zc_locations = np.where(signs[:-1] != signs[1:])[0]
            
            if len(zc_locations) < 2:
                mean_intervals[k] = np.nan
                continue
            
            intervals = np.diff(zc_locations)
            mean_intervals[k] = np.mean(intervals) / fs
        
        mean_interval_dict[key] = mean_intervals
    
    return mean_interval_dict


def compute_pa_rms(pa_dict):
    """
    Compute RMS of principal accelerations.
    
    Quantifies the overall magnitude of postural accelerations.
    
    Parameters
    ----------
    pa_dict : dict
        Principal accelerations
        
    Returns
    -------
    rms_dict : dict
        {key: array(n_components,)} with RMS values
    """
    rms_dict = {}
    
    for key, pa in pa_dict.items():
        rms_dict[key] = np.sqrt(np.mean(pa**2, axis=0))
    
    return rms_dict


def compute_zc_per_second(pa_dict, fs=100):
    """
    Compute zero-crossings per second (rate).
    
    Normalizes N_k by trial duration.
    
    Parameters
    ----------
    pa_dict : dict
        Principal accelerations
    fs : float, default=100
        Sampling frequency in Hz
        
    Returns
    -------
    rate_dict : dict
        {key: array(n_components,)} with ZC rates in Hz
    """
    n_dict = compute_zero_crossings(pa_dict)
    rate_dict = {}
    
    for key, n_k in n_dict.items():
        trial_duration = pa_dict[key].shape[0] / fs
        rate_dict[key] = n_k / trial_duration
    
    return rate_dict