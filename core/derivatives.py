"""
Derivative computation using central differences.

Matches diff_central.m from PManalyzer.
"""

import numpy as np


def central_difference(x, fs=1.0):
    """
    Compute central difference of time series.
    
    Matches the MATLAB function diff_central.m from PManalyzer:
    x_diff = x(3:end) - x(1:end-2)
    
    Note: This loses 2 frames (one at start, one at end).
    
    Parameters
    ----------
    x : array-like
        Input time series, shape (n_frames,) or (n_frames, n_components)
    fs : float, optional
        Sampling frequency for scaling derivative. Default is 1.0.
        
    Returns
    -------
    x_diff : ndarray
        Central difference, shape (n_frames-2,) or (n_frames-2, n_components)
        Scaled by fs/2 to get proper units
        
    Examples
    --------
    >>> x = np.array([0, 1, 4, 9, 16])  # t^2
    >>> dx = central_difference(x, fs=1.0)
    >>> dx  # Should approximate 2*t: [2, 4, 6]
    array([2., 4., 6.])
    
    References
    ----------
    Haid et al. (2019) Front. Neuroinform. 13:24
    """
    x = np.asarray(x)
    
    if x.ndim == 1:
        diff = x[2:] - x[:-2]
    elif x.ndim == 2:
        diff = x[2:, :] - x[:-2, :]
    else:
        raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")
    
    # Scale by fs/2 because central difference spans 2 time steps
    return diff * (fs / 2.0)


def compute_velocities(pp_dict, fs=100):
    """
    Compute principal velocities (PV) from principal positions (PP).
    
    Parameters
    ----------
    pp_dict : dict
        Dictionary of PP time series {key: array(n_frames, n_components)}
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    pv_dict : dict
        Dictionary of PV time series {key: array(n_frames-2, n_components)}
        
    Notes
    -----
    Each PV array has 2 fewer frames than the corresponding PP array.
    """
    pv_dict = {}
    for key, pp in pp_dict.items():
        pv_dict[key] = central_difference(pp, fs=fs)
    return pv_dict


def compute_accelerations(pv_dict, fs=100):
    """
    Compute principal accelerations (PA) from principal velocities (PV).
    
    Parameters
    ----------
    pv_dict : dict
        Dictionary of PV time series {key: array(n_frames, n_components)}
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    pa_dict : dict
        Dictionary of PA time series {key: array(n_frames-2, n_components)}
        
    Notes
    -----
    Each PA array has 4 fewer frames than the original PP array
    (2 lost in PV computation, 2 more lost here).
    """
    pa_dict = {}
    for key, pv in pv_dict.items():
        pa_dict[key] = central_difference(pv, fs=fs)
    return pa_dict


def compute_all_derivatives(pp_dict, fs=100):
    """
    Compute both PV and PA from PP in one call.
    
    Parameters
    ----------
    pp_dict : dict
        Dictionary of PP time series
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    pv_dict : dict
        Principal velocities
    pa_dict : dict
        Principal accelerations
    """
    pv_dict = compute_velocities(pp_dict, fs=fs)
    pa_dict = compute_accelerations(pv_dict, fs=fs)
    return pv_dict, pa_dict