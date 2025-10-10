"""
Kinematic metrics derived from principal movements.

Implements metrics from:
- Haid et al. (2019) Front. Neuroinform. 13:24
- Federolf et al. (2013) J. Biomech. 46(15):2626-2633
"""

import numpy as np


def compute_rVAR(pp_dict):
    """
    Compute relative variance (rVAR) for each component in each trial.
    
    rVAR quantifies the relative contribution of each PM to the
    total variance in a trial's movement.
    
    Matches the rVARk computation from PManalyzer:
    rVAR_k = VAR(PP_k) / sum(VAR(PP_j)) * 100
    
    Parameters
    ----------
    pp_dict : dict
        Principal positions {key: array(n_frames, n_components)}
        
    Returns
    -------
    rvar_dict : dict
        {key: array(n_components,)} with rVAR values in percent
        
    Examples
    --------
    >>> pp = {(1, 1): np.random.randn(100, 10)}
    >>> rvar = compute_rVAR(pp)
    >>> np.sum(rvar[(1, 1)])  # Should sum to 100
    100.0
    
    References
    ----------
    Federolf et al. (2013) J. Biomech. 46(15):2626-2633
    """
    rvar_dict = {}
    
    for key, pp in pp_dict.items():
        # Variance of each component
        var_k = np.var(pp, axis=0, ddof=1)  # Use ddof=1 for sample variance
        
        # Total variance
        total_var = np.sum(var_k)
        
        # Relative variance in percent
        rvar_dict[key] = (var_k / total_var) * 100
    
    return rvar_dict


def compute_rSTD(pp_dict):
    """
    Compute relative standard deviations (rSTD).
    
    Similar to rVAR but uses standard deviations instead of variances,
    providing a measure that scales with the original movement amplitude.
    
    Parameters
    ----------
    pp_dict : dict
        Principal positions {key: array(n_frames, n_components)}
        
    Returns
    -------
    rstd_dict : dict
        {key: array(n_components,)} with rSTD values in percent
        
    References
    ----------
    Haid & Federolf (2018) Entropy 20(1):30
    """
    rstd_dict = {}
    
    for key, pp in pp_dict.items():
        # Standard deviation of each component
        std_k = np.std(pp, axis=0, ddof=1)
        
        # Total standard deviation
        total_std = np.sum(std_k)
        
        # Relative standard deviation in percent
        rstd_dict[key] = (std_k / total_std) * 100
    
    return rstd_dict


def compute_cumulative_variance(rvar_dict):
    """
    Compute cumulative relative variance.
    
    CUM_rVAR_k = sum of rVAR_1 to rVAR_k
    
    Parameters
    ----------
    rvar_dict : dict
        Output from compute_rVAR
        
    Returns
    -------
    cum_rvar_dict : dict
        {key: array(n_components,)} with cumulative rVAR
    """
    cum_rvar_dict = {}
    
    for key, rvar in rvar_dict.items():
        cum_rvar_dict[key] = np.cumsum(rvar)
    
    return cum_rvar_dict


def compute_residual_variance(rvar_dict, threshold_component=7):
    """
    Compute residual variance after threshold component.
    
    RV = 100 - CUM_rVAR_m
    
    Quantifies the variance not explained by the first m components,
    serving as a measure of movement complexity.
    
    Parameters
    ----------
    rvar_dict : dict
        Output from compute_rVAR
    threshold_component : int, default=7
        Component threshold (m)
        
    Returns
    -------
    rv_dict : dict
        {key: float} residual variance for each trial
        
    References
    ----------
    Zago et al. (2017) Hum. Mov. Sci. 54:144-153
    """
    rv_dict = {}
    
    for key, rvar in rvar_dict.items():
        if len(rvar) < threshold_component:
            cum_var = np.sum(rvar)
        else:
            cum_var = np.sum(rvar[:threshold_component])
        
        rv_dict[key] = 100 - cum_var
    
    return rv_dict


def compute_pv_std(pv_dict):
    """
    Compute standard deviation of principal velocities.
    
    PV-STD quantifies the amplitude of postural movement speed.
    Often used to detect differences between conditions.
    
    Parameters
    ----------
    pv_dict : dict
        Principal velocities {key: array(n_frames, n_components)}
        
    Returns
    -------
    pv_std_dict : dict
        {key: array(n_components,)} with PV-STD for each component
        
    References
    ----------
    Haid et al. (2018) Front. Aging Neurosci. 10:22
    """
    pv_std_dict = {}
    
    for key, pv in pv_dict.items():
        pv_std_dict[key] = np.std(pv, axis=0, ddof=1)
    
    return pv_std_dict


def compute_pp_range(pp_dict):
    """
    Compute range (max - min) of principal positions.
    
    Parameters
    ----------
    pp_dict : dict
        Principal positions
        
    Returns
    -------
    range_dict : dict
        {key: array(n_components,)} with ranges
    """
    range_dict = {}
    
    for key, pp in pp_dict.items():
        range_dict[key] = np.ptp(pp, axis=0)  # ptp = peak-to-peak
    
    return range_dict


def compute_rms(time_series_dict):
    """
    Compute root mean square of time series.
    
    RMS = sqrt(mean(x^2))
    
    Parameters
    ----------
    time_series_dict : dict
        Any time series (PP, PV, or PA)
        
    Returns
    -------
    rms_dict : dict
        {key: array(n_components,)} with RMS values
    """
    rms_dict = {}
    
    for key, ts in time_series_dict.items():
        rms_dict[key] = np.sqrt(np.mean(ts**2, axis=0))
    
    return rms_dict