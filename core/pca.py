"""
Principal Component Analysis for kinematic data.

Implements the PCA-based movement analysis approach from:
- Haid et al. (2019) Front. Neuroinform. 13:24
- Federolf (2016) J. Biomech. 49(3):364-370
- Federolf et al. (2013) J. Biomech. 46(15):2626-2633
"""

import numpy as np
from sklearn.decomposition import PCA
from .derivatives import compute_all_derivatives


class PrincipalMovementAnalysis:
    """
    Principal Component Analysis for whole-body kinematic data.
    
    This class implements the PCA-based approach for analyzing
    postural movements, extracting principal movements (PMs) from
    kinematic data.
    
    The key idea is to perform PCA on concatenated data from multiple
    trials/subjects to obtain a shared coordinate system (posture space),
    then project individual trials onto this space to get comparable
    movement time series.
    
    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. If None, keep all components.
        
    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        Fitted PCA object
    preprocessor_ : Preprocessor or None
        Preprocessor used (if provided during fitting)
    trial_info_ : list
        List of (key, n_frames) for each trial in concatenation order
    pp_ : dict
        Principal positions {key: array(n_frames, n_components)}
    pv_ : dict
        Principal velocities (computed on demand)
    pa_ : dict
        Principal accelerations (computed on demand)
        
    Examples
    --------
    >>> from pypmanalyzer.core import Preprocessor, PrincipalMovementAnalysis
    >>> 
    >>> # Prepare data
    >>> data_dict = {
    ...     (1, 1): markers_subject1_trial1,
    ...     (1, 2): markers_subject1_trial2,
    ...     (2, 1): markers_subject2_trial1,
    ... }
    >>> 
    >>> # Preprocess
    >>> prep = Preprocessor(filter_cutoff=7.0)
    >>> data_processed = prep.fit_transform(data_dict, subject_info, fs=100)
    >>> 
    >>> # Fit PCA
    >>> pma = PrincipalMovementAnalysis()
    >>> pma.fit(data_processed)
    >>> 
    >>> # Get results
    >>> pp = pma.pp_  # Principal positions
    >>> eigenvalues = pma.explained_variance_ratio_  # rEV_k
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca_ = None
        self.preprocessor_ = None
        self.trial_info_ = []
        self.pp_ = {}
        self.pv_ = None
        self.pa_ = None
        
    def fit(self, data_dict):
        """
        Fit PCA on concatenated data from multiple trials.
        
        This creates a shared coordinate system (posture space) that
        applies to all trials.
        
        Parameters
        ----------
        data_dict : dict
            Preprocessed kinematic data {key: array(n_frames, n_markers*3)}
            
        Returns
        -------
        self : PrincipalMovementAnalysis
            Fitted instance
        """
        # Concatenate all trials vertically
        data_concat, trial_info = self._concatenate_trials(data_dict)
        
        # Fit PCA using sklearn
        # Note: sklearn's PCA does the Y = X/sqrt(n-1) scaling internally
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(data_concat)
        
        # Store trial information
        self.trial_info_ = trial_info
        
        # Compute scores (PP) for each trial
        self._compute_pp(data_dict)
        
        return self
    
    def _concatenate_trials(self, data_dict):
        """
        Concatenate trials vertically.
        
        Returns
        -------
        data_concat : ndarray
            Concatenated data (sum of all n_frames, n_markers*3)
        trial_info : list
            List of (key, n_frames) in concatenation order
        """
        data_list = []
        trial_info = []
        
        for key in sorted(data_dict.keys()):
            data = data_dict[key]
            data_list.append(data)
            trial_info.append((key, data.shape[0]))
        
        data_concat = np.vstack(data_list)
        return data_concat, trial_info
    
    def _compute_pp(self, data_dict):
        """
        Compute principal positions (scores) for each trial.
        
        Matches the score computation in pca_PMAnalyzer.m:
        Scores_TrialSubj{TrialNow,SubjNow} = Data_all_for_scores{TrialNow,SubjNow}*PCs;
        """
        self.pp_ = {}
        for key, data in data_dict.items():
            self.pp_[key] = self.pca_.transform(data)
    
    def compute_pv(self, fs=100):
        """
        Compute principal velocities (first derivative of PP).
        
        Parameters
        ----------
        fs : float, default=100
            Sampling frequency in Hz
            
        Returns
        -------
        pv_dict : dict
            Principal velocities {key: array(n_frames-2, n_components)}
        """
        if self.pp_ is None or len(self.pp_) == 0:
            raise ValueError("Must fit PCA and compute PP first")
        
        from .derivatives import compute_velocities
        self.pv_ = compute_velocities(self.pp_, fs=fs)
        return self.pv_
    
    def compute_pa(self, fs=100):
        """
        Compute principal accelerations (second derivative of PP).
        
        Parameters
        ----------
        fs : float, default=100
            Sampling frequency in Hz
            
        Returns
        -------
        pa_dict : dict
            Principal accelerations {key: array(n_frames-4, n_components)}
        """
        if self.pv_ is None:
            self.compute_pv(fs=fs)
        
        from .derivatives import compute_accelerations
        self.pa_ = compute_accelerations(self.pv_, fs=fs)
        return self.pa_
    
    def compute_all(self, fs=100):
        """
        Compute PP, PV, and PA in one call.
        
        Returns
        -------
        pp : dict
        pv : dict
        pa : dict
        """
        if self.pv_ is None:
            self.compute_pv(fs=fs)
        if self.pa_ is None:
            self.compute_pa(fs=fs)
        return self.pp_, self.pv_, self.pa_
    
    # Convenience properties to match sklearn and paper notation
    @property
    def components_(self):
        """PC eigenvectors (principal movements)."""
        return self.pca_.components_
    
    @property
    def explained_variance_(self):
        """Eigenvalues (absolute variance explained)."""
        return self.pca_.explained_variance_
    
    @property
    def explained_variance_ratio_(self):
        """Relative eigenvalues (rEV_k) as percentages."""
        return self.pca_.explained_variance_ratio_ * 100
    
    @property
    def n_components_fitted_(self):
        """Number of components in fitted model."""
        return self.pca_.n_components_