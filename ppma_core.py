"""
Python implementation of PManalyzer (Principal Movement Analysis)
Based on Federolf et al.'s MATLAB PManalyzer

Core module for preprocessing and PCA on kinematic motion capture data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA as SklearnPCA


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrialKey:
    """Identifier for a trial"""
    subject: str
    trial: str
    
    def __hash__(self):
        return hash((self.subject, self.trial))
    
    def __eq__(self, other):
        return self.subject == other.subject and self.trial == other.trial
    
    def __repr__(self):
        return f"s{self.subject}_t{self.trial}"


@dataclass
class SubjectInfo:
    """Subject metadata for normalization/weighting"""
    height: float  # in meters
    mass: float = 70.0  # in kg (default)
    gender: str = 'M'  # 'M' or 'F'


# ============================================================================
# PREPROCESSING
# ============================================================================

class PreProcessor:
    """
    Handles preprocessing of kinematic data
    Follows the structure of optionsPreProcessing.m and pca_PMAnalyzer.m
    """
    
    def __init__(
        self,
        center: bool = True,
        normalize_method: str = 'height',  # 'height', 'MED', 'max_range', 'none'
        weight_method: str = 'body_segments',  # 'body_segments', 'equal', 'none'
        order: str = 'center_weight_normalize',  # or other permutations
        filter_cutoff: float = None,  # Hz, None for no filtering
        filter_order: int = 3
    ):
        self.center = center
        self.normalize_method = normalize_method
        self.weight_method = weight_method
        self.order = order
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        
        # Storage for inverse operations
        self.mean_matrices = {}  # For centering
        self.norm_factors = {}   # For normalization
        self.weight_vectors = {} # For weighting (inverse)
        
    def fit_transform(
        self,
        data_dict: Dict[TrialKey, np.ndarray],
        subject_info: Dict[TrialKey, SubjectInfo],
        fs: float = 100.0
    ) -> Dict[TrialKey, np.ndarray]:
        """
        Preprocess all trials
        
        Args:
            data_dict: {TrialKey: (n_frames, n_coords)} - raw marker coordinates
            subject_info: {TrialKey: SubjectInfo} - subject metadata
            fs: sampling frequency in Hz
            
        Returns:
            Preprocessed data dictionary
        """
        processed = {}
        
        for key, data in data_dict.items():
            print(f"\nPreprocessing {key}...")
            info = subject_info.get(key, SubjectInfo(height=1.75))
            
            # Apply preprocessing pipeline in specified order
            data_proc = self._preprocess_trial(data, key, info, fs)
            processed[key] = data_proc
            
        return processed
    
    def _preprocess_trial(
        self,
        data: np.ndarray,
        key: TrialKey,
        info: SubjectInfo,
        fs: float
    ) -> np.ndarray:
        """Preprocess a single trial following the specified order"""
        
        # Filter first if requested (before anything else)
        if self.filter_cutoff is not None:
            data = self._apply_filter(data, fs)
            print(f"  ✓ Filtered (cutoff: {self.filter_cutoff} Hz)")
        
        # Parse order string and apply steps
        steps = self.order.lower().split('_')
        
        for step in steps:
            if step == 'center' and self.center:
                data = self._center(data, key)
                print(f"  ✓ Centered")
            elif step == 'weight':
                data = self._weight(data, key, info)
                print(f"  ✓ Weighted ({self.weight_method})")
            elif step == 'normalize':
                data = self._normalize(data, key, info)
                print(f"  ✓ Normalized ({self.normalize_method})")
        
        return data
    
    def _center(self, data: np.ndarray, key: TrialKey) -> np.ndarray:
        """Center each coordinate by subtracting mean"""
        mean_vec = np.mean(data, axis=0, keepdims=True)
        self.mean_matrices[key] = mean_vec
        return data - mean_vec
    
    def _normalize(
        self,
        data: np.ndarray,
        key: TrialKey,
        info: SubjectInfo
    ) -> np.ndarray:
        """Normalize data by specified method"""
        
        if self.normalize_method == 'none':
            self.norm_factors[key] = 1.0
            return data
        
        elif self.normalize_method == 'height':
            # Normalize by subject height
            norm_factor = info.height
            self.norm_factors[key] = norm_factor
            return data / norm_factor
        
        elif self.normalize_method == 'MED':
            # Mean Euclidean Distance (Federolf's method)
            # Compute mean distance from origin for each frame
            n_markers = data.shape[1] // 3
            data_reshaped = data.reshape(-1, n_markers, 3)
            
            # Euclidean distance for each marker at each frame
            distances = np.linalg.norm(data_reshaped, axis=2)  # (frames, markers)
            
            # Mean distance across all markers and frames
            med = np.mean(distances)
            self.norm_factors[key] = med
            return data / med
        
        elif self.normalize_method == 'max_range':
            # Normalize by maximum range across all coordinates
            max_range = np.max(data, axis=0) - np.min(data, axis=0)
            norm_factor = np.max(max_range)
            self.norm_factors[key] = norm_factor
            return data / norm_factor
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
    
    def _weight(
        self,
        data: np.ndarray,
        key: TrialKey,
        info: SubjectInfo
    ) -> np.ndarray:
        """Apply weighting by body segment masses"""
        
        if self.weight_method == 'none':
            weight_vec = np.ones(data.shape[1])
            self.weight_vectors[key] = weight_vec
            return data
        
        elif self.weight_method == 'equal':
            # Equal weighting (sqrt weighting for variance scaling)
            weight_vec = np.ones(data.shape[1])
            self.weight_vectors[key] = weight_vec
            return data * weight_vec
        
        elif self.weight_method == 'body_segments':
            # Weight by body segment masses
            # This is simplified - ideally you'd have a full marker-to-segment mapping
            n_coords = data.shape[1]
            n_markers = n_coords // 3
            
            # Default: equal weighting (you should customize this!)
            # In practice, you'd assign weights based on which segment each marker is on
            weights_per_marker = np.ones(n_markers)  # TODO: Use actual segment masses
            
            # Replicate for x, y, z
            weight_vec = np.repeat(weights_per_marker, 3)
            
            # Apply square root for variance-based weighting (as in Federolf)
            weight_vec = np.sqrt(weight_vec)
            
            self.weight_vectors[key] = 1.0 / weight_vec  # Store inverse for reconstruction
            return data * weight_vec
        
        else:
            raise ValueError(f"Unknown weighting method: {self.weight_method}")
    
    def _apply_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Apply butterworth low-pass filter"""
        nyquist = fs / 2.0
        cutoff_norm = self.filter_cutoff / nyquist
        
        # Design filter
        b, a = butter(self.filter_order, cutoff_norm, btype='low')
        
        # Apply to each coordinate (column)
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            # Handle NaNs by only filtering valid data
            valid = ~np.isnan(data[:, i])
            if np.sum(valid) > 0:
                filtered[valid, i] = filtfilt(b, a, data[valid, i])
                filtered[~valid, i] = np.nan
            else:
                filtered[:, i] = np.nan
        
        return filtered


# ============================================================================
# PCA ANALYSIS
# ============================================================================

class PrincipalMovementAnalysis:
    """
    Principal Component Analysis for movement data
    Follows the structure of pca_PMAnalyzer.m
    """
    
    def __init__(self, n_components: int = None):
        """
        Args:
            n_components: Number of PCs to keep (None = all)
        """
        self.n_components = n_components
        self.pcs_ = None  # Principal components (eigenvectors)
        self.eigenvalues_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None  # % variance explained
        self.pp_ = {}  # Principal positions (scores) per trial
        self.n_components_fitted_ = None
        
    def fit(self, data_dict: Dict[TrialKey, np.ndarray]):
        """
        Fit PCA on concatenated data from all trials
        
        Args:
            data_dict: {TrialKey: preprocessed_data}
        """
        print("\n" + "="*60)
        print("FITTING PCA")
        print("="*60)
        
        # Concatenate all trials
        data_all = []
        trial_lengths = {}
        
        for key, data in data_dict.items():
            data_all.append(data)
            trial_lengths[key] = len(data)
        
        data_concat = np.vstack(data_all)
        print(f"Total data shape: {data_concat.shape}")
        print(f"  ({len(data_dict)} trials concatenated)")
        
        # Perform SVD-based PCA (like in pca_PMAnalyzer.m)
        # Y = data / sqrt(n-1)
        n_samples = data_concat.shape[0]
        Y = data_concat / np.sqrt(n_samples - 1)
        
        # SVD: Y = U * S * V^T
        # PCs = V, eigenvalues = S^2
        U, S, VT = np.linalg.svd(Y, full_matrices=False)
        
        self.pcs_ = VT.T  # (n_features, n_components)
        self.eigenvalues_ = S ** 2
        
        # Explained variance ratio
        total_var = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = 100 * self.eigenvalues_ / total_var
        
        # Determine how many components to keep
        if self.n_components is None:
            self.n_components_fitted_ = len(self.eigenvalues_)
        else:
            self.n_components_fitted_ = min(self.n_components, len(self.eigenvalues_))
        
        # Trim to requested components
        self.pcs_ = self.pcs_[:, :self.n_components_fitted_]
        
        print(f"\nPCs computed: {self.n_components_fitted_}")
        print(f"\nVariance explained by first 8 PMs:")
        for i in range(min(8, self.n_components_fitted_)):
            print(f"  PM{i+1}: {self.explained_variance_ratio_[i]:.2f}%")
        
        cumsum = np.cumsum(self.explained_variance_ratio_[:self.n_components_fitted_])
        print(f"\nCumulative variance: {cumsum[-1]:.2f}%")
        
        # Compute scores (PP) for each trial
        print("\nComputing principal positions (PP) for each trial...")
        for key, data in data_dict.items():
            self.pp_[key] = data @ self.pcs_
            print(f"  {key}: PP shape = {self.pp_[key].shape}")
        
        return self
    
    def compute_pv(self, fs: float = 100.0) -> Dict[TrialKey, np.ndarray]:
        """
        Compute principal velocities (1st derivative of PP)
        
        Args:
            fs: Sampling frequency in Hz
            
        Returns:
            Dictionary of PV time series
        """
        pv_dict = {}
        
        for key, pp in self.pp_.items():
            # Central difference (like diff_central.m)
            pv = (pp[2:, :] - pp[:-2, :]) * fs / 2.0
            
            # Pad to maintain length (duplicate first/last)
            pv = np.vstack([pv[0:1, :], pv, pv[-1:, :]])
            
            pv_dict[key] = pv
        
        return pv_dict
    
    def compute_pa(self, fs: float = 100.0) -> Dict[TrialKey, np.ndarray]:
        """
        Compute principal accelerations (2nd derivative of PP)
        
        Args:
            fs: Sampling frequency in Hz
            
        Returns:
            Dictionary of PA time series
        """
        pv_dict = self.compute_pv(fs)
        pa_dict = {}
        
        for key, pv in pv_dict.items():
            # Central difference again
            pa = (pv[2:, :] - pv[:-2, :]) * fs / 2.0
            
            # Pad to maintain length
            pa = np.vstack([pa[0:1, :], pa, pa[-1:, :]])
            
            pa_dict[key] = pa
        
        return pa_dict


# ============================================================================
# METRICS / VARIABLES
# ============================================================================

def compute_rVAR(pp_dict: Dict[TrialKey, np.ndarray]) -> Dict[TrialKey, np.ndarray]:
    """
    Compute relative variance (rVAR) for each trial
    Following optionsVariablesComp.m logic
    
    rVAR_k = VAR(PP_k) / sum_i(VAR(PP_i)) * 100
    
    Returns:
        Dictionary mapping TrialKey to rVAR array (one value per PM)
    """
    rvar_dict = {}
    
    for key, pp in pp_dict.items():
        # Variance of each PM (column)
        var_per_pm = np.var(pp, axis=0)
        
        # Total variance
        total_var = np.sum(var_per_pm)
        
        # Relative variance (%)
        rvar = 100 * var_per_pm / total_var
        
        rvar_dict[key] = rvar
    
    return rvar_dict


def compute_zero_crossings(
    time_series: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Compute number of zero crossings (NoZC) for each PM
    Following optionsVariablesComp.m
    
    This counts how many times the control system changes direction!
    
    Args:
        time_series: (n_frames, n_pms) array
        axis: Axis along which to count (0 = time)
        
    Returns:
        Array of NoZC for each PM
    """
    # Sign of time series
    signs = np.sign(time_series)
    
    # Find where sign changes
    sign_changes = np.diff(signs, axis=axis) != 0
    
    # Count changes for each PM
    nozc = np.sum(sign_changes, axis=axis)
    
    return nozc


def compute_zc_per_second(
    time_series: np.ndarray,
    fs: float,
    axis: int = 0
) -> np.ndarray:
    """Zero crossings per second (NoZCps)"""
    nozc = compute_zero_crossings(time_series, axis)
    duration = time_series.shape[axis] / fs
    return nozc / duration


def compute_pv_std(pv_dict: Dict[TrialKey, np.ndarray]) -> Dict[TrialKey, np.ndarray]:
    """
    Compute standard deviation of principal velocities
    Quantifies movement speed variability
    """
    pv_std_dict = {}
    
    for key, pv in pv_dict.items():
        pv_std_dict[key] = np.std(pv, axis=0)
    
    return pv_std_dict


def compute_rms(time_series_dict: Dict[TrialKey, np.ndarray]) -> Dict[TrialKey, np.ndarray]:
    """Compute RMS (root mean square) for each PM"""
    rms_dict = {}
    
    for key, ts in time_series_dict.items():
        rms_dict[key] = np.sqrt(np.mean(ts ** 2, axis=0))
    
    return rms_dict