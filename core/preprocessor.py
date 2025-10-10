"""
Data preprocessing for kinematic analysis.

Implements centering, weighting, normalization, and filtering
following Haid et al. (2019) Front. Neuroinform. 13:24
"""

import numpy as np
from scipy import signal
import warnings


class Preprocessor:
    """
    Preprocess kinematic data for PCA.
    
    Implements the preprocessing pipeline from PManalyzer, including:
    - Centering (remove mean position)
    - Weighting (by segment masses)
    - Normalization (by height or mean Euclidean distance)
    - Filtering (low-pass Butterworth)
    
    The order of operations is configurable to match different
    normalization strategies.
    
    Parameters
    ----------
    center : bool, default=True
        Whether to center data (subtract mean)
    weight_method : str or None, default=None
        Weighting method. Options:
        - None: no weighting
        - 'standard_male': standard male body mass distribution
        - 'standard_female': standard female body mass distribution
        - dict: custom weights {marker_index: weight}
    normalize_method : str or None, default='height'
        Normalization method. Options:
        - None: no normalization
        - 'height': normalize by body height
        - 'MED': mean Euclidean distance
    order : str, default='center_weight_normalize'
        Order of preprocessing operations. Options:
        - 'center_weight_normalize'
        - 'center_normalize_weight'
        - 'normalize_center_weight' (required for height normalization)
    filter_cutoff : float or None, default=None
        Low-pass filter cutoff frequency in Hz. If None, no filtering.
    filter_order : int, default=3
        Butterworth filter order
        
    Attributes
    ----------
    mean_ : dict
        Mean values for each trial {trial_key: array}
    norm_factor_ : dict
        Normalization factors {trial_key: float}
    weights_ : dict
        Weights for each trial {trial_key: array}
        
    Examples
    --------
    >>> prep = Preprocessor(
    ...     center=True,
    ...     weight_method='standard_male',
    ...     normalize_method='height',
    ...     order='normalize_center_weight',
    ...     filter_cutoff=7.0
    ... )
    >>> data_dict = {(1, 1): marker_data}  # {(subject, trial): array}
    >>> subject_info = {(1, 1): {'height': 1.75, 'gender': 'male'}}
    >>> processed = prep.fit_transform(data_dict, subject_info)
    """
    
    def __init__(self,
                 center=True,
                 weight_method=None,
                 normalize_method='height',
                 order='center_weight_normalize',
                 filter_cutoff=None,
                 filter_order=3):
        
        self.center = center
        self.weight_method = weight_method
        self.normalize_method = normalize_method
        self.order = order
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        
        # Storage for preprocessing parameters
        self.mean_ = {}
        self.norm_factor_ = {}
        self.weights_ = {}
        
        # Validate order
        valid_orders = [
            'center_weight_normalize',
            'center_normalize_weight',
            'normalize_center_weight',
            'normalize_weight_center',
            'weight_center_normalize',
            'weight_normalize_center'
        ]
        if order not in valid_orders:
            raise ValueError(f"order must be one of {valid_orders}")
    
    def fit_transform(self, data_dict, subject_info=None, fs=100):
        """
        Preprocess all trials.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary of kinematic data {key: array(n_frames, n_markers*3)}
            where key is typically (subject_id, trial_id)
        subject_info : dict, optional
            Subject information {key: {'height': float, 'gender': str}}
            Required if using height normalization or gender-specific weights
        fs : float, default=100
            Sampling frequency in Hz (for filtering)
            
        Returns
        -------
        processed_dict : dict
            Preprocessed data with same keys as input
        """
        if subject_info is None:
            subject_info = {}
        
        processed_dict = {}
        
        for key, data in data_dict.items():
            info = subject_info.get(key, {})
            processed = self._process_single_trial(data, info, fs, key)
            processed_dict[key] = processed
            
        return processed_dict
    
    def _process_single_trial(self, data, info, fs, key):
        """Process a single trial through the preprocessing pipeline."""
        data = np.asarray(data, dtype=float)
        
        # Define the three operations
        ops = {
            'center': (self._center_data, {}),
            'weight': (self._weight_data, {'info': info}),
            'normalize': (self._normalize_data, {'info': info})
        }
        
        # Parse order string
        order_list = self.order.split('_')
        
        # Apply operations in order
        for op_name in order_list:
            if op_name == 'center' and self.center:
                func, kwargs = ops['center']
                data, mean = func(data)
                self.mean_[key] = mean
            elif op_name == 'weight' and self.weight_method is not None:
                func, kwargs = ops['weight']
                data, weights = func(data, **kwargs)
                self.weights_[key] = weights
            elif op_name == 'normalize' and self.normalize_method is not None:
                func, kwargs = ops['normalize']
                data, norm_factor = func(data, **kwargs)
                self.norm_factor_[key] = norm_factor
        
        # Apply filtering last (always after other preprocessing)
        if self.filter_cutoff is not None:
            data = self._filter_data(data, fs)
        
        return data
    
    def _center_data(self, data):
        """
        Center data by subtracting mean of each column.
        
        Matches center_data function in pca_PMAnalyzer.m
        """
        mean = np.mean(data, axis=0)
        data_centered = data - mean
        return data_centered, mean
    
    def _weight_data(self, data, info):
        """
        Apply segment mass weighting.
        
        Parameters
        ----------
        data : ndarray
            Shape (n_frames, n_markers*3)
        info : dict
            Must contain 'gender' if using standard weights
            
        Returns
        -------
        data_weighted : ndarray
            Weighted data
        weights : ndarray
            Applied weights (for inverse transform)
        """
        n_markers = data.shape[1] // 3
        
        if isinstance(self.weight_method, dict):
            # Custom weights provided
            weights = self._get_custom_weights(n_markers)
        elif self.weight_method in ['standard_male', 'standard_female']:
            # Standard body mass distribution
            gender = info.get('gender', 'male')
            weights = self._get_standard_weights(n_markers, gender)
        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method}")
        
        # Apply weights (repeat for x, y, z)
        weights_full = np.repeat(weights, 3)
        data_weighted = data * weights_full
        
        return data_weighted, weights_full
    
    def _normalize_data(self, data, info):
        """
        Normalize data by height or MED.
        
        Parameters
        ----------
        data : ndarray
            Shape (n_frames, n_markers*3)
        info : dict
            Must contain 'height' if using height normalization
            
        Returns
        -------
        data_normalized : ndarray
            Normalized data
        norm_factor : float
            Normalization factor
        """
        if self.normalize_method == 'height':
            if 'height' not in info:
                raise ValueError("Subject height required for height normalization")
            norm_factor = info['height']
            
        elif self.normalize_method == 'MED':
            # Mean Euclidean distance
            norm_factor = self._compute_MED(data)
            
        else:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")
        
        data_normalized = data / norm_factor
        return data_normalized, norm_factor
    
    def _filter_data(self, data, fs):
        """
        Apply zero-phase Butterworth low-pass filter.
        
        Uses filtfilt for zero-phase filtering (matches MATLAB).
        """
        nyquist = fs / 2.0
        cutoff_normalized = self.filter_cutoff / nyquist
        
        if cutoff_normalized >= 1.0:
            warnings.warn(
                f"Filter cutoff ({self.filter_cutoff} Hz) >= Nyquist "
                f"frequency ({nyquist} Hz). No filtering applied."
            )
            return data
        
        # Design Butterworth filter
        b, a = signal.butter(self.filter_order, cutoff_normalized, btype='low')
        
        # Apply zero-phase filter to each column
        data_filtered = signal.filtfilt(b, a, data, axis=0)
        
        return data_filtered
    
    def _compute_MED(self, data):
        """
        Compute Mean Euclidean Distance.
        
        MED = mean distance of all markers from the mean position
        at each time point.
        """
        n_markers = data.shape[1] // 3
        
        # Reshape to (n_frames, n_markers, 3)
        data_3d = data.reshape(-1, n_markers, 3)
        
        # Compute mean position at each frame
        mean_pos = np.mean(data_3d, axis=1, keepdims=True)  # (n_frames, 1, 3)
        
        # Euclidean distances from mean
        distances = np.linalg.norm(data_3d - mean_pos, axis=2)  # (n_frames, n_markers)
        
        # Mean over all frames and markers
        med = np.mean(distances)
        
        return med
    
    def _get_standard_weights(self, n_markers, gender):
        """
        Get standard segment mass weights.
        
        Based on de Leva (1996) and Defense Technical Info Center (1988).
        
        Note: This is a simplified version. In practice, you need to
        know which markers correspond to which body segments.
        """
        # This is a placeholder - in real implementation, you'd need
        # a mapping from markers to segments
        warnings.warn(
            "Standard weighting requires marker-to-segment mapping. "
            "Using uniform weights as placeholder."
        )
        weights = np.ones(n_markers)
        return weights
    
    def _get_custom_weights(self, n_markers):
        """Get custom weights from user-provided dictionary."""
        weights = np.ones(n_markers)
        for marker_idx, weight in self.weight_method.items():
            if marker_idx < n_markers:
                weights[marker_idx] = weight
        return weights