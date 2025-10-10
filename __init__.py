"""
PyPMAnalyzer: Python implementation of Principal Movement Analysis

Python implementation of methods from:
- Haid et al. (2019) Front. Neuroinform. 13:24
- Federolf (2016) J. Biomech. 49(3):364-370
- Federolf et al. (2013) J. Biomech. 46(15):2626-2633

Original MATLAB implementation: PManalyzer
https://www.uibk.ac.at/isw/forschung/neurophysiologie-bewegungswissenschaft/software.html
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Core functionality
from .core.pca import PrincipalMovementAnalysis
from .core.preprocessor import Preprocessor
from .core.derivatives import (
    central_difference,
    compute_velocities,
    compute_accelerations,
    compute_all_derivatives
)

# I/O
from .io.readers import (
    read_csv,
    read_qualisys_tsv,
    load_directory,
    parse_subject_info_from_filename
)

# Metrics
from .metrics.kinematic import (
    compute_rVAR,
    compute_rSTD,
    compute_cumulative_variance,
    compute_residual_variance,
    compute_pv_std,
    compute_pp_range,
    compute_rms
)

from .metrics.control import (
    compute_zero_crossings,
    compute_zc_variability,
    compute_mean_zc_interval,
    compute_pa_rms,
    compute_zc_per_second
)

__all__ = [
    # Core
    'PrincipalMovementAnalysis',
    'Preprocessor',
    'central_difference',
    'compute_velocities',
    'compute_accelerations',
    'compute_all_derivatives',
    # I/O
    'read_csv',
    'read_qualisys_tsv',
    'read_c3d',
    'read_mat',
    'load_directory',
    'parse_subject_info_from_filename',
    # Kinematic metrics
    'compute_rVAR',
    'compute_rSTD',
    'compute_cumulative_variance',
    'compute_residual_variance',
    'compute_pv_std',
    'compute_pp_range',
    'compute_rms',
    # Control metrics
    'compute_zero_crossings',
    'compute_zc_variability',
    'compute_mean_zc_interval',
    'compute_pa_rms',
    'compute_zc_per_second',
]