"""
Animation package for PyPMAnalyzer
Provides visualization tools for motion capture data and principal movements
"""

from .animator_2d import Animator2D
from .animator_comparison import AnimatorComparison
from .reconstruction import PMReconstructor
from .marker_sets import QualisysMarkerSet
from .styles import AnimationStyle
from .plot_utils import get_axis_limits, setup_2d_axis
from .core import SkeletonRenderer, AnimationManager

__all__ = [
    'Animator2D',
    'AnimatorComparison',
    'PMReconstructor',
    'QualisysMarkerSet',
    'AnimationStyle',
    'SkeletonRenderer',
    'AnimationManager',
    'get_axis_limits',
    'setup_2d_axis',
]