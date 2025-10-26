"""
Visual styles and color schemes for animations
"""

from typing import List
from .marker_sets import QualisysMarkerSet


class AnimationStyle:
    """
    Manages colors and visual styles for animations
    """
    
    @staticmethod
    def get_marker_colors(n_markers: int, style: str) -> List[str]:
        """
        Get marker colors based on style and actual number of markers
        
        Args:
            n_markers: Number of markers in the data
            style: 'qtm' (colored by side) or 'black' (all black)
            
        Returns:
            List of color strings
        """
        if style == 'qtm':
            # Use Qualisys colors, extend or truncate as needed
            base_colors = QualisysMarkerSet.MARKER_COLORS
            if n_markers <= len(base_colors):
                return base_colors[:n_markers]
            else:
                # Extend by repeating gray for extra markers
                return base_colors + ['#808080'] * (n_markers - len(base_colors))
        else:  # 'black'
            return ['black'] * n_markers
    
    @staticmethod
    def get_bone_color(style: str) -> str:
        """
        Get bone/skeleton color based on style
        
        Args:
            style: 'qtm' or 'black'
            
        Returns:
            Color string
        """
        return 'yellow' if style == 'qtm' else 'black'
    
    @staticmethod
    def get_marker_edge_color(style: str) -> str:
        """Get marker edge color"""
        return 'black' if style == 'qtm' else 'white'