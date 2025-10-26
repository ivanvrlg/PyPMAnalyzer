"""
Marker set definitions for motion capture systems
"""

from typing import List


class QualisysMarkerSet:
    """
    53-marker Qualisys marker set with colors and connections
    """
    
    # Marker colors (Left=Blue, Right=Green, Center=Orange)
    MARKER_COLORS = [
        "#00CCFF", "#009300", "#FF9D3B", "#009300", "#009300", "#009300", 
        "#009300", "#C0FF57", "#009300", "#009300", "#00CCFF", "#00CCFF",
        "#00CCFF", "#00CCFF", "#A0C0FF", "#00CCFF", "#00CCFF", "#FF9D3B",
        "#FF9D3B", "#FF9D3B", "#FF9D3B", "#FF9D3B", "#FF9D3B", "#FF9D3B",
        "#FF9D3B", "#009300", "#009300", "#009300", "#00CCFF", "#00CCFF",
        "#00CCFF", "#009300", "#009300", "#009300", "#009300", "#009300",
        "#009300", "#009300", "#009300", "#009300", "#009300", "#009300",
        "#00CCFF", "#00CCFF", "#00CCFF", "#00CCFF", "#00CCFF", "#00CCFF",
        "#00CCFF", "#00CCFF", "#00CCFF", "#00CCFF", "#00CCFF"
    ]
    
    # Skeleton connections (indices only, colors applied separately)
    BONE_CONNECTIONS = [
        # Head
        [0, 1], [0, 2], [1, 2],
        # Neck to shoulders
        [2, 19], [19, 17], [17, 3], [17, 10],
        # Spine
        [19, 20, 21, 22, 23, 24],
        # Torso
        [17, 18], [18, 24],
        # Right arm
        [3, 4], [4, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 9],
        # Left arm
        [10, 11], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 16],
        # Pelvis - Right
        [24, 25], [24, 26], [24, 27], [25, 26], [26, 27], [27, 25],
        # Pelvis - Left
        [24, 28], [24, 29], [24, 30], [28, 29], [29, 30], [30, 28],
        # Right leg - thigh
        [27, 31], [31, 32], [31, 33], [32, 34], [33, 34],
        # Right leg - shank
        [34, 35], [35, 36], [35, 37],
        # Right foot
        [36, 38], [37, 38], [38, 39], [38, 40], [38, 41], [39, 40], [40, 41],
        # Left leg - thigh
        [30, 42], [42, 43], [42, 44], [43, 45], [44, 45],
        # Left leg - shank
        [45, 46], [46, 47], [46, 48],
        # Left foot
        [47, 49], [48, 49], [49, 50], [49, 51], [49, 52], [50, 51], [51, 52],
    ]