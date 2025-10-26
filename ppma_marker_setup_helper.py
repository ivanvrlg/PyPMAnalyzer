"""
Marker Setup Helper
Tool to help define marker connectivity for your specific marker set
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qualisys_loader import QualisysLoader


# ============================================================================
# MARKER CONNECTION TEMPLATES
# ============================================================================

# Common marker set templates
# Customize these based on YOUR marker placement!

FULL_BODY_TEMPLATE = {
    'name': 'Full Body (53 markers - customize!)',
    'connections': [
        # HEAD (markers 0-3)
        ([0, 1, 2, 3, 0], 'darkblue', 2.0),
        
        # TRUNK (markers 4-10)
        ([4, 5, 6, 7, 4], 'navy', 2.5),  # Upper trunk
        ([7, 8, 9, 10, 7], 'navy', 2.5),  # Lower trunk
        
        # RIGHT ARM (markers 11-20)
        ([5, 11], 'red', 2.0),  # Shoulder
        ([11, 12, 13], 'red', 2.0),  # Upper arm
        ([13, 14, 15], 'darkred', 2.0),  # Forearm
        ([15, 16, 17, 18, 19, 20], 'darkred', 1.5),  # Hand
        
        # LEFT ARM (markers 21-30)
        ([6, 21], 'blue', 2.0),  # Shoulder
        ([21, 22, 23], 'blue', 2.0),  # Upper arm
        ([23, 24, 25], 'darkblue', 2.0),  # Forearm
        ([25, 26, 27, 28, 29, 30], 'darkblue', 1.5),  # Hand
        
        # PELVIS (markers 31-34)
        ([31, 32, 33, 34, 31], 'darkgreen', 2.5),
        
        # RIGHT LEG (markers 35-42)
        ([32, 35], 'red', 2.5),  # Hip
        ([35, 36, 37], 'red', 2.5),  # Thigh
        ([37, 38, 39], 'darkred', 2.5),  # Shank
        ([39, 40, 41, 42], 'darkred', 2.0),  # Foot
        
        # LEFT LEG (markers 43-50)
        ([34, 43], 'blue', 2.5),  # Hip
        ([43, 44, 45], 'blue', 2.5),  # Thigh
        ([45, 46, 47], 'darkblue', 2.5),  # Shank
        ([47, 48, 49, 50], 'darkblue', 2.0),  # Foot
        
        # SPINE CONNECTION
        ([7, 31], 'darkgreen', 2.0),
    ],
    'marker_names': [
        # You'll need to fill this with your actual marker names!
        'HEAD_1', 'HEAD_2', 'HEAD_3', 'HEAD_4',
        # ... etc
    ]
}


SIMPLIFIED_TEMPLATE = {
    'name': 'Simplified Body',
    'connections': [
        # Just major segments
        ([0, 1, 2], 'darkblue', 3.0),  # Head-spine
        ([2, 3, 4, 2], 'navy', 2.5),  # Shoulders
        ([3, 5, 6, 7], 'red', 2.0),  # Right arm
        ([4, 8, 9, 10], 'blue', 2.0),  # Left arm
        ([2, 11], 'darkgreen', 2.0),  # Spine
        ([11, 12, 13, 11], 'darkgreen', 2.5),  # Pelvis
        ([12, 14, 15, 16], 'darkred', 2.5),  # Right leg
        ([13, 17, 18, 19], 'darkblue', 2.5),  # Left leg
    ]
}


# ============================================================================
# INTERACTIVE MARKER IDENTIFIER
# ============================================================================

class MarkerSetupTool:
    """Interactive tool to identify markers and create connections"""
    
    def __init__(self, session_path: str, trial_name: str):
        """
        Args:
            session_path: Path to Qualisys session
            trial_name: Trial to use for marker identification
        """
        self.session_path = session_path
        self.trial_name = trial_name
        
        # Load data
        print(f"Loading {trial_name}...")
        loader = QualisysLoader(session_path, trial_name)
        loader.load_marker_file()
        
        self.marker_data = loader.get_marker_coordinates()
        self.marker_names = loader.get_marker_names()
        self.n_markers = len(self.marker_names)
        
        print(f"Loaded {self.n_markers} markers")
    
    def plot_marker_positions(self, frame_idx: int = 0):
        """
        Plot 3D marker positions with labels
        Helps identify which markers are which
        """
        frame_data = self.marker_data[frame_idx]
        
        # Extract x, y, z
        x = frame_data[0::3]
        y = frame_data[1::3]
        z = frame_data[2::3]
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot markers
        scatter = ax.scatter(x, y, z, c=range(self.n_markers), 
                           cmap='tab20', s=100, alpha=0.8,
                           edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, (xi, yi, zi, name) in enumerate(zip(x, y, z, self.marker_names)):
            ax.text(xi, yi, zi, f'  {i}: {name}', 
                   fontsize=8, ha='left', va='bottom')
        
        ax.set_xlabel('X (AP)', fontsize=12)
        ax.set_ylabel('Y (ML)', fontsize=12)
        ax.set_zlabel('Z (Vertical)', fontsize=12)
        ax.set_title(f'Marker Positions - Frame {frame_idx}\n'
                    f'Use marker indices to define connections',
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Marker Index', fontsize=11)
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), 
                             y.max()-y.min(), 
                             z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def print_marker_list(self):
        """Print all marker names with indices"""
        print("\n" + "="*70)
        print("MARKER LIST")
        print("="*70)
        
        for i, name in enumerate(self.marker_names):
            print(f"  {i:2d}: {name}")
        
        print("\nUse these indices to define connections!")
    
    def test_connections(
        self,
        connections: list,
        frame_idx: int = 0,
        view_3d: bool = True
    ):
        """
        Test marker connections by plotting them
        
        Args:
            connections: List of ([indices], color, width)
            frame_idx: Which frame to plot
            view_3d: Use 3D view (else 2D sagittal/frontal)
        """
        frame_data = self.marker_data[frame_idx]
        
        # Extract coordinates
        x = frame_data[0::3]
        y = frame_data[1::3]
        z = frame_data[2::3]
        
        if view_3d:
            # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot connections
            for indices, color, width in connections:
                if isinstance(indices, int):
                    indices = [indices]
                
                for i in range(len(indices)-1):
                    idx1, idx2 = indices[i], indices[i+1]
                    ax.plot([x[idx1], x[idx2]],
                           [y[idx1], y[idx2]],
                           [z[idx1], z[idx2]],
                           color=color, linewidth=width)
                
                # Plot markers
                ax.scatter(x[indices], y[indices], z[indices],
                          c=color, s=50, edgecolors='black', alpha=0.8)
            
            ax.set_xlabel('X (AP)')
            ax.set_ylabel('Y (ML)')
            ax.set_zlabel('Z (Vertical)')
            ax.set_title('Testing Marker Connections - 3D View', 
                        fontsize=14, fontweight='bold')
            
        else:
            # 2D views
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Sagittal view (x-z)
            for indices, color, width in connections:
                if isinstance(indices, int):
                    indices = [indices]
                
                for i in range(len(indices)-1):
                    idx1, idx2 = indices[i], indices[i+1]
                    ax1.plot([x[idx1], x[idx2]],
                            [z[idx1], z[idx2]],
                            color=color, linewidth=width, marker='o',
                            markersize=6, markerfacecolor=color,
                            markeredgecolor='black')
            
            ax1.set_xlabel('X (AP)')
            ax1.set_ylabel('Z (Vertical)')
            ax1.set_title('Sagittal View', fontsize=13, fontweight='bold')
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            
            # Frontal view (y-z)
            for indices, color, width in connections:
                if isinstance(indices, int):
                    indices = [indices]
                
                for i in range(len(indices)-1):
                    idx1, idx2 = indices[i], indices[i+1]
                    ax2.plot([y[idx1], y[idx2]],
                            [z[idx1], z[idx2]],
                            color=color, linewidth=width, marker='o',
                            markersize=6, markerfacecolor=color,
                            markeredgecolor='black')
            
            ax2.set_xlabel('Y (ML)')
            ax2.set_ylabel('Z (Vertical)')
            ax2.set_title('Frontal View', fontsize=13, fontweight='bold')
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_connection_code(self, connections: list, var_name: str = "MARKER_CONNECTIONS"):
        """
        Generate Python code for the connections
        
        Args:
            connections: List of ([indices], color, width)
            var_name: Variable name for the output
        """
        print("\n" + "="*70)
        print("PYTHON CODE FOR CONNECTIONS")
        print("="*70)
        print(f"\n{var_name} = [")
        
        for indices, color, width in connections:
            print(f"    ({indices}, '{color}', {width}),")
        
        print("]")
        print("\n" + "="*70)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_identify_markers():
    """Identify markers in your dataset"""
    
    # Setup
    session_path = "path/to/your/qualisys/session"
    trial_name = "Walk 1"
    
    tool = MarkerSetupTool(session_path, trial_name)
    
    # Print marker list
    tool.print_marker_list()
    
    # Plot marker positions with labels
    tool.plot_marker_positions(frame_idx=0)


def example_test_connections():
    """Test your connection definitions"""
    
    # Setup
    session_path = "path/to/your/qualisys/session"
    trial_name = "Walk 1"
    
    tool = MarkerSetupTool(session_path, trial_name)
    
    # Define connections based on marker indices you identified
    # EXAMPLE - CUSTOMIZE THIS FOR YOUR MARKER SET!
    my_connections = [
        # Head/neck
        ([0, 1, 2], 'darkblue', 2.5),
        
        # Torso
        ([2, 3, 4, 5, 3], 'navy', 2.0),
        
        # Right arm
        ([3, 6, 7, 8], 'red', 2.0),
        
        # Left arm
        ([5, 9, 10, 11], 'blue', 2.0),
        
        # Pelvis
        ([4, 12, 13, 12], 'darkgreen', 2.5),
        
        # Right leg
        ([12, 14, 15, 16], 'darkred', 2.5),
        
        # Left leg
        ([13, 17, 18, 19], 'darkblue', 2.5),
    ]
    
    # Test in 3D
    print("Testing connections in 3D...")
    tool.test_connections(my_connections, frame_idx=0, view_3d=True)
    
    # Test in 2D
    print("Testing connections in 2D...")
    tool.test_connections(my_connections, frame_idx=0, view_3d=False)
    
    # Generate code
    tool.generate_connection_code(my_connections)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║              MARKER SETUP HELPER                              ║
    ╚════════════════════════════════════════════════════════════════╝
    
    This tool helps you define marker connectivity for stick figures.
    
    WORKFLOW:
    1. Run example_identify_markers() to see all markers with indices
    2. Create your connections list based on marker indices
    3. Run example_test_connections() to verify connections
    4. Copy generated code to your animation script!
    
    TIPS:
    - Connections are defined as: ([marker_indices], color, linewidth)
    - Indices can form chains: [0, 1, 2, 3] draws 0→1→2→3
    - Closed loops: [0, 1, 2, 3, 0] draws a square
    - Colors: 'red', 'blue', 'darkgreen', etc.
    - Typical linewidth: 2.0-3.0 for major segments, 1.5 for smaller ones
    """)
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "identify":
            example_identify_markers()
        elif sys.argv[1] == "test":
            example_test_connections()
    else:
        print("\nRun with:")
        print("  python marker_setup_helper.py identify  # Identify markers")
        print("  python marker_setup_helper.py test      # Test connections")