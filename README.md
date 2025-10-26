# PyPMAnalyzer

Python implementation of Principal Movement Analysis (PMA) for motion capture data, based on the PMAnalyser by Federolf.

## Overview

PyPMAnalyzer provides tools for analyzing motion capture data using Principal Component Analysis (PCA) to identify and visualize principal movements (PMs) in human motion.

## Features

- ✅ Principal Movement Analysis (PMA) using PCA
- ✅ Motion reconstruction from principal components
- ✅ 2D animation visualization (sagittal and frontal views)
- ✅ Side-by-side comparison of multiple PMs
- ✅ Support for Qualisys marker sets
- ✅ Flexible styling options (colored or monochrome)
- ✅ Video export (MP4, GIF)

## Installation

### For Lab Members

```bash
# Install CMD_Loaders (lab package)
git clone git@github.com:CMD-Lab/CMD_Loaders.git
pip install -e CMD_Loaders

# Install PyPMAnalyzer
git clone git@github.com:yourusername/PyPMAnalyzer.git
pip install -e PyPMAnalyzer
```

### For External Users

```bash
git clone https://github.com/yourusername/PyPMAnalyzer.git
cd PyPMAnalyzer
pip install -e .
```

**Note**: Some features require the CMD_Loaders package for data loading. Contact the lab for access.

## Quick Start

```python
from visualize import visualize_original_motion, visualize_principal_movements
from ppma_core import PrincipalMovementAnalysis

# Load your motion data (n_frames, n_coords)
# data = ... your loading code ...

# Visualize original motion
visualize_original_motion(
    data,
    title="Walking Trial",
    fps=100,
    style='qtm'  # colored markers
)

# Run PMA analysis
pma = PrincipalMovementAnalysis()
pma.fit(data_dict)

# Visualize principal movements
visualize_principal_movements(
    pma,
    pm_indices=[0, 1, 2],  # First 3 PMs
    amplifications=[2.0, 2.0, 2.0],
    title="Top 3 Principal Movements"
)
```

## Usage Examples

### Load and Visualize Motion Data

```python
from cmd_loaders import load_qualisys_tsv
from visualize import visualize_original_motion

# Load data
data = load_qualisys_tsv("path/to/trial.tsv")

# Create animation
visualize_original_motion(
    data,
    title="Walking Trial",
    save_path="output/walking.mp4",
    fps=100,
    style='qtm'
)
```

### Principal Movement Analysis

```python
from ppma_core import PrincipalMovementAnalysis
from ppma_pipeline import PreProcessor
from visualize import visualize_principal_movements

# Preprocess data
preprocessor = PreProcessor()
data_dict = preprocessor.fit_transform(raw_data_dict)

# Run PMA
pma = PrincipalMovementAnalysis()
pma.fit(data_dict)

# Visualize results
visualize_principal_movements(
    pma,
    pm_indices=[0, 1, 2],
    preprocessor=preprocessor,
    n_frames=60,
    fps=20,
    save_path="output/pms.mp4"
)
```

### Advanced: Custom Animation

```python
from animation import Animator2D, PMReconstructor

# Create custom animator
animator = Animator2D()

# Animate with custom settings
animator.animate(
    data,
    title="Custom Animation",
    show_sagittal=True,
    show_frontal=True,
    show_bones=True,
    style='black',  # monochrome style
    bone_width=3.0,
    marker_size=8,
    fps=30
)
```

## Project Structure

```
pypmanalyzer/
├── ppma_core.py              # Core PMA algorithms
├── ppma_pipeline.py          # Analysis pipeline
├── visualize.py              # High-level visualization API
└── animation/                # Animation subpackage
    ├── animator_2d.py        # 2D animations
    ├── animator_comparison.py # Multi-PM comparisons
    ├── reconstruction.py     # PM reconstruction
    ├── marker_sets.py        # Skeleton definitions
    ├── styles.py             # Visual styles
    └── utils.py              # Utilities
```

## Documentation

For detailed documentation, see:
- [Structure Guide](STRUCTURE_GUIDE.md) - Architecture and extension guide
- [Setup Instructions](SETUP_INSTRUCTIONS.md) - Installation and configuration

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.6.0

## Contributing

This project is part of the Computational Movement Disorders Lab. For contributions or questions, please contact the lab.

## Citation

If you use PyPMAnalyzer in your research, please cite:

```
[Your citation information here]
```

Based on PMAnalyser by Federolf et al.

## License

MIT License - see LICENSE file for details

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Lab**: Computational Movement Disorders Lab
- **Institution**: Your Institution

## Acknowledgments

- Based on PMAnalyser by Peter Federolf
- Developed at the Computational Movement Disorders Lab
- Thanks to all lab members for feedback and testing
