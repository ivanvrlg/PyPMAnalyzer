"""
Basic example of using PyPMAnalyzer.

This script demonstrates the complete workflow:
1. Load data
2. Preprocess
3. Fit PCA
4. Compute metrics
5. Basic visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import PyPMAnalyzer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pypmanalyzer import (
    load_directory,
    Preprocessor,
    PrincipalMovementAnalysis,
    compute_rVAR,
    compute_pv_std,
    compute_zero_crossings,
    compute_zc_variability
)


def main():
    """Run basic PyPMAnalyzer example."""
    
    print("="*60)
    print("PyPMAnalyzer Basic Example")
    print("="*60)
    
    # ========================================
    # Step 1: Load data
    # ========================================
    print("\n1. Loading data...")
    
    # Option A: Load from directory
    data_dir = Path(r'S:\IoN_MathMD\CMD_lab\Exported_QTM_data\sub-012\ses-001_date-2025-05-15\TSV')  # Adjust to your data path
    
    if data_dir.exists():
        data_dict, metadata = load_directory(
            data_dir,
            pattern='*.tsv',  # or '*.csv'
            subject_trial_pattern=r'subject(\d+)_trial(\d+)',
            # For Qualisys TSV:
            # reader_func=lambda f: read_qualisys_tsv(f, skip_rows=11),
        )
    else:
        # Option B: Create synthetic data for testing
        print("Data directory not found, creating synthetic data...")
        data_dict = create_synthetic_data()
    
    print(f"Loaded {len(data_dict)} trials")
    for key, data in data_dict.items():
        print(f"  Trial {key}: shape {data.shape}")
    
    # ========================================
    # Step 2: Create subject info (for preprocessing)
    # ========================================
    print("\n2. Preparing subject info...")
    
    subject_info = {}
    for key in data_dict.keys():
        subject_id, trial_id = key
        subject_info[key] = {
            'height': 1.59,  # meters - adjust as needed
            'gender': 'female'
        }
    
    # ========================================
    # Step 3: Preprocess
    # ========================================
    print("\n3. Preprocessing...")
    
    preprocessor = Preprocessor(
        center=True,
        weight_method=None,  # Set to 'standard_male' if you have proper marker mapping
        normalize_method='height',
        order='normalize_center_weight',  # height normalization first
        filter_cutoff=7.0,
        filter_order=3
    )
    
    data_preprocessed = preprocessor.fit_transform(
        data_dict,
        subject_info,
        fs=100  # Adjust to your sampling frequency
    )
    
    print("Preprocessing complete!")
    
    # ========================================
    # Step 4: Fit PCA
    # ========================================
    print("\n4. Fitting PCA...")
    
    pma = PrincipalMovementAnalysis(n_components=15)
    pma.fit(data_preprocessed)
    
    print(f"PCA fitted with {pma.n_components_fitted_} components")
    print(f"Explained variance (first 5 PMs): {pma.explained_variance_ratio_[:5]}")
    
    # ========================================
    # Step 5: Compute derivatives
    # ========================================
    print("\n5. Computing derivatives...")
    
    pp, pv, pa = pma.compute_all(fs=100)
    
    print(f"Computed PP, PV, PA for {len(pp)} trials")
    
    # ========================================
    # Step 6: Compute metrics
    # ========================================
    print("\n6. Computing metrics...")
    
    # Kinematic metrics
    rvar = compute_rVAR(pp)
    pv_std = compute_pv_std(pv)
    
    # Control metrics
    n_zc = compute_zero_crossings(pa)
    sigma = compute_zc_variability(pa, fs=100)
    
    print("Metrics computed!")
    
    # Print example results
    key = list(pp.keys())[0]
    print(f"\nExample results for trial {key}:")
    print(f"  rVAR (first 5 PMs): {rvar[key][:5]}")
    print(f"  N (zero-crossings, first 5 PMs): {n_zc[key][:5]}")
    print(f"  σ (ZC variability, first 5 PMs): {sigma[key][:5]}")
    
    # ========================================
    # Step 7: Basic visualization
    # ========================================
    print("\n7. Creating visualizations...")
    
    visualize_results(pma, pp, rvar, key)
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


def create_synthetic_data(n_trials=5, n_frames=1000, n_markers=20):
    """Create synthetic kinematic data for testing."""
    print("  Generating synthetic data...")
    
    data_dict = {}
    
    for i in range(n_trials):
        subject_id = (i // 2) + 1
        trial_id = (i % 2) + 1
        
        # Create random walk data with some structure
        data = np.zeros((n_frames, n_markers * 3))
        
        for j in range(n_markers * 3):
            # Random walk with drift
            noise = np.random.randn(n_frames) * 0.01
            drift = np.sin(np.linspace(0, 2*np.pi, n_frames)) * 0.05
            data[:, j] = np.cumsum(noise) + drift + np.random.randn() * 0.1
        
        data_dict[(subject_id, trial_id)] = data
    
    return data_dict


def visualize_results(pma, pp, rvar, key):
    """Create basic visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scree plot
    ax = axes[0, 0]
    n_comp = min(15, len(pma.explained_variance_ratio_))
    ax.bar(range(1, n_comp+1), pma.explained_variance_ratio_[:n_comp])
    ax.set_xlabel('PM Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Scree Plot')
    ax.grid(alpha=0.3)
    
    # 2. Cumulative variance
    ax = axes[0, 1]
    cumvar = np.cumsum(pma.explained_variance_ratio_[:n_comp])
    ax.plot(range(1, n_comp+1), cumvar, 'o-')
    ax.axhline(95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('PM Component')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('Cumulative Explained Variance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. PP time series (first 3 components)
    ax = axes[1, 0]
    for i in range(min(3, pp[key].shape[1])):
        ax.plot(pp[key][:, i], label=f'PM{i+1}', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('PP (AU)')
    ax.set_title(f'Principal Positions - Trial {key}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. rVAR comparison
    ax = axes[1, 1]
    n_comp_rvar = min(10, len(rvar[key]))
    ax.bar(range(1, n_comp_rvar+1), rvar[key][:n_comp_rvar])
    ax.set_xlabel('PM Component')
    ax.set_ylabel('rVAR (%)')
    ax.set_title(f'Relative Variance - Trial {key}')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pypmanalyzer_example_results.png', dpi=150)
    print("  Saved figure: pypmanalyzer_example_results.png")
    plt.show()


if __name__ == '__main__':
    main()