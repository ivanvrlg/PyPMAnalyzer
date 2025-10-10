"""Quick test of PyPMAnalyzer installation and functionality."""

import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from pypmanalyzer import (
        PrincipalMovementAnalysis,
        Preprocessor,
        central_difference,
        compute_rVAR,
        compute_zero_crossings
    )
    
    print("✓ All imports successful!")


def test_derivatives():
    """Test derivative computation."""
    print("\nTesting derivatives...")
    
    from pypmanalyzer import central_difference
    
    # Test on simple function: x = t^2, dx/dt = 2t
    t = np.linspace(0, 10, 100)
    x = t**2
    dx = central_difference(x, fs=10)  # fs = 10 Hz (dt = 0.1)
    
    # Expected: approximately 2*t (at middle points)
    t_mid = t[1:-1]
    expected = 2 * t_mid
    
    error = np.mean(np.abs(dx - expected))
    print(f"  Mean error in derivative: {error:.6f}")
    
    if error < 0.01:
        print("✓ Derivatives working correctly!")
    else:
        print("✗ Derivative computation may have issues")


def test_pca_pipeline():
    """Test complete PCA pipeline."""
    print("\nTesting PCA pipeline...")
    
    from pypmanalyzer import (
        Preprocessor,
        PrincipalMovementAnalysis,
        compute_rVAR
    )
    
    # Create synthetic data
    data_dict = {
        (1, 1): np.random.randn(100, 60),  # 20 markers * 3
        (1, 2): np.random.randn(100, 60),
        (2, 1): np.random.randn(100, 60),
    }
    
    # Preprocess
    prep = Preprocessor(center=True, normalize_method=None)
    data_prep = prep.fit_transform(data_dict)
    
    # Fit PCA
    pma = PrincipalMovementAnalysis(n_components=10)
    pma.fit(data_prep)
    
    # Compute metrics
    rvar = compute_rVAR(pma.pp_)
    
    # Verify
    assert len(pma.pp_) == 3, "Should have 3 trials"
    assert pma.pp_[(1,1)].shape[1] == 10, "Should have 10 components"
    assert np.abs(np.sum(rvar[(1,1)]) - 100) < 0.01, "rVAR should sum to 100"
    
    print(f"  Explained variance (first 5): {pma.explained_variance_ratio_[:5]}")
    print(f"  rVAR for trial (1,1): {rvar[(1,1)][:5]}")
    print("✓ PCA pipeline working!")


def test_control_metrics():
    """Test control metrics."""
    print("\nTesting control metrics...")
    
    from pypmanalyzer import compute_zero_crossings, compute_zc_variability
    
    # Create oscillating signal
    t = np.linspace(0, 10, 1000)
    pa = np.sin(2 * np.pi * t).reshape(-1, 1)  # 1 Hz sine wave
    
    pa_dict = {(1, 1): pa}
    
    n = compute_zero_crossings(pa_dict)
    sigma = compute_zc_variability(pa_dict, fs=100)
    
    # Sine wave at 1 Hz over 10 seconds should have ~20 zero-crossings
    print(f"  Zero-crossings: {n[(1,1)][0]:.0f} (expected ~20)")
    print(f"  σ: {sigma[(1,1)][0]:.4f} s")
    
    if 18 <= n[(1,1)][0] <= 22:
        print("✓ Control metrics working!")
    else:
        print("✗ Control metrics may have issues")


if __name__ == '__main__':
    print("="*60)
    print("PyPMAnalyzer Test Suite")
    print("="*60)
    
    test_imports()
    test_derivatives()
    test_pca_pipeline()
    test_control_metrics()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)