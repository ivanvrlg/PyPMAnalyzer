"""
Examples demonstrating how to load trials from multiple directories

This script shows different ways to load walking recordings from different
subjects stored in different directories.
"""

from ppma_pipeline import (
    load_multiple_trials,          # Original function (single directory)
    load_trials_from_specs,        # Load using TrialSpec objects
    load_trials_from_paths,        # Load using (path, name) tuples
    load_trials_from_config,       # Load from YAML/JSON config file
    run_pm_analysis_pipeline,      # Main analysis pipeline
    TrialSpec
)


# =============================================================================
# METHOD 1: Original API (Single Directory)
# =============================================================================
# Use this when all trials are in the same directory

def example_single_directory():
    """Load multiple trials from a single session directory (original API)"""
    print("\n" + "="*70)
    print("METHOD 1: Single Directory (Original API)")
    print("="*70)

    session_path = "/data/subject01/session1"
    trial_names = ["Walk_01", "Walk_02", "Run_01"]
    subject_ids = ["S01", "S01", "S01"]  # Same subject, different trials

    data_dict = load_multiple_trials(session_path, trial_names, subject_ids)
    print(f"\nLoaded {len(data_dict)} trials")
    return data_dict


# =============================================================================
# METHOD 2: TrialSpec Objects (Multiple Directories)
# =============================================================================
# Use this for full control when loading from different directories

def example_trial_specs():
    """Load trials from different directories using TrialSpec objects"""
    print("\n" + "="*70)
    print("METHOD 2: TrialSpec Objects (Multiple Directories)")
    print("="*70)

    # Create specifications for each trial
    # Each trial can be in a completely different directory!
    trial_specs = [
        TrialSpec("/data/subject01/session1", "Walk_01", "S01"),
        TrialSpec("/data/subject02/session1", "Walk_01", "S02"),
        TrialSpec("/data/subject03/session2", "Walk_02", "S03"),
        TrialSpec("/data/subject04/baseline", "Walk_01", "S04"),
        TrialSpec("/data/subject05/baseline", "Walk_01", "S05"),
    ]

    # Load all trials (skip_errors=True means continue if one fails)
    data_dict = load_trials_from_specs(trial_specs, skip_errors=True)
    print(f"\nLoaded {len(data_dict)} trials")
    return data_dict


# =============================================================================
# METHOD 3: Path Tuples (Multiple Directories - Simpler Syntax)
# =============================================================================
# Use this for a simpler syntax when loading from different directories

def example_path_tuples():
    """Load trials using (session_path, trial_name) tuples"""
    print("\n" + "="*70)
    print("METHOD 3: Path Tuples (Multiple Directories - Simpler)")
    print("="*70)

    # Specify each trial as (session_path, trial_name)
    trial_paths = [
        ("/data/subject01/session1", "Walk_01"),
        ("/data/subject02/session1", "Walk_01"),
        ("/data/subject03/session2", "Walk_02"),
        ("/data/subject04/baseline", "Walk_01"),
        ("/data/subject05/baseline", "Walk_01"),
    ]

    # Specify subject IDs separately
    subject_ids = ["S01", "S02", "S03", "S04", "S05"]

    data_dict = load_trials_from_paths(
        trial_paths,
        subject_ids=subject_ids,
        skip_errors=True
    )
    print(f"\nLoaded {len(data_dict)} trials")
    return data_dict


# =============================================================================
# METHOD 4: Configuration File (YAML or JSON)
# =============================================================================
# Use this for the most convenient batch loading - just maintain a config file!

def example_config_file_yaml():
    """Load trials from a YAML configuration file"""
    print("\n" + "="*70)
    print("METHOD 4a: YAML Configuration File (Most Convenient!)")
    print("="*70)

    # All trial specifications are in the YAML file
    # See trials_config.yaml for the format
    data_dict = load_trials_from_config(
        "examples/trials_config.yaml",
        skip_errors=True
    )
    print(f"\nLoaded {len(data_dict)} trials from config file")
    return data_dict


def example_config_file_json():
    """Load trials from a JSON configuration file"""
    print("\n" + "="*70)
    print("METHOD 4b: JSON Configuration File")
    print("="*70)

    # All trial specifications are in the JSON file
    # See trials_config.json for the format
    data_dict = load_trials_from_config(
        "examples/trials_config.json",
        skip_errors=True
    )
    print(f"\nLoaded {len(data_dict)} trials from config file")
    return data_dict


# =============================================================================
# METHOD 5: Using Pre-Loaded Data with the Pipeline
# =============================================================================
# Once you've loaded data, you can pass it directly to the analysis pipeline

def example_pipeline_with_preloaded_data():
    """Run the complete pipeline with pre-loaded data"""
    print("\n" + "="*70)
    print("METHOD 5: Pipeline with Pre-Loaded Data")
    print("="*70)

    # First, load data from multiple directories
    trial_paths = [
        ("/data/subject01/session1", "Walk_01"),
        ("/data/subject02/session1", "Walk_01"),
        ("/data/subject03/session2", "Walk_02"),
    ]
    subject_ids = ["S01", "S02", "S03"]

    data_dict = load_trials_from_paths(trial_paths, subject_ids, skip_errors=True)

    # Now run the complete analysis pipeline with the pre-loaded data
    # No need to specify session_path or trial_names!
    preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics = run_pm_analysis_pipeline(
        data_dict=data_dict,           # Pass pre-loaded data
        n_components=12,
        normalize_method='height',
        filter_cutoff=7.0,
        fs=100.0,
        trim_start=0.5,                # Trim first 0.5 seconds
        trim_end=10.5                  # Trim after 10.5 seconds
    )

    print("\n Pipeline complete!")
    return preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics


# =============================================================================
# METHOD 6: Traditional Pipeline (Backward Compatible)
# =============================================================================
# The original pipeline API still works for single-directory cases

def example_traditional_pipeline():
    """Run pipeline the traditional way (single directory)"""
    print("\n" + "="*70)
    print("METHOD 6: Traditional Pipeline (Backward Compatible)")
    print("="*70)

    # This is the original API - still works perfectly!
    preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics = run_pm_analysis_pipeline(
        session_path="/data/subject01/session1",
        trial_names=["Walk_01", "Walk_02", "Run_01"],
        subject_ids=["S01", "S01", "S01"],
        n_components=12,
        normalize_method='height',
        filter_cutoff=7.0,
        fs=100.0
    )

    print("\nPipeline complete!")
    return preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics


# =============================================================================
# COMPLETE WORKFLOW EXAMPLE
# =============================================================================

def complete_example():
    """
    Complete example: Load 5 walking trials from different subjects
    and run the full analysis pipeline
    """
    print("\n" + "="*80)
    print(" COMPLETE WORKFLOW: Multi-Subject Gait Analysis ")
    print("="*80)

    # STEP 1: Define trials from different subjects/directories
    print("\nSTEP 1: Defining trials from different subjects...")

    trial_specs = [
        TrialSpec("/data/subject01/session1", "Walk_01", "S01"),
        TrialSpec("/data/subject02/session1", "Walk_01", "S02"),
        TrialSpec("/data/subject03/session2", "Walk_02", "S03"),
        TrialSpec("/data/subject04/baseline", "Walk_01", "S04"),
        TrialSpec("/data/subject05/baseline", "Walk_01", "S05"),
    ]

    # STEP 2: Load all trials
    print("\nSTEP 2: Loading trials...")
    data_dict = load_trials_from_specs(trial_specs, skip_errors=True)

    # STEP 3: Run complete analysis pipeline
    print("\nSTEP 3: Running Principal Movement Analysis...")
    preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics = run_pm_analysis_pipeline(
        data_dict=data_dict,
        n_components=12,
        normalize_method='height',
        filter_cutoff=7.0,
        fs=100.0,
        trim_start=0.5,    # Trim first 0.5s
        trim_end=10.5      # Trim after 10.5s
    )

    # STEP 4: Access results
    print("\nSTEP 4: Accessing results...")
    print(f"  Total variance explained: {pma.explained_variance_ratio_.sum():.2f}%")
    print(f"  Number of trials analyzed: {len(pp_dict)}")
    print("\n  Movement structure (rVAR) for first 3 PMs:")
    for key in list(pp_dict.keys())[:3]:
        rvar = metrics['rVAR'][key]
        print(f"    {key}: PM1={rvar[0]:.1f}%, PM2={rvar[1]:.1f}%, PM3={rvar[2]:.1f}%")

    return preprocessor, pma, pp_dict, pv_dict, pa_dict, metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     Multi-Directory Trial Loading Examples for PyPMAnalyzer          ║
    ║                                                                      ║
    ║  Choose a method to load trials from different directories:         ║
    ║                                                                      ║
    ║  1. Original API (single directory)                                 ║
    ║  2. TrialSpec objects (full control)                                ║
    ║  3. Path tuples (simpler syntax)                                    ║
    ║  4. YAML/JSON config files (most convenient!)                       ║
    ║  5. Pre-loaded data with pipeline                                   ║
    ║  6. Traditional pipeline (backward compatible)                      ║
    ║                                                                      ║
    ║  See examples below for detailed usage of each method.              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Uncomment the example you want to run:

    # data_dict = example_single_directory()
    # data_dict = example_trial_specs()
    # data_dict = example_path_tuples()
    # data_dict = example_config_file_yaml()
    # data_dict = example_config_file_json()
    # results = example_pipeline_with_preloaded_data()
    # results = example_traditional_pipeline()
    # results = complete_example()

    print("\n✓ Examples script loaded. Uncomment the example you want to run.")
