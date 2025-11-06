# Multi-Directory Trial Loading Examples

This directory contains examples and configuration files demonstrating how to load trials from multiple directories in PyPMAnalyzer.

## The Problem

Previously, PyPMAnalyzer could only load trials from a single session directory:

```python
# Old way - all trials must be in the same directory
data_dict = load_multiple_trials(
    session_path="/data/subject01/session1",
    trial_names=["Walk_01", "Walk_02", "Run_01"]
)
```

This made it difficult to load recordings from different subjects stored in different directories.

## The Solution

Now you can easily load trials from different directories using several methods:

### Method 1: TrialSpec Objects

```python
from ppma_pipeline import load_trials_from_specs, TrialSpec

specs = [
    TrialSpec("/data/subject01/session1", "Walk_01", "S01"),
    TrialSpec("/data/subject02/session1", "Walk_01", "S02"),
    TrialSpec("/data/subject03/session2", "Walk_02", "S03"),
]

data_dict = load_trials_from_specs(specs)
```

### Method 2: Path Tuples (Simpler)

```python
from ppma_pipeline import load_trials_from_paths

trial_paths = [
    ("/data/subject01/session1", "Walk_01"),
    ("/data/subject02/session1", "Walk_01"),
    ("/data/subject03/session2", "Walk_02"),
]
subject_ids = ["S01", "S02", "S03"]

data_dict = load_trials_from_paths(trial_paths, subject_ids)
```

### Method 3: Configuration Files (Most Convenient!)

Create a YAML or JSON config file:

**trials_config.yaml:**
```yaml
trials:
  - session_path: /data/subject01/session1
    trial_name: Walk_01
    subject_id: S01
  - session_path: /data/subject02/session1
    trial_name: Walk_01
    subject_id: S02
  - session_path: /data/subject03/session2
    trial_name: Walk_02
    subject_id: S03
```

Then load:
```python
from ppma_pipeline import load_trials_from_config

data_dict = load_trials_from_config("trials_config.yaml")
```

### Method 4: Direct Pipeline Integration

Pass pre-loaded data directly to the pipeline:

```python
from ppma_pipeline import load_trials_from_paths, run_pm_analysis_pipeline

# Load from different directories
trial_paths = [
    ("/data/subject01/session1", "Walk_01"),
    ("/data/subject02/session1", "Walk_01"),
    ("/data/subject03/session2", "Walk_02"),
]
data_dict = load_trials_from_paths(trial_paths, ["S01", "S02", "S03"])

# Run analysis on pre-loaded data
results = run_pm_analysis_pipeline(
    data_dict=data_dict,
    n_components=12,
    normalize_method='height',
    filter_cutoff=7.0,
    fs=100.0
)
```

## Files in This Directory

- **multi_directory_loading_examples.py** - Comprehensive examples showing all methods
- **trials_config.yaml** - Example YAML configuration file
- **trials_config.json** - Example JSON configuration file
- **README.md** - This file

## Quick Start

1. **Choose your method** - See examples above or in `multi_directory_loading_examples.py`

2. **For config files** - Copy and modify `trials_config.yaml` or `trials_config.json` with your trial paths

3. **Run your analysis:**

   ```python
   from ppma_pipeline import load_trials_from_config, run_pm_analysis_pipeline

   # Load trials
   data_dict = load_trials_from_config("my_trials.yaml")

   # Run analysis
   results = run_pm_analysis_pipeline(
       data_dict=data_dict,
       n_components=12,
       fs=100.0
   )
   ```

## Error Handling

All loading functions support `skip_errors=True` to continue loading even if some trials fail:

```python
# Skip trials that fail to load
data_dict = load_trials_from_config(
    "trials_config.yaml",
    skip_errors=True  # Continue even if some trials fail
)
```

## Backward Compatibility

The original API still works perfectly for single-directory cases:

```python
# Original API - still works!
data_dict = load_multiple_trials(
    session_path="/data/session1",
    trial_names=["Walk_01", "Walk_02"]
)

# Original pipeline API - still works!
results = run_pm_analysis_pipeline(
    session_path="/data/session1",
    trial_names=["Walk_01", "Walk_02"],
    n_components=12
)
```

## Tips

1. **Use config files** for batch processing - they're the most maintainable
2. **Use `skip_errors=True`** when loading many trials to handle failures gracefully
3. **Use absolute paths** in config files to avoid path confusion
4. **Organize by task** - create separate config files for different analyses

## Example Use Cases

### Use Case 1: Multi-Subject Gait Analysis

Load walking trials from 20 different subjects:

```yaml
# gait_study.yaml
trials:
  - session_path: /data/controls/subject01/baseline
    trial_name: Walk_01
    subject_id: C01
  - session_path: /data/controls/subject02/baseline
    trial_name: Walk_01
    subject_id: C02
  # ... 18 more subjects
```

```python
data_dict = load_trials_from_config("gait_study.yaml")
results = run_pm_analysis_pipeline(data_dict=data_dict, n_components=12)
```

### Use Case 2: Longitudinal Study

Load pre- and post-intervention trials:

```yaml
# longitudinal_study.yaml
trials:
  - session_path: /data/subject01/pre_intervention
    trial_name: Walk_01
    subject_id: S01_pre
  - session_path: /data/subject01/post_intervention
    trial_name: Walk_01
    subject_id: S01_post
  - session_path: /data/subject02/pre_intervention
    trial_name: Walk_01
    subject_id: S02_pre
  - session_path: /data/subject02/post_intervention
    trial_name: Walk_01
    subject_id: S02_post
```

### Use Case 3: Different Tasks

Compare different movement tasks across subjects:

```python
specs = [
    TrialSpec("/data/s01/walking", "Walk_01", "S01_walk"),
    TrialSpec("/data/s01/running", "Run_01", "S01_run"),
    TrialSpec("/data/s02/walking", "Walk_01", "S02_walk"),
    TrialSpec("/data/s02/running", "Run_01", "S02_run"),
]
data_dict = load_trials_from_specs(specs)
```

## Need Help?

Check `multi_directory_loading_examples.py` for detailed examples of each method.
