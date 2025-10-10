import sys
import os

print("="*60)
print("DIAGNOSTIC")
print("="*60)

# Check current directory
print(f"\nCurrent directory: {os.getcwd()}")

# Check if pypmanalyzer folder exists
print(f"\npypmanalyzer folder exists: {os.path.exists('pypmanalyzer')}")

if os.path.exists('pypmanalyzer'):
    print(f"Contents: {os.listdir('pypmanalyzer')}")
    
    # Check if __init__.py exists
    init_exists = os.path.exists('pypmanalyzer/__init__.py')
    print(f"__init__.py exists: {init_exists}")
    
    # Check subfolders
    for subfolder in ['core', 'io', 'metrics']:
        path = f'pypmanalyzer/{subfolder}'
        exists = os.path.exists(path)
        print(f"{subfolder}/ exists: {exists}")
        if exists:
            init_path = f'{path}/__init__.py'
            print(f"  - {subfolder}/__init__.py exists: {os.path.exists(init_path)}")

# Add to path
sys.path.insert(0, os.getcwd())
print(f"\nAdded to sys.path: {os.getcwd()}")

# Try to import
print("\nAttempting import...")
try:
    import pypmanalyzer
    print("✓ SUCCESS!")
    print(f"Version: {pypmanalyzer.__version__}")
    print(f"Location: {pypmanalyzer.__file__}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()