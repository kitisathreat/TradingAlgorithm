#!/usr/bin/env python3
"""
Diagnostic script to identify ModelTrainer import issues on EC2
"""

import sys
import os
from pathlib import Path

def check_python_path():
    """Check Python path and available modules"""
    print("=== Python Path Analysis ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    
    print("\nPython path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print("\n=== Module Availability ===")
    
    # Check for key dependencies
    dependencies = [
        'tensorflow',
        'numpy',
        'pandas',
        'sklearn',
        'yfinance',
        'flask',
        'flask_socketio'
    ]
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {dep}: {version}")
        except ImportError as e:
            print(f"✗ {dep}: {e}")

def check_model_trainer_import():
    """Check ModelTrainer import with detailed diagnostics"""
    print("\n=== ModelTrainer Import Test ===")
    
    # Add potential paths to sys.path
    potential_paths = [
        Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python",
        Path(__file__).parent / "_2_Orchestrator_And_ML_Python",
        Path("/var/app/current/_2_Orchestrator_And_ML_Python"),
        Path("/var/_2_Orchestrator_And_ML_Python")
    ]
    
    print("Checking potential ModelTrainer paths:")
    for path in potential_paths:
        print(f"  {path}: {'exists' if path.exists() else 'missing'}")
        if path.exists():
            print(f"    Contents: {list(path.iterdir())}")
    
    # Try different import approaches
    import_attempts = [
        ('_2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer', 'ModelTrainer'),
        ('interactive_training_app.backend.model_trainer', 'ModelTrainer'),
        ('backend.model_trainer', 'ModelTrainer'),
        ('model_trainer', 'ModelTrainer')
    ]
    
    for import_path, class_name in import_attempts:
        print(f"\nTrying: from {import_path} import {class_name}")
        try:
            # Add parent directories to path
            for potential_path in potential_paths:
                if potential_path.exists():
                    sys.path.insert(0, str(potential_path))
            
            module = __import__(import_path, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"✓ Success: {class_name} imported from {import_path}")
            
            # Try to instantiate
            try:
                instance = class_obj()
                print(f"✓ Success: {class_name} instantiated")
                
                # Try basic methods
                try:
                    state = instance.get_model_state()
                    print(f"✓ Success: get_model_state() returned {state}")
                except Exception as e:
                    print(f"✗ get_model_state() failed: {e}")
                
                return True
                
            except Exception as e:
                print(f"✗ Instantiation failed: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
        except ImportError as e:
            print(f"✗ Import failed: {e}")
        except AttributeError as e:
            print(f"✗ Attribute error: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    return False

def check_file_structure():
    """Check the file structure around the ModelTrainer"""
    print("\n=== File Structure Analysis ===")
    
    current_dir = Path(__file__).parent
    print(f"Current directory: {current_dir}")
    
    # Check for ModelTrainer files
    model_trainer_paths = [
        current_dir.parent / "_2_Orchestrator_And_ML_Python" / "interactive_training_app" / "backend" / "model_trainer.py",
        current_dir / "_2_Orchestrator_And_ML_Python" / "interactive_training_app" / "backend" / "model_trainer.py",
        Path("/var/app/current/_2_Orchestrator_And_ML_Python/interactive_training_app/backend/model_trainer.py"),
        Path("/var/_2_Orchestrator_And_ML_Python/interactive_training_app/backend/model_trainer.py")
    ]
    
    for path in model_trainer_paths:
        print(f"\nChecking: {path}")
        if path.exists():
            print(f"  ✓ File exists")
            print(f"  Size: {path.stat().st_size} bytes")
            
            # Check file contents (first few lines)
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()[:10]
                    print(f"  First 10 lines:")
                    for i, line in enumerate(lines, 1):
                        print(f"    {i}: {line.rstrip()}")
            except Exception as e:
                print(f"  ✗ Error reading file: {e}")
        else:
            print(f"  ✗ File missing")

def main():
    """Run all diagnostics"""
    print("=" * 80)
    print("           MODELTRAINER DIAGNOSTIC TOOL")
    print("=" * 80)
    
    check_python_path()
    check_file_structure()
    success = check_model_trainer_import()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ ModelTrainer import successful!")
    else:
        print("✗ ModelTrainer import failed")
    print("=" * 80)

if __name__ == '__main__':
    main() 