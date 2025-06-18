#!/usr/bin/env python3
"""
Verify deployment package contents
"""

import zipfile
import os

def verify_deployment_package(zip_path='deployment.zip'):
    """Verify that the deployment package contains all necessary files"""
    
    if not os.path.exists(zip_path):
        print(f"ERROR: Deployment package not found: {zip_path}")
        return False
    
    print(f"Verifying deployment package: {zip_path}")
    print("=" * 50)
    
    required_files = [
        # Flask web files
        'flask_app.py',
        'wsgi.py',
        'requirements.txt',
        'Procfile',
        'gunicorn.conf.py',
        'config.py',
        '.ebextensions/01_flask.config',
        '.ebextensions/02_dependencies.config',
        
        # Orchestrator modules
        '_2_Orchestrator_And_ML_Python/__init__.py',
        '_2_Orchestrator_And_ML_Python/interactive_training_app/__init__.py',
        '_2_Orchestrator_And_ML_Python/interactive_training_app/backend/model_trainer.py',
        '_2_Orchestrator_And_ML_Python/stock_selection_utils.py',
        '_2_Orchestrator_And_ML_Python/sp100_symbols.json',
        '_2_Orchestrator_And_ML_Python/sp100_symbols_with_market_cap.json',
    ]
    
    missing_files = []
    found_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_list = zipf.namelist()
        
        print(f"Total files in package: {len(file_list)}")
        print()
        
        for required_file in required_files:
            if required_file in file_list:
                found_files.append(required_file)
                print(f"✓ {required_file}")
            else:
                missing_files.append(required_file)
                print(f"✗ {required_file} - MISSING")
    
    print()
    print("=" * 50)
    print(f"Found: {len(found_files)}/{len(required_files)} required files")
    
    if missing_files:
        print(f"Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✓ All required files present!")
        return True

if __name__ == '__main__':
    success = verify_deployment_package()
    if success:
        print("\nDeployment package is ready for AWS!")
    else:
        print("\nDeployment package is missing required files!") 