#!/usr/bin/env python3
"""
Flask Deployment Zip Creator for Elastic Beanstalk
Creates a deployment package for AWS Elastic Beanstalk Flask applications.
Includes ML orchestrator components for full trading functionality.
"""

import os
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import argparse

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_colored(text, color):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.RESET}")

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'flask_app.py',
        'wsgi.py', 
        'requirements.txt',
        'Procfile',
        'gunicorn.conf.py',
        'config.py'
    ]
    
    required_dirs = [
        '.ebextensions'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    return missing_files, missing_dirs

def check_ml_components():
    """Check if ML orchestrator components exist"""
    ml_path = Path(__file__).parent.parent.parent / "_2_Orchestrator_And_ML_Python"
    
    required_ml_files = [
        'model_bridge.py',
        'stock_selection_utils.py',
        'date_range_utils.py',
        'main.py',
        '__init__.py'
    ]
    
    required_ml_dirs = [
        'interactive_training_app'
    ]
    
    optional_ml_files = [
        'sp100_symbols.json',
        'sp100_symbols_with_market_cap.json',
        'trading_executor.py'
    ]
    
    missing_files = []
    missing_dirs = []
    optional_missing = []
    
    if not ml_path.exists():
        return False, ["ML orchestrator directory not found"], [], []
    
    for file in required_ml_files:
        if not (ml_path / file).exists():
            missing_files.append(file)
    
    for dir_path in required_ml_dirs:
        if not (ml_path / dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file in optional_ml_files:
        if not (ml_path / file).exists():
            optional_missing.append(file)
    
    return True, missing_files, missing_dirs, optional_missing

def create_ebignore(temp_dir):
    """Create .ebignore file to exclude unnecessary files"""
    ebignore_content = """# Elastic Beanstalk ignore file
# Exclude files that shouldn't be deployed
*.bat
*.sh
*.pyc
__pycache__/
.git/
.gitignore
README*.md
AWS_DEPLOYMENT_GUIDE.md
README_DEPLOYMENT.md
test_*.py
run_*.py
setup_*.py
deploy_*.bat
deploy_*.sh
docker-compose.yml
Dockerfile
.env
*.log
logs/
venv/
__pycache__/
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.vscode/
.idea/
create_deployment_zip.py
create_deployment_zip.bat
run_deployment_creator.bat
"""
    
    with open(os.path.join(temp_dir, '.ebignore'), 'w') as f:
        f.write(ebignore_content)

def copy_files_to_temp(temp_dir, include_static=True):
    """Copy all necessary files to temporary directory"""
    # Core Flask files
    core_files = [
        'flask_app.py',
        'wsgi.py',
        'requirements.txt',
        'Procfile',
        'gunicorn.conf.py',
        'config.py'
    ]
    
    for file in core_files:
        if os.path.exists(file):
            shutil.copy2(file, temp_dir)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ✗ Missing {file}")
    
    # Copy .ebextensions directory
    if os.path.exists('.ebextensions'):
        shutil.copytree('.ebextensions', os.path.join(temp_dir, '.ebextensions'))
        print("  ✓ Copied .ebextensions/")
    else:
        print("  ✗ Missing .ebextensions/")
    
    # Copy templates directory
    if os.path.exists('templates'):
        shutil.copytree('templates', os.path.join(temp_dir, 'templates'))
        print("  ✓ Copied templates/")
    else:
        print("  ⚠ No templates/ directory found")
    
    # Copy static directory if it exists and include_static is True
    if include_static and os.path.exists('static'):
        shutil.copytree('static', os.path.join(temp_dir, 'static'))
        print("  ✓ Copied static/")
    elif os.path.exists('static'):
        print("  ⚠ Skipping static/ (use --include-static to include)")

def copy_ml_components(temp_dir):
    """Copy ML orchestrator components to temporary directory"""
    ml_path = Path(__file__).parent.parent.parent / "_2_Orchestrator_And_ML_Python"
    
    if not ml_path.exists():
        print("  ✗ ML orchestrator directory not found")
        return False
    
    # Create ML orchestrator directory
    ml_temp_dir = os.path.join(temp_dir, '_2_Orchestrator_And_ML_Python')
    os.makedirs(ml_temp_dir, exist_ok=True)
    
    # Core ML files
    core_ml_files = [
        'model_bridge.py',
        'stock_selection_utils.py',
        'date_range_utils.py',
        'main.py',
        '__init__.py'
    ]
    
    for file in core_ml_files:
        src = ml_path / file
        if src.exists():
            shutil.copy2(src, ml_temp_dir)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ✗ Missing {file}")
    
    # Optional ML files
    optional_ml_files = [
        'sp100_symbols.json',
        'sp100_symbols_with_market_cap.json',
        'trading_executor.py'
    ]
    
    for file in optional_ml_files:
        src = ml_path / file
        if src.exists():
            shutil.copy2(src, ml_temp_dir)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ⚠ Optional file not found: {file}")
    
    # Copy interactive training app
    training_app_path = ml_path / 'interactive_training_app'
    if training_app_path.exists():
        shutil.copytree(training_app_path, os.path.join(ml_temp_dir, 'interactive_training_app'))
        print("  ✓ Copied interactive_training_app/")
    else:
        print("  ✗ Missing interactive_training_app/")
    
    return True

def create_zip_archive(temp_dir, output_name='deployment.zip'):
    """Create ZIP archive from temporary directory"""
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure forward slashes for zip archive paths
                arcname = os.path.relpath(file_path, temp_dir).replace('\\', '/')
                zipf.write(file_path, arcname)
    
    return os.path.getsize(output_name)

def list_zip_contents(zip_path):
    """List contents of the created ZIP file"""
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        print("\n" + Colors.BLUE + "Deployment package contents:" + Colors.RESET)
        for info in zipf.infolist():
            size_kb = info.file_size / 1024
            print(f"  {info.filename:<50} {size_kb:>8.1f} KB")

def main():
    parser = argparse.ArgumentParser(description='Create Flask deployment package for Elastic Beanstalk')
    parser.add_argument('--output', '-o', default='deployment.zip', 
                       help='Output ZIP file name (default: deployment.zip)')
    parser.add_argument('--include-static', action='store_true',
                       help='Include static files directory in deployment')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep temporary directory for inspection')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--skip-ml-check', action='store_true',
                       help='Skip ML component validation')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Trading Algorithm - Flask EB Deployment")
    print("=" * 50)
    print()
    
    # Check if running from correct directory
    print_colored("[1/8] Checking environment...", Colors.BLUE)
    missing_files, missing_dirs = check_requirements()
    
    if missing_files or missing_dirs:
        print_colored("ERROR: Missing required files/directories:", Colors.RED)
        for file in missing_files:
            print_colored(f"  - {file}", Colors.RED)
        for dir_path in missing_dirs:
            print_colored(f"  - {dir_path}", Colors.RED)
        print_colored("\nPlease run this script from the flask_web directory.", Colors.RED)
        sys.exit(1)
    
    print_colored("✓ All required files found", Colors.GREEN)
    
    # Check ML components
    if not args.skip_ml_check:
        print()
        print_colored("[2/8] Checking ML orchestrator components...", Colors.BLUE)
        ml_available, missing_ml_files, missing_ml_dirs, optional_missing = check_ml_components()
        
        if not ml_available:
            print_colored("ERROR: ML orchestrator directory not found!", Colors.RED)
            print_colored("This deployment will have limited functionality.", Colors.RED)
            print_colored("The deployment will not be able to:", Colors.YELLOW)
            print_colored("- Make trading predictions", Colors.YELLOW)
            print_colored("- Train neural networks", Colors.YELLOW)
            print_colored("- Process user training data", Colors.YELLOW)
            print_colored("- Provide full trading functionality", Colors.YELLOW)
            print()
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Deployment cancelled.")
                sys.exit(1)
        elif missing_ml_files or missing_ml_dirs:
            print_colored("ERROR: Missing critical ML components:", Colors.RED)
            for file in missing_ml_files:
                print_colored(f"  - {file}", Colors.RED)
            for dir_path in missing_ml_dirs:
                print_colored(f"  - {dir_path}", Colors.RED)
            print()
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Deployment cancelled.")
                sys.exit(1)
        else:
            print_colored("✓ All ML components found", Colors.GREEN)
            if optional_missing:
                print_colored("⚠ Optional files missing:", Colors.YELLOW)
                for file in optional_missing:
                    print_colored(f"  - {file}", Colors.YELLOW)
    
    # Clean up existing deployment files
    print()
    print_colored("[3/8] Cleaning up previous deployment files...", Colors.BLUE)
    if os.path.exists(args.output):
        os.remove(args.output)
        print_colored("  Removed old deployment.zip", Colors.YELLOW)
    
    # Create temporary directory
    print()
    print_colored("[4/8] Creating deployment package...", Colors.BLUE)
    temp_dir = tempfile.mkdtemp(prefix='flask_deploy_')
    print(f"  Using temporary directory: {temp_dir}")
    
    # Copy files to temporary directory
    copy_files_to_temp(temp_dir, args.include_static)
    
    # Copy ML components
    print()
    print_colored("[5/8] Copying ML orchestrator components...", Colors.BLUE)
    ml_copied = copy_ml_components(temp_dir)
    
    # Create .ebignore file
    print()
    print_colored("[6/8] Creating .ebignore file...", Colors.BLUE)
    create_ebignore(temp_dir)
    print("  ✓ Created .ebignore")
    
    # Create ZIP archive
    print()
    print_colored("[7/8] Creating ZIP archive...", Colors.BLUE)
    zip_size = create_zip_archive(temp_dir, args.output)
    zip_size_mb = zip_size / (1024 * 1024)
    
    if os.path.exists(args.output):
        print_colored("✓ Deployment package created successfully", Colors.GREEN)
        print(f"  File: {args.output}")
        print(f"  Size: {zip_size_mb:.2f} MB")
    else:
        print_colored("✗ Failed to create deployment package", Colors.RED)
        sys.exit(1)
    
    # Clean up temporary directory
    if not args.no_cleanup:
        shutil.rmtree(temp_dir)
        print("  ✓ Cleaned up temporary directory")
    else:
        print(f"  ⚠ Temporary directory preserved: {temp_dir}")
    
    # Validate and show contents
    print()
    print_colored("[8/8] Validating deployment package...", Colors.BLUE)
    
    if zip_size_mb < 1:
        print_colored("Warning: Deployment package seems very small. Please verify contents.", Colors.YELLOW)
    
    list_zip_contents(args.output)
    
    # Show next steps
    print()
    print_colored("=" * 50, Colors.GREEN)
    print_colored("DEPLOYMENT PACKAGE READY!", Colors.GREEN)
    print_colored("=" * 50, Colors.GREEN)
    print()
    print_colored("Next Steps:", Colors.BLUE)
    print()
    print_colored("1. Go to AWS Elastic Beanstalk Console:", Colors.YELLOW)
    print("   https://console.aws.amazon.com/elasticbeanstalk/")
    print()
    print_colored("2. Create new application or select existing environment", Colors.YELLOW)
    print()
    print_colored("3. Upload deployment.zip file", Colors.YELLOW)
    print()
    print_colored("4. Deploy and wait for completion", Colors.YELLOW)
    print()
    print_colored("Environment Configuration:", Colors.BLUE)
    print("- Platform: Python 3.9")
    print("- Instance type: t3.micro (free tier) or t3.medium")
    print("- Environment type: Single instance (free tier)")
    print()
    print_colored("Files included in deployment.zip:", Colors.BLUE)
    print("✓ flask_app.py (main Flask application)")
    print("✓ wsgi.py (WSGI entry point)")
    print("✓ requirements.txt (Python dependencies)")
    print("✓ Procfile (process definition)")
    print("✓ gunicorn.conf.py (Gunicorn configuration)")
    print("✓ config.py (application configuration)")
    print("✓ .ebextensions/ (Elastic Beanstalk configuration)")
    print("✓ templates/ (HTML templates)")
    print("✓ .ebignore (deployment exclusions)")
    if ml_copied:
        print("✓ _2_Orchestrator_And_ML_Python/ (ML orchestrator)")
        print("✓ model_bridge.py (neural network bridge)")
        print("✓ interactive_training_app/ (training interface)")
        print("✓ stock_selection_utils.py (stock utilities)")
        print("✓ date_range_utils.py (date utilities)")
        print("✓ sp100_symbols.json (stock data)")
    else:
        print_colored("✗ ML components missing - limited functionality", Colors.RED)
    if args.include_static:
        print("✓ static/ (static files)")
    print()
    print_colored("Functionality included:", Colors.BLUE)
    if ml_copied:
        print("✓ Full trading algorithm with neural networks")
        print("✓ Real-time stock data fetching")
        print("✓ User training data collection")
        print("✓ Model training and predictions")
        print("✓ WebSocket support for real-time updates")
        print("✓ Technical analysis and sentiment analysis")
    else:
        print_colored("✗ Basic Flask interface only", Colors.RED)
        print_colored("✗ No ML capabilities", Colors.RED)
        print_colored("✗ Limited trading functionality", Colors.RED)
    print()
    print_colored("Ready for deployment!", Colors.GREEN)
    print()

if __name__ == "__main__":
    main() 