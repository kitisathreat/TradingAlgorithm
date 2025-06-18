import zipfile
import os
import shutil

EXCLUDES = ['venv', '__pycache__', '.git', 'deployment.zip', 'deployment_fixed.zip']
ZIP_NAME = 'deployment.zip'

# Get the repository root (two levels up from flask_web)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ORCHESTRATOR_PATH = os.path.join(REPO_ROOT, '_2_Orchestrator_And_ML_Python')

print(f"Repository root: {REPO_ROOT}")
print(f"Orchestrator path: {ORCHESTRATOR_PATH}")

# Create a temporary directory for staging
STAGE_DIR = 'deployment_stage'
if os.path.exists(STAGE_DIR):
    shutil.rmtree(STAGE_DIR)
os.makedirs(STAGE_DIR)

# Copy flask_web files to staging
print("Copying Flask web files...")
for root, dirs, files in os.walk('.'):
    # Skip excluded directories
    if any(skip in root for skip in EXCLUDES):
        continue
    
    # Create relative path for staging
    rel_path = os.path.relpath(root, '.')
    stage_path = os.path.join(STAGE_DIR, rel_path)
    
    for file in files:
        if file.endswith('.pyc'):
            continue
        if file in ['deployment.zip', 'deployment_fixed.zip']:
            continue
            
        src_file = os.path.join(root, file)
        dst_file = os.path.join(stage_path, file)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)

# Copy orchestrator modules to staging
print("Copying orchestrator modules...")
if os.path.exists(ORCHESTRATOR_PATH):
    # Copy the entire orchestrator directory
    orchestrator_stage = os.path.join(STAGE_DIR, '_2_Orchestrator_And_ML_Python')
    shutil.copytree(ORCHESTRATOR_PATH, orchestrator_stage, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', 'logs', 'venv', '.git'
    ))
    print(f"Copied orchestrator to: {orchestrator_stage}")
else:
    print(f"WARNING: Orchestrator path not found: {ORCHESTRATOR_PATH}")

# Create the deployment zip
print("Creating deployment zip...")
with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(STAGE_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            # Calculate relative path from staging directory
            arcname = os.path.relpath(filepath, STAGE_DIR)
            # Use forward slashes for all paths in the zip
            zipf.write(filepath, arcname.replace(os.sep, '/'))

# Clean up staging directory
shutil.rmtree(STAGE_DIR)

print(f"Created {ZIP_NAME} with orchestrator modules included.")
print("Deployment package includes:")
print("- Flask web application")
print("- Orchestrator modules (_2_Orchestrator_And_ML_Python)")
print("- Model trainer and stock selection utilities")
print("- All required data files") 