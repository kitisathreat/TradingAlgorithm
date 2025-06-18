import zipfile
import os

EXCLUDES = ['venv', '__pycache__', '.git']
ZIP_NAME = 'deployment.zip'

with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('.'):
        # Skip excluded directories
        if any(skip in root for skip in EXCLUDES):
            continue
        for file in files:
            if file.endswith('.pyc'):
                continue
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, '.')
            # Use forward slashes for all paths in the zip
            zipf.write(filepath, arcname.replace(os.sep, '/'))
print(f"Created {ZIP_NAME} with forward slashes for EB deployment.") 