import zipfile

with zipfile.ZipFile('deployment.zip', 'r') as z:
    files = z.namelist()
    print(f"Total files: {len(files)}")
    print("\nFirst 20 files:")
    for i, f in enumerate(files[:20]):
        print(f"{i+1:2d}. {f}")
    
    print("\nFiles containing 'orchestrator':")
    orchestrator_files = [f for f in files if 'orchestrator' in f.lower()]
    for f in orchestrator_files:
        print(f"  {f}")
    
    print("\nFiles containing '_2_':")
    orchestrator_files = [f for f in files if '_2_' in f]
    for f in orchestrator_files:
        print(f"  {f}") 