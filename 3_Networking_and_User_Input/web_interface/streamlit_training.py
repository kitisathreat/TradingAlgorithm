# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH)) 