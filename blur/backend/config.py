from pathlib import Path

# Paths
file_path = Path(__file__).resolve()
PROJECT_ROOT = file_path.parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"
DOCKER_PATH = PROJECT_ROOT / "docker"
WIDER_FACE_PATH = DATA_PATH / "WIDER_FACE"
CUSTOM_FACE_PATH = DATA_PATH / "CUSTOM_FACE"

# Constants
SEED = 42
