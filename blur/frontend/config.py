"""All frontend constants"""

from pathlib import Path

# Paths
file_path = Path(__file__).resolve()

ROOT = file_path.parent

TEMPLATES_PATH = ROOT / "templates"
STATIC_PATH = ROOT / "static"

TRITON_URL = "aaa-ml-blur-project_blur-backend_1"
TRITON_PORT = 8001

# Constants
PORT = 8080
