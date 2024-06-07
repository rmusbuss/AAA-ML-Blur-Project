"""All backend constants"""

from pathlib import Path

# Paths
file_path = Path(__file__).resolve()

ROOT = file_path.parent

TEMPLATES_PATH = ROOT / "templates"
STATIC_PATH = ROOT / "static"

# Constants
PORT = 8080

