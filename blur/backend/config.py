"""All backend constants"""

from pathlib import Path

# Paths
file_path = Path(__file__).resolve()
PROJECT_ROOT = file_path.parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"
DOCKER_PATH = PROJECT_ROOT / "docker"
WIDER_FACE_PATH = DATA_PATH / "WIDER_FACE"
CUSTOM_FACE_PATH = DATA_PATH / "CUSTOM_FACE"
CASCADE_XML = "haarcascade_frontalface_default.xml"
TORCH_WEIGHTS = DATA_PATH / "resnet50.pth"
VALID_IMAGE_PATH = DATA_PATH / "valid_image.jpg"

# Other constants
SEED = 42
SCALE_FACTOR = 1.21
MIN_NEIGHBORS = 9
MIN_SIZE = (34, 54)
