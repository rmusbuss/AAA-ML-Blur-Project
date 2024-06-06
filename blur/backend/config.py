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

# Other constants
SEED = 42
CONFIDENCE = 0.99
NMS_THRESHOLD = 0.4
TOP_K = 5000
KEEP_TOP_K = 750

# Model config
CFG = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}
