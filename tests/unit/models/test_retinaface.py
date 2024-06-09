"""Tests for RetinaFace detector model"""

from PIL import Image

import pytest
import torch

from blur.backend.config import VALID_IMAGE_PATH
from blur.backend.retinaface.detector import FaceDetector


@pytest.fixture
def face_detector():
    return FaceDetector(
        device=torch.device("cpu"),
    )


@pytest.fixture()
def img():
    return Image.open(VALID_IMAGE_PATH)


def test_initialization(face_detector):
    assert face_detector.device == torch.device("cpu")
    assert face_detector.model is not None


def test_detect(face_detector, img):
    results = face_detector.detect([img])

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], torch.Tensor)


def test_batch_processing(face_detector, img):
    batch_images = [img, img]
    results = face_detector.detect(batch_images)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, torch.Tensor) for result in results)


if __name__ == "__main__":
    pytest.main()
