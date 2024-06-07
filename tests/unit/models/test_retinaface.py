"""Tests for RetinaFace detector model"""

import pytest
import torch
from blur.backend.config import (
    CFG,
    CONFIDENCE,
    KEEP_TOP_K,
    NMS_THRESHOLD,
    TOP_K,
    VALID_IMAGE_PATH,
)
from blur.backend.models import FaceDetector
from PIL import Image


@pytest.fixture
def face_detector():
    return FaceDetector(
        cfg=CFG,
        device=torch.device("cpu"),
        confidence_threshold=CONFIDENCE,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
        keep_top_k=KEEP_TOP_K,
    )


@pytest.fixture()
def img():
    return Image.open(VALID_IMAGE_PATH)


def test_initialization(face_detector):
    assert face_detector.cfg == CFG
    assert face_detector.device == torch.device("cpu")
    assert face_detector.confidence_threshold == CONFIDENCE
    assert face_detector.nms_thresh == NMS_THRESHOLD
    assert face_detector.top_k == TOP_K
    assert face_detector.keep_top_k == KEEP_TOP_K
    assert face_detector.model is not None


def test_pre_processor(face_detector, img):
    processed_img, scale = face_detector.pre_processor(img)

    assert isinstance(processed_img, torch.Tensor)
    assert processed_img.shape == (
        3,
        face_detector.cfg["image_size"],
        face_detector.cfg["image_size"],
    )
    assert isinstance(scale, torch.Tensor)
    assert scale.shape == (4,)


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
