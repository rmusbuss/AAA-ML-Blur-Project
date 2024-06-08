"""Tests for Haar Cascade model"""

import cv2
import numpy as np
import pytest
from blur.backend.config import (
    MIN_NEIGHBORS,
    MIN_SIZE,
    SCALE_FACTOR,
    VALID_IMAGE_PATH,
)
from blur.backend.cascade.cascade import Cascade


@pytest.fixture
def cascade():
    return Cascade()


def test_init_with_default_params(cascade):
    assert cascade.predict_params["scaleFactor"] == SCALE_FACTOR
    assert cascade.predict_params["minNeighbors"] == MIN_NEIGHBORS
    assert cascade.predict_params["minSize"] == MIN_SIZE


def test_init_with_custom_params():
    custom_params = {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)}
    cascade = Cascade(predict_params=custom_params)
    assert cascade.predict_params["scaleFactor"] == 1.1
    assert cascade.predict_params["minNeighbors"] == 5
    assert cascade.predict_params["minSize"] == (30, 30)


def test_repr(cascade):
    expected_repr = (
        f"Cascade model with predict params = {cascade.predict_params}"
    )
    assert repr(cascade) == expected_repr


def test_predict_with_valid_input(cascade):
    # 100x100 белый квадрат на черном фоне
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (30, 30), (70, 70), (255, 255, 255), -1)

    images = [test_image]

    predictions = cascade.predict(images)
    assert isinstance(predictions, list)
    assert len(predictions) == 0


def test_predict_with_invalid_input_dimensions(cascade):
    invalid_image = np.zeros((100, 100), dtype=np.uint8)
    images = [invalid_image]

    with pytest.raises(AssertionError):
        cascade.predict(images)


def test_predict_with_idx(cascade):
    # 100x100 белый квадрат на черном фоне
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (30, 30), (70, 70), (255, 255, 255), -1)

    images = [test_image]
    idx = np.array([0])

    predictions = cascade.predict(images, idx=idx)
    assert isinstance(predictions, list)
    assert len(predictions) == 0


def test_predict_with_invalid_idx(cascade):
    # 100x100 белый квадрат на черном фоне
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (30, 30), (70, 70), (255, 255, 255), -1)

    images = [test_image]
    invalid_idx = np.array([[0]])  # Неверная форма

    with pytest.raises(AssertionError):
        cascade.predict(images, idx=invalid_idx)


def test_predict_with_valid_image(cascade):
    # Загрузка валидного изображения с лицом
    valid_image = cv2.imread(str(VALID_IMAGE_PATH))

    images = [valid_image]

    predictions = cascade.predict(images)
    assert isinstance(predictions, list)
    assert len(predictions) == 1

    prediction = predictions[0]

    assert prediction["label"] == "face"
    assert prediction["width"] == 700
    assert prediction["height"] == 700


if __name__ == "__main__":
    pytest.main()
