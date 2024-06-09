import pytest
import torch
from PIL import Image

from blur.backend.config import VALID_IMAGE_PATH
from blur.processor_utils import Processor


@pytest.fixture
def processor():
    return Processor(
        device=torch.device("cpu"),
    )


@pytest.fixture()
def img():
    return Image.open(VALID_IMAGE_PATH)


def test_pre_processor(processor, img):
    processed_img, scale = processor.pre_process(img)

    assert isinstance(processed_img, torch.Tensor)
    assert processed_img.shape == (
        3,
        processor.cfg["image_size"],
        processor.cfg["image_size"],
    )
    assert isinstance(scale, torch.Tensor)
    assert scale.shape == (4,)


if __name__ == "__main__":
    pytest.main()
