"""Detector based on RetinaFace"""

from typing import Optional

import torch

from blur.backend.config import TORCH_WEIGHTS
from blur.backend.retinaface.core import RetinaFace
from blur.processor_utils import PROCESSOR_CONFIG, Processor


class FaceDetector:
    """Face Detector model based on RetinaFace"""

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.processor = Processor(device=self.device)
        self.model = RetinaFace(cfg=PROCESSOR_CONFIG)
        self.model.load_state_dict(torch.load(TORCH_WEIGHTS))
        self.model = self.model.to(self.device)

    def detect(self, images: list):
        """
        Entry point for prediction

        :param images: list of PIL images
        :return:
            List of predictions [boxes]
        """
        batch_size = len(images)
        batch, scales = [], []

        for image in images:
            image, scale = self.processor.pre_process(image)
            batch.append(image)
            scales.append(scale)

        batch = torch.stack(batch)
        scales = torch.stack(scales)

        with torch.no_grad():
            model_output = self.model(batch)

        output = []
        for idx in range(batch_size):
            boxes = self.processor.post_process(idx, model_output, scales)
            output.append(boxes)

        return output
