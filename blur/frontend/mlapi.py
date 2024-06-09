"""MLApi to work with Triton server"""

import numpy as np
import torch
import tritonclient.grpc as grpcclient
from PIL import Image, ImageFilter

from blur.frontend.config import TRITON_PORT, TRITON_URL
from blur.processor_utils import Processor


class MLApi:
    def __init__(self):
        self.triton_client = grpcclient.InferenceServerClient(
            url=f"{TRITON_URL}:{TRITON_PORT}"
        )
        self.processor = Processor()

    @staticmethod
    def convert_bbox(x, y, width, height):
        """Convert COCO to Pascal VOC"""

        x_min, y_min = x, y
        x_max, y_max = x + width, y + height
        return x_min, y_min, x_max, y_max

    def blur_image(
        self,
        image: Image,
        coords: list,
        sigma: int = 52,
        is_pascal_voc: bool = False,
    ):
        """
        Blur image by coordinates

        :param image: PIL image
        :param coords: coordinated to blur
        :param sigma: level of blurring
        :param is_pascal_voc: if coordinates in Pascal VOC notation
        """
        if is_pascal_voc:
            x_min, y_min, x_max, y_max = coords
        else:
            x, y, width, height = (
                coords[0],
                coords[1],
                coords[2] - coords[0],
                coords[3] - coords[1],
            )
            x_min, y_min, x_max, y_max = self.convert_bbox(x, y, width, height)

        mask = np.zeros_like(np.array(image), dtype=np.uint8)
        mask[int(y_min) : int(y_max), int(x_min) : int(x_max)] = 255

        image_np = np.array(image)
        blurred_image_np = np.zeros_like(image_np)

        # Blurring for R, G, B
        for channel_id in range(3):
            blurred_image = Image.fromarray(image_np[:, :, channel_id]).filter(
                ImageFilter.GaussianBlur(sigma)
            )
            blurred_channel_np = np.array(blurred_image)
            blurred_image_np[:, :, channel_id] = np.where(
                mask[:, :, channel_id] == 255,
                blurred_channel_np,
                image_np[:, :, channel_id],
            )

        blurred_image = Image.fromarray(blurred_image_np)
        return blurred_image

    def run_model(self, image: Image):
        # Image Preprocessing
        img, scale = self.processor.pre_process(image)
        scale = torch.stack([scale])

        # Preprocessing for Triton
        input_data = img.cpu().numpy()
        input_data = input_data.reshape([-1] + list(input_data.shape))
        inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT0"),
            grpcclient.InferRequestedOutput("OUTPUT1"),
            grpcclient.InferRequestedOutput("OUTPUT2"),
        ]

        result = self.triton_client.infer(
            model_name="resnet18", inputs=inputs, outputs=outputs
        )

        loc = torch.tensor(result.as_numpy("OUTPUT0")).to("cuda")
        conf = torch.tensor(result.as_numpy("OUTPUT1")).to("cuda")
        landmarks = result.as_numpy("OUTPUT2")
        model_output = [loc, conf, landmarks]

        # Image Postprocessing
        for bbox in self.processor.post_process(0, model_output, scale).cpu():
            coords = [x.item() for x in bbox]
            image = self.blur_image(image, coords)

        return image
