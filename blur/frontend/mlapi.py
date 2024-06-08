import sys
sys.path.append('/app/')
sys.path.append('/home/yaroslav/progs/final_ml_fin_proj/AAA-ML-Blur-Project')

import numpy as np
from PIL import Image
import torch
import tritonclient.grpc as grpcclient

from blur.backend.config import CFG
from blur.backend.retinaface.detector import FaceDetector


class MLApi:
    def __init__(self, url: str):
        self.triton_client = grpcclient.InferenceServerClient(url=f'{url}:8001')
        self.facedet = FaceDetector(cfg=CFG, is_infer=True)

    def convert_bbox(self, x, y, width, height):
        x_min = x
        y_min = y
        x_max = x + width
        y_max = y + height
        return x_min, y_min, x_max, y_max

    def blur_image(self, image: Image, coords, sigma=52, is_coords=False):
        if not is_coords:
            x, y, width, height = coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]
            x_min, y_min, x_max, y_max = self.convert_bbox(x, y, width, height)
        else:
            x_min, y_min, x_max, y_max = coords

        mask = np.zeros_like(np.array(image), dtype=np.uint8)  # initialize mask
        mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255  # fill with white pixels
        image_np = np.array(image)
        mask_np = mask
        # Создание пустого массива для размытого изображения
        blurred_image_np = np.zeros_like(image_np)

        # Применение размытия к изображению с учетом маски
        for i in range(3):  # для каждого канала R, G, B
            blurred_channel_np = np.array(Image.fromarray(image_np[:, :, i]).filter(ImageFilter.GaussianBlur(sigma)))
            blurred_image_np[:, :, i] = np.where(mask[:, :, i] == 255, blurred_channel_np, image_np[:, :, i])

        # Преобразование numpy массива обратно в изображение
        blurred_image = Image.fromarray(blurred_image_np)
        return blurred_image

    def run_model(self, image: Image):
        # preprocessing
        img, scale = self.facedet.pre_processor(image)

        # preprocessing for sending to triton
        input_data = img.cpu().numpy()
        input_data = input_data.reshape([-1] + list(input_data.shape))
        inputs = [grpcclient.InferInput('INPUT', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput('OUTPUT0'), grpcclient.InferRequestedOutput('OUTPUT1'),
                   grpcclient.InferRequestedOutput('OUTPUT2')]

        # getting result from triton
        print('sending')
        result = self.triton_client.infer(model_name='resnet18', inputs=inputs, outputs=outputs)
        print('got result')
        loc = result.as_numpy('OUTPUT0')
        conf = result.as_numpy('OUTPUT1')
        landmarks = result.as_numpy('OUTPUT2')

        loc = torch.tensor(loc).to('cuda')
        conf = torch.tensor(conf).to('cuda')

        for bbox in self.facedet.post_processor(0, loc, conf, landmarks, torch.stack([scale])).cpu():
            coords = [x.item() for x in bbox]
            print(coords)
            image = self.blur_image(image, coords)

        return image
