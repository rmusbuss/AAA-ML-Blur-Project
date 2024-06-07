"""Models Core"""

from itertools import product
from math import ceil

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange

from blur.backend.config import (
    CASCADE_XML,
    CONFIDENCE,
    KEEP_TOP_K,
    NMS_THRESHOLD,
    TOP_K,
    TORCH_WEIGHTS,
    SCALE_FACTOR,
    MIN_NEIGHBORS,
    MIN_SIZE,
)
from blur.backend.retinaface.core import RetinaFace


class Cascade:
    """Haar Cascade using OpenCV"""

    def __init__(self, predict_params: dict | None = None):

        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_XML)

        self.predict_params = {
            "scaleFactor": SCALE_FACTOR,
            "minNeighbors": MIN_NEIGHBORS,
            "minSize": MIN_SIZE,
        }

        if predict_params is not None:
            self.predict_params.update(predict_params)

    def __repr__(self):
        return f"Cascade model with predict params = {self.predict_params}"

    def predict(
        self,
        images: list[np.ndarray],
        idx: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Find faces

        :param images: list on images in np.ndarray
        :param idx: image idx (optional)
        :return:
            List of faces info for all images
        """

        assert (images[0].ndim == 3) and (images[0].shape[2] == 3)
        batch_size = len(images)
        predictions = []
        for img_id in range(batch_size):
            image = images[img_id]

            if idx is not None:
                assert idx.ndim == 1
                img_id = idx[img_id]

            faces = self.model.detectMultiScale(
                image,
                **self.predict_params,
            )

            for (x, y, w, h) in faces:
                predictions.append(
                    {
                        "img_id": img_id,
                        "label": "face",
                        "confidence": 1.0,
                        "xmin": x,
                        "xmax": x + w,
                        "ymin": y,
                        "ymax": y + h,
                        "width": w,
                        "height": h,
                    }
                )

        return predictions


class FaceDetector:
    """Face Detector model based on RetinaFace"""

    def __init__(
        self,
        cfg: dict,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        confidence_threshold: float = CONFIDENCE,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
        keep_top_k: int = KEEP_TOP_K,
    ):
        self.cfg = cfg
        self.model = RetinaFace(cfg=self.cfg, phase="eval")
        self.model.load_state_dict(torch.load(TORCH_WEIGHTS))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

        self.confidence_threshold = confidence_threshold
        self.nms_thresh = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k

    def pre_processor(self, img: Image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-process image to the necessary format

        :param img: PIL Image
        :return:
            Scaled image and scale info
        """

        width, height = img.size
        scale = torch.Tensor([width, height, width, height]).to(self.device)
        img = img.resize(
            (self.cfg["image_size"], self.cfg["image_size"]), Image.BILINEAR
        )
        img = torch.tensor(np.array(img), dtype=torch.float32).to(self.device)
        img -= torch.tensor([104, 117, 123]).to(self.device)
        img = img.permute(2, 0, 1)
        return img, scale

    def detect(self, images: list) -> list:
        """
        Entry point for prediction

        :param images: list of PIL images
        :return:
            List of predictions [boxes]
        """

        batch_size = len(images)
        batch, scales = [], []

        for image in images:
            image, scale = self.pre_processor(image)
            batch.append(image)
            scales.append(scale)

        batch = torch.stack(batch)
        scales = torch.stack(scales)

        with torch.no_grad():
            model_output = self.model(batch)

        output = []
        for idx in trange(batch_size):
            boxes = self.post_processor(idx, model_output, scales)
            output.append(boxes)

        return output

    def post_processor(
        self,
        idx: int,
        model_output: tuple,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Post-processing of images to enhance results

        :param idx: image id
        :param model_output: result of self.model(batch)
        :param scales: scales info from `pre-processor`
        :return:
            Boxes with faces on image
        """
        loc, conf, _ = model_output
        priors = self.prior_box().to(self.device)
        boxes = self.decode(loc.data[idx], priors)
        boxes = boxes * scales[idx]
        scores = conf[idx][:, 1]

        # Ignore low scores
        index = torch.where(scores > self.confidence_threshold)[0]
        boxes = boxes[index]
        scores = scores[index]

        # Keep top-K before NMS
        order = scores.argsort(dim=0, descending=True)[: self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # Do NMS
        keep = self.nms(boxes, scores, self.nms_thresh)
        boxes = torch.abs(boxes[keep, :])
        scores = scores[:, None][keep, :]

        # Keep top-K faster NMS
        boxes = boxes[: self.keep_top_k, :]
        scores = scores[: self.keep_top_k, :]

        return boxes

    def prior_box(self):
        """
        Prior-box realization
        Source: https://github.com/fmassa/object-detection.torch
        """
        image_size = (self.cfg["image_size"], self.cfg["image_size"])
        steps = self.cfg["steps"]
        feature_maps = [
            [ceil(image_size[0] / step), ceil(image_size[1] / step)]
            for step in steps
        ]
        min_sizes_ = self.cfg["min_sizes"]
        anchors = []

        for k, f in enumerate(feature_maps):
            min_sizes = min_sizes_[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        return output

    def decode(self, loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """
        Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Source: https://github.com/Hakuyume/chainer-ssd

        :param loc: locations
        :param priors: result of self.prior_box()
        :return:
            Decoded boxes
        """
        variances = self.cfg["variance"]
        boxes = torch.cat(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def nms(
        box: torch.Tensor, scores: torch.Tensor, thresh: float
    ) -> list[torch.Tensor]:
        """
        Non-maximum suppression to leave bbox

        :param box: original boxes
        :param scores: scores from model
        :param thresh: MNS threshold
        :return:
            List of bbox to leave
        """
        x1 = box[:, 0]
        y1 = box[:, 1]
        x2 = box[:, 2]
        y2 = box[:, 3]
        zero = torch.tensor([0.0]).to(scores.device)

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort(descending=True)

        keep = []
        while order.shape[0] > 0:
            i = order[0]
            keep.append(i)
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.max(zero, xx2 - xx1 + 1)
            h = torch.max(zero, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = torch.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
