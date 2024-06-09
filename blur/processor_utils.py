"""Processor core for image pre- & post- resize"""

from itertools import product as product
from math import ceil
from typing import Optional

import numpy as np
import torch
from PIL import Image

PROCESSOR_CONFIG = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "loc_weight": 2.0,
    "image_size": 840,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
    # Post-processing
    "confidence": 0.99,
    "nms_threshold": 0.4,
    "top_k": 5000,
    "keep_top_k": 750,
}


class Processor:
    def __init__(self, device: Optional[torch.device] = None):
        self.cfg = PROCESSOR_CONFIG

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

    def pre_process(self, img: Image):
        """
        Pre-process image to the necessary format

        :param img: PIL Image
        :return:
            Scaled image and scale info
        """
        W, H = img.size
        scale = torch.Tensor([W, H, W, H]).to(self.device)
        img = img.resize(
            (self.cfg["image_size"], self.cfg["image_size"]), Image.BILINEAR
        )
        img = torch.tensor(np.array(img), dtype=torch.float32).to(self.device)
        img -= torch.tensor([104, 117, 123]).to(self.device)
        img = img.permute(2, 0, 1)
        return img, scale

    def post_process(self, idx, model_output, scales):
        """
        Post-processing of images to enhance results

        :param idx: image id
        :param model_output: result of self.model(batch)
        :param scales: scales info from `pre-processor`
        :return:
            Boxes with faces on image
        """
        loc, conf, landmarks = model_output
        priors = self.prior_box().to(self.device)
        boxes = self.decode(loc.data[idx], priors)
        boxes = boxes * scales[idx]
        scores = conf[idx][:, 1]

        # Ignore low scores
        index = torch.where(scores > self.cfg["confidence"])[0]
        boxes = boxes[index]
        scores = scores[index]

        # Keep top-K before NMS
        order = scores.argsort(dim=0, descending=True)[: self.cfg["top_k"]]
        boxes = boxes[order]
        scores = scores[order]

        # Do NMS
        keep = self.nms(boxes, scores, self.cfg["nms_threshold"])
        boxes = torch.abs(boxes[keep, :])
        scores = scores[:, None][keep, :]

        # Keep top-K faster NMS
        boxes = boxes[: self.cfg["keep_top_k"], :]
        scores = scores[: self.cfg["keep_top_k"], :]

        return boxes

    def prior_box(self):
        """
        Prior-box realization
        Source: https://github.com/fmassa/object-detection.torch
        """
        steps, all_min_sizes, image_size = (
            self.cfg["steps"],
            self.cfg["min_sizes"],
            self.cfg["image_size"],
        )
        feature_maps = [
            [ceil(image_size / step), ceil(image_size / step)] for step in steps
        ]

        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = all_min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size
                    s_ky = min_size / image_size
                    dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        return output

    def decode(self, loc, priors):
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
    def nms(box, scores, thresh):
        """
        Non-maximum suppression to leave bbox

        :param box: original boxes
        :param scores: scores from model
        :param thresh: MNS threshold
        :return:
            List of bbox to leave
        """
        x1, y1 = box[:, 0], box[:, 1]
        x2, y2 = box[:, 2], box[:, 3]
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
