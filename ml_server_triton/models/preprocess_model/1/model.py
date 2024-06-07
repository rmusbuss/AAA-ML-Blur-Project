import numpy as np
from itertools import product as product
from math import ceil
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from PIL import Image

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
CONFIDENCE = 0.99
NMS_THRESHOLD = 0.4
TOP_K = 5000
KEEP_TOP_K = 750

class FaceDetector:
    def __init__(
        self,
        cfg: dict,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        confidence_threshold: float = CONFIDENCE,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
        keep_top_k: int = KEEP_TOP_K,
        is_infer=False
    ):
        """RetinaFace Detector with 5points landmarks"""
        RetinaFace = None
        TORCH_WEIGHTS = None

        self.cfg = cfg
        self.is_infer = is_infer
        self.device = device

        if not self.is_infer:
            self.model = RetinaFace(cfg=self.cfg)
            self.model.load_state_dict(torch.load(TORCH_WEIGHTS))
            self.model = self.model.to(self.device)
        else:
            self.model = None

        self.confidence_threshold = confidence_threshold
        self.nms_thresh = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k

    def pre_processor(self, img):
        """Process image to the necessary format"""
        W, H = img.size
        scale = torch.Tensor([W, H, W, H]).to(self.device)
        img = img.resize(
            (self.cfg["image_size"], self.cfg["image_size"]), Image.BILINEAR
        )
        img = torch.tensor(np.array(img), dtype=torch.float32).to(self.device)
        img -= torch.tensor([104, 117, 123]).to(self.device)
        img = img.permute(2, 0, 1)
        return img, scale

    def post_processor(self, idx, loc, conf, landmarks, scales):
        """Post processing of images to enhance results"""
        priors = self.prior_box(
            image_size=(self.cfg["image_size"], self.cfg["image_size"]),
        ).to(self.device)
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

    def detect(self, images: list):
        """Entry point for prediction"""
        batch_size = len(images)
        batch, scales = [], []

        for image in images:
            image, scale = self.pre_processor(image)
            batch.append(image)
            scales.append(scale)

        batch = torch.stack(batch)
        scales = torch.stack(scales)

        with torch.no_grad():
            loc, conf, landmarks = self.model(batch)

        output = []
        for idx in trange(batch_size):
            boxes = self.post_processor(idx, loc, conf, landmarks, scales)
            output.append(boxes)

        return output
    
    def prior_box(self, image_size=None):
        """
        Prior box realization
        
        Source: https://github.com/fmassa/object-detection.torch
        """
        
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

    def decode(self, loc, priors):
        """
        Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        
        Source: https://github.com/Hakuyume/chainer-ssd
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
        """Non maximum suppression"""
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



class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'INPUT',
            'data_type': 'TYPE_UINT8',
            'dims': [-1, -1, 3],
            # this parameter will set `INPUT0 as an optional input`
            'optional': True
        }]
        outputs = [{
            'name': 'OUTPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [3, 840, 840]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [-1]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(32)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        self.detector = FaceDetector(cfg=CFG, is_infer=True)

        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them.
        # Reusing the same pb_utils.InferenceResponse object for multiple
        # requests may result in segmentation faults. You should avoid storing
        # any of the input Tensors in the class attributes as they will be
        # overridden in subsequent inference requests. You can make a copy of
        # the underlying NumPy array and store it if it is required.
        for request in requests:
            # Perform inference on the request and append it to responses
            # list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT")

            out_0 = self.detector.pre_processor(Image.fromarray((in_0.as_numpy()[0]).astype(np.uint8)))

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0[0].cpu().numpy())
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_0[1].cpu().numpy())

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)


        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
