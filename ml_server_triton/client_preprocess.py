import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
from PIL import Image
import time

# Загрузка клиента
triton_client = grpcclient.InferenceServerClient(url='0.0.0.0:8001')

class TritonInferCLient:
    def __init__(self):
        self.img = None
        self.scale = None

    def run_preprocess_model(self, img_name='test3'):
        input_data = np.array(Image.open(f'test_images/{img_name}.jpg')).astype(np.uint8)  # Измените размерность по мере необходимости
        input_data = input_data.reshape([-1]+list(input_data.shape))

        inputs = [grpcclient.InferInput('INPUT', input_data.shape, 'UINT8')]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput('OUTPUT0'), grpcclient.InferRequestedOutput('OUTPUT1')]

        # Send request
        print('send request for result 1')
        a = time.perf_counter()
        result = triton_client.infer(model_name='preprocess_model', inputs=inputs, outputs=outputs)
        print(f'result 1 got: {time.perf_counter() - a}')


        self.img = result.as_numpy('OUTPUT0')
        self.scale = result.as_numpy('OUTPUT1')

    def run_facerec_model(self):
        ############################################### MODEL ###############################################33
        input_data = self.img
        input_data = input_data.reshape([-1]+list(input_data.shape))

        inputs = [grpcclient.InferInput('INPUT', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [grpcclient.InferRequestedOutput('OUTPUT0'), grpcclient.InferRequestedOutput('OUTPUT1'), grpcclient.InferRequestedOutput('OUTPUT2')]
        print('send request for result 2')
        a = time.perf_counter()
        result = triton_client.infer(model_name='resnet18', inputs=inputs, outputs=outputs)
        print(f'result 2 got: {time.perf_counter() - a}')

        self.loc = result.as_numpy('OUTPUT0')
        self.conf = result.as_numpy('OUTPUT1')
        self.landmarks = result.as_numpy('OUTPUT2')  
        self.img = None
        self.scale = None

    def run_pipeline(self):
        self.run_preprocess_model()
        self.run_facerec_model()