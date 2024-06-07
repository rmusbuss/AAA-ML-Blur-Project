import torch
import torchvision.models as models

# Загрузка модели
model = models.resnet18(pretrained=True)
model.eval()

# Пример входных данных
dummy_input = torch.randn(1, 3, 224, 224)

# Экспорт в ONNX
# onnx_program = torch.onnx.dynamo_export(model, dummy_input)
# onnx_program.save("model.onnx")

torch.onnx.export(model, dummy_input, 
                  "model.onnx", input_names=['input'], 
                  output_names=['output'], opset_version=11,
                  export_params=True,
                  dynamic_axes={'input': {0: 'batch_size', 1: 'height', 2: 'width'},    # variable length axes
                                'output': {0: 'batch_size', 1: 'height', 2: 'width'}}
                  
                  )
