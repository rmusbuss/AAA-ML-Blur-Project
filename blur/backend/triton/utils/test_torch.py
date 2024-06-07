import torch
import torchvision.models as models
import time
import numpy as np

# Загрузка модели
model = models.resnet18(pretrained=True).to(device='cuda')
model.eval()

# Пример входных данных
dummy_input = torch.randn(1, 3, 224, 224).to(device='cuda')

times = []
for i in range(300):
    a = time.perf_counter()
    model(dummy_input)
    b = time.perf_counter()
    times.append(b-a)

times = np.array(times)
print(f'mean:{times.mean()=}')