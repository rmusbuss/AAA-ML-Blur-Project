# Базовый образ Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:23.04-py3

# Установка PyTorch и других необходимых библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir torch torchvision torchaudio

# Убедитесь, что необходимые папки созданы
RUN mkdir -p /models

# Установите рабочий каталог
WORKDIR /models

# Запуск Triton Inference Server по умолчанию
CMD ["tritonserver", "--model-repository=/models"]
