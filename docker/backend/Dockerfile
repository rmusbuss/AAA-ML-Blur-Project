FROM nvcr.io/nvidia/tritonserver:23.04-py3

WORKDIR /model

RUN apt-get update && \
    apt-get install -y --no-install-recommends

COPY docker/backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
EXPOSE 8002
EXPOSE 8003

#CMD ["tritonserver", "--model-repository=/models/blur/backend/triton"]