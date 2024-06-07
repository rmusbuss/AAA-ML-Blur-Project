FROM nvcr.io/nvidia/tritonserver:23.04-py3

RUN mkdir -p /models

WORKDIR /models

EXPOSE 8001 8002 8003

CMD ["tritonserver", "--model-repository=/models"]
