sudo docker build -t tritonserver_with_pytorch .
sudo docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models:/models \
    tritonserver_with_pytorch