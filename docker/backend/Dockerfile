FROM nvcr.io/nvidia/tritonserver:23.04-py3

WORKDIR /model

RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get install git-lfs

RUN git lfs install && \
    pip install poetry==1.8.2

COPY poetry.lock /model

COPY pyproject.toml /model

RUN poetry install

#COPY . .

EXPOSE 8001

#CMD ["tritonserver", "--model-repository=/models"]