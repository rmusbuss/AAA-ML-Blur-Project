FROM python:3.10-slim

WORKDIR /app

# Dependencies
COPY blur/frontend/requirements.txt .
RUN python -m pip install -r requirements.txt

RUN mkdir -p /app/blur/backend/retinaface && \
    mkdir -p /app/blur/frontend

COPY blur/frontend /app/blur/frontend/

COPY blur/processor_utils.py /app/blur/

COPY setup.py .

RUN python -m pip install -e .

EXPOSE 8080

CMD ["python", "/app/blur/frontend/run.py"]
