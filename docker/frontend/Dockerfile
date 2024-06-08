FROM python:3.10-slim

WORKDIR /app

# Dependencies
COPY blur/frontend/requirements.txt .
RUN python -m pip install -r requirements.txt

RUN mkdir -p /app/blur/backend/retinaface && \
    mkdir -p /app/blur/frontend

# Frontend part
COPY blur/frontend /app

# Backend part
COPY blur/backend/retinaface /app/blur/backend/retinaface
COPY blur/backend/config.py /app/blur/backend/

COPY setup.py .
RUN python -m pip install -e .

EXPOSE 8080

CMD ["python", "run.py"]

