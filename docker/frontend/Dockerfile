FROM python:3.10-slim

WORKDIR /app

COPY blur/frontend/requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY blur/frontend /app

EXPOSE 8080

CMD ["python", "run.py"]
