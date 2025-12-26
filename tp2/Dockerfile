FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc 

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

RUN uv sync

COPY main.py .

COPY app.py .

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

RUN ["uv", "run", "main.py"]

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
