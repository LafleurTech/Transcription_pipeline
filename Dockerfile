
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git config --global --add safe.directory /app
# RUN git submodule update --init --recursive

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1
