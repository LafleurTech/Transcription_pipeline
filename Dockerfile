

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

ENTRYPOINT ["python", "main.py"]
