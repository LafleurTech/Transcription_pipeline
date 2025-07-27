FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 build-essential git portaudio19-dev libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git config --global --add safe.directory /app
# RUN git submodule update --init --recursive

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000 8501

# ENTRYPOINT ["python3", "main.py"]
