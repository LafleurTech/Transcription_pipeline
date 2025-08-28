FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY requirements.txt ./

# trust repo path for git, install uv, then install deps
RUN git config --global --add safe.directory /app \
    && curl -LsSf https://astral.sh/uv/0.1.16/install.sh | sh \
    && uv pip install --system --no-cache -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 portaudio19-dev libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /usr/local /usr/local

COPY app/ ./app/
COPY config/ ./config/
COPY interface/ ./interface/
COPY lib/ ./lib/
COPY models/ ./models/
COPY Src/ ./Src/

COPY ffmpeg-normalize/ ./ffmpeg-normalize/
COPY NeMo/ ./NeMo/

COPY main.py ./

ENV PYTHONPATH="/app"

EXPOSE 8000 8501

# CMD ["python", "main.py"]
# ENTRYPOINT ["python3", "main.py"]
