FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 portaudio19-dev libportaudio2 \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure both cargo bin (uv install destination) and local bin are on PATH
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

ENV UV_SYSTEM_PYTHON=1

FROM base AS build
WORKDIR /app

COPY requirements.txt ./

RUN git config --global --add safe.directory /app \
    && curl -LsSf https://astral.sh/uv/0.1.16/install.sh | sh \
    && mv /root/.cargo/bin/uv /usr/local/bin/uv \
    && uv pip install --system --no-cache -r requirements.txt

RUN source $HOME/.cargo/env

FROM base AS runtime
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
