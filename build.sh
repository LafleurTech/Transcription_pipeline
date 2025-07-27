#!/bin/bash

git pull
git submodule update --init --recursive

docker build -t transcription_pipeline:latest .
# docker-compose build

docker-compose up -d
