build:
    #!/bin/bash
	set -e
	git submodule update --init --recursive
	docker build -t transcription_pipeline:latest .
	
# docker-compose up
