build:
    #!/bin/bash
	set -e
	git submodule update --init --recursive
	docker build -t my-app .
