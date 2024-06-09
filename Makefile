## Install Python dependencies
install:
	@echo "Installing python dependencies..."
	sudo apt-get install git-lfs
	git lfs install
	git lfs pull
	pip install poetry==1.8.2
	poetry install

## Activate virtual environment
activate:
	@echo "Activating virtual environment..."
	poetry shell

## Create JH-kernel
kernel:
	@echo "Creating JH-kernel..."
	poetry run ipython kernel install --user --name=blur
	poetry run jupyter lab

## Setup project
setup: install activate

## Pylint backend
pylint:
	pylint blur/backend

## Flake8 backend
flake8:
	flake8 blur/backend

## Lint code
lint: pylint flake8

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v -s

## Run tests
tests: test

## Build containers
build:
	docker-compose up -d

## Down containers
down:
	docker-compose down

## Build backend
backend-build:
	docker build \
	-f docker/backend/Dockerfile \
	-t blur-backend:0.1.0 .

## Run backend
backend-run:
	docker run \
	-p 8001:8001 -p 8002:8002 -p 8003:8003 \
	-v ./ml_server_triton/models:/models \
	-it blur-backend:0.1.0 bash

## Build frontend
frontend-build:
	docker build \
	-f docker/frontend/Dockerfile \
	-t blur-frontend:0.1.0 .
	
## Run frontend
frontend-run:
	docker run -p 8080:8080 -it blur-frontend:0.1.0 bash

## Show help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=10 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
