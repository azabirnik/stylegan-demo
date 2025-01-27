# Makefile for Style Transfer Inference API using Docker Compose

# Variables
COMPOSE_FILE=docker-compose.yaml
IMAGE_NAME=style-transfer-api
CONTAINER_NAME=style-transfer-api-container
HOST_PORT=2026
CONTAINER_PORT=8000
SAMPLES_DIR=./samples
OUTPUT_IMAGE=stylized_output.png

.PHONY: help docker-build docker-run docker-stop docker-remove test-inference

## @HELP: Display this help message
help:
	@echo "Makefile for Style Transfer Inference API using Docker Compose"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## @DOCKER-BUILD: Build the Docker images using Docker Compose
docker-build: ## Build the Docker images
	docker-compose build

## @DOCKER-RUN: Run the Docker containers using Docker Compose
docker-run: ## Run the Docker containers
	docker-compose up -d

## @DOCKER-STOP: Stop the running Docker containers
docker-stop: ## Stop the Docker containers
	docker-compose down

## @DOCKER-REMOVE: Remove Docker containers and volumes
docker-remove: ## Remove the Docker containers and associated volumes
	docker-compose down --volumes

## @TEST-INFERENCE: Perform a test inference using sample images
test-inference: ## Perform a test inference using sample images
	@if [ ! -f "$(SAMPLES_DIR)/content.jpg" ] || [ ! -f "$(SAMPLES_DIR)/style.jpg" ]; then \
		echo "Please ensure 'content.jpg' and 'style.jpg' exist in the $(SAMPLES_DIR) directory."; \
		exit 1; \
	fi
	@echo "Performing test inference..."
	docker-compose run --rm test-inference
	@echo "Stylized image saved to stylized_output/stylized_output.png"

