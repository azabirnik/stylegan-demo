# Makefile for building and running frontend and backend services

.PHONY: help frontend frontend-build frontend-run frontend-stop \
        backend backend-build backend-run backend-stop

# Default target
help:
	@echo "Makefile for managing the project."
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  frontend          Build and run the frontend service"
	@echo "  frontend-build    Build the frontend Docker image"
	@echo "  frontend-run      Run the frontend service"
	@echo "  frontend-stop     Stop the frontend service"
	@echo ""
	@echo "  backend           Build and run the backend service"
	@echo "  backend-build     Build the backend Docker image"
	@echo "  backend-run       Run the backend service"
	@echo "  backend-stop      Stop the backend service"
	@echo ""
	@echo "  help              Show this help message"

# Frontend targets
frontend: frontend-build frontend-run

frontend-build:
	cd frontend && docker-compose build

frontend-run:
	cd frontend && docker-compose up -d

frontend-stop:
	cd frontend && docker-compose down

# Backend targets
backend: backend-build backend-run

backend-build:
	cd api && docker-compose build

backend-run:
	cd api && docker-compose up -d

backend-stop:
	cd api && docker-compose down