# vLLM Offline API with Structured Output

This project provides an offline API for extracting structured information using the vLLM library. It leverages FastAPI for serving the API and Docker for containerization.


### Key Files

- **`docker-compose.yml`**: Defines the Docker Compose configuration for the project.
- **`vllm/dockerfile`**: Dockerfile for building the container with the required dependencies.
- **`vllm/run.txt`**: Contains useful commands for managing the Docker container.
- **`vllm/Code/main.py`**: FastAPI application for serving the vLLM-based API.

## Features

- Offline model serving using Hugging Face models.
- Guided decoding with JSON schema validation.
- GPU support via NVIDIA Docker integration.

## Prerequisites

- Docker and Docker Compose installed.
- NVIDIA GPU with drivers and CUDA support.
- Hugging Face CLI installed for downloading models.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vLLM_Offline
   ```
2.Download the model using Hugging Face CLI:
  ```bash
  huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./vllm/Model/models--Qwen--Qwen2.5-3B-Instruct/
  ```
3.Build the Docker container:
  ```bash
  sudo docker compose build
  ```
4.Start the API:
  ```bash
  sudo docker compose up -d
  ```
5.Access the API at http://localhost:8000.

**API Endpoints**
/extract (POST)
Extracts structured information from the provided prompts.

Request Body
  ```json
  {
    "prompts": ["Your input prompt here"]
  }
  ```
Response Body
  ```json
    {
      "results": [
        {
          "name": "John Doe",
          "age": 30
        }
      ]
    }
  ```

**Environment Variables**
- MODEL_PATH: Path to the Hugging Face model directory.
- NVIDIA_VISIBLE_DEVICES: Specifies which GPUs are visible to the container.
- NVIDIA_DRIVER_CAPABILITIES: Specifies GPU capabilities required by the container.



