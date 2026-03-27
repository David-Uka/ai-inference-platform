# AI Inference Platform

A containerized inference service with versioned sentiment APIs, async job processing, Kubernetes manifests, GPU-aware deployments, and Prometheus/Grafana monitoring.

## What It Does

- Serves synchronous inference on `/v1/infer` and `/v2/infer`
- Accepts asynchronous jobs on `/jobs` and processes them with a Redis-backed worker
- Exposes metrics on `/metrics`
- Runs locally, in Docker, or on Kubernetes

## Requirements

- Python 3.11+
- Docker
- Redis
- Kubernetes and `kubectl` if you want to deploy the cluster manifests

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r api/requirements.txt
```

### 2. Start Redis

```bash
docker run --name redis -p 6379:6379 redis:7-alpine
```

### 3. Start the API

```bash
cd api
export REDIS_URL=redis://localhost:6379/0
export ENABLE_ASYNC_QUEUE=true
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Start the worker

Open another shell:

```bash
cd api
export REDIS_URL=redis://localhost:6379/0
export ENABLE_ASYNC_QUEUE=true
python worker.py
```

## API Usage

### Health and metadata

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
curl http://localhost:8000/metadata
curl http://localhost:8000/versions
```

### Synchronous inference

```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"text":"The deployment completed successfully."}'
```

```bash
curl -X POST http://localhost:8000/v2/infer \
  -H "Content-Type: application/json" \
  -d '{"text":"The deployment completed successfully."}'
```

### Asynchronous inference

Create a job:

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"version":"v2","text":"The GPU worker picked up the request."}'
```

Poll the job:

```bash
curl http://localhost:8000/jobs/<job-id>
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

## Configuration

Common environment variables:

- `REDIS_URL`: Redis connection string. Default is `redis://redis:6379/0`
- `ENABLE_ASYNC_QUEUE`: Enables `/jobs` and the worker loop. Default is `true`
- `ENABLE_REDIS_CACHE`: Enables cached inference responses. Default is `false`
- `INFERENCE_DEVICE`: `auto`, `cpu`, or `cuda`
- `MODEL_V1_NAME`: Model for `/v1/infer`
- `MODEL_V2_NAME`: Model for `/v2/infer`
- `CACHE_TTL_SECONDS`: Cache TTL for synchronous inference results
- `JOB_RESULT_TTL_SECONDS`: How long async job results stay in Redis

## Docker

Build the image:

```bash
docker build -t ai-inference-platform:latest ./api
```

Run the API:

```bash
docker run --rm -p 8000:8000 \
  --cpus="1" \
  --memory="512m" \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -e ENABLE_ASYNC_QUEUE=true \
  -v "$(pwd)/models:/models" \
  ai-inference-platform:latest
```

Run the worker:

```bash
docker run --rm \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -e ENABLE_ASYNC_QUEUE=true \
  -v "$(pwd)/models:/models" \
  ai-inference-platform:latest python worker.py
```

Run with GPU:

```bash
docker run --rm -p 8000:8000 \
  --gpus all \
  -e INFERENCE_DEVICE=cuda \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -v "$(pwd)/models:/models" \
  ai-inference-platform:latest
```

## Kubernetes

Apply the base services:

```bash
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

Enable Redis response caching in the API:

```bash
kubectl set env deployment/ai-api ENABLE_REDIS_CACHE=true REDIS_URL=redis://redis:6379/0
```

Deploy monitoring:

```bash
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml
```

Forward the UIs:

```bash
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/grafana 3000:3000
```

Deploy the GPU workload:

```bash
kubectl apply -f k8s/gpu-deployment.yaml
```

## Files

- `api/app.py`: API server
- `api/worker.py`: async worker
- `api/model.py`: model registry and inference engines
- `api/queue.py`: Redis job queue
- `k8s/`: Kubernetes manifests
- `scripts/load_test.sh`: simple request generator
