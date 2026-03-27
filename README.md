# AI Inference Platform

A portfolio-ready project that ties together container internals, Kubernetes scheduling, and GPU-aware AI deployment.

## What This Project Demonstrates

- A FastAPI inference API backed by Hugging Face Transformers
- Docker image optimization with multi-stage builds and runtime model caching
- Kubernetes deployments with resource requests, limits, node selectors, and tolerations
- GPU-aware scheduling with a dedicated deployment that requests `nvidia.com/gpu`
- Versioned model serving with `/v1` and `/v2` routes
- Optional Redis response caching and a Redis-backed async worker queue
- Prometheus and Grafana observability for request rate, latency, cache behavior, and process resource usage

## Architecture

```text
Client -> FastAPI API -> Hugging Face model pipeline
                    -> Redis cache (optional)
                    -> Redis job queue -> Worker
                    -> /metrics -> Prometheus -> Grafana
                    -> Docker container
                    -> Kubernetes Deployment / Service / HPA
```

## Repository Layout

```text
ai-inference-platform/
├── api/
│   ├── app.py
│   ├── metrics.py
│   ├── model.py
│   ├── queue.py
│   ├── worker.py
│   ├── Dockerfile
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   ├── gpu-deployment.yaml
│   ├── hpa.yaml
│   ├── monitoring/
│   │   ├── grafana.yaml
│   │   └── prometheus.yaml
│   ├── redis.yaml
│   ├── service.yaml
│   └── worker-deployment.yaml
├── scripts/
│   └── load_test.sh
└── README.md
```

## API Endpoints

- `GET /healthz`: liveness check
- `GET /readyz`: readiness check after model warm-up
- `GET /metadata`: device, model, and cache metadata
- `GET /versions`: configured model versions and their backing models
- `GET /metrics`: Prometheus-compatible metrics endpoint
- `POST /v1/infer`: baseline sentiment inference
- `POST /v2/infer`: alternate sentiment inference
- `POST /jobs`: enqueue an async inference job
- `GET /jobs/{job_id}`: poll an async job result

Example request:

```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"text":"Kubernetes scheduling is finally starting to click for me."}'
```

```bash
curl -X POST http://localhost:8000/v2/infer \
  -H "Content-Type: application/json" \
  -d '{"text":"This cluster rollout feels stable and fast."}'
```

## Local Development

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r api/requirements.txt
```

### 2. Run the API locally

```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Enable Redis caching optionally

Run Redis:

```bash
docker run --name redis -p 6379:6379 redis:7-alpine
```

Start the API with caching enabled:

```bash
export ENABLE_REDIS_CACHE=true
export REDIS_URL=redis://localhost:6379/0
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the async worker

The queue flow uses the same Redis instance:

```bash
cd api
export ENABLE_ASYNC_QUEUE=true
export REDIS_URL=redis://localhost:6379/0
python worker.py
```

### 5. Submit async jobs

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"version":"v2","text":"The GPU-aware deployment finally worked."}'
```

Then poll the returned `job_id`:

```bash
curl http://localhost:8000/jobs/<job-id>
```

Sample metric queries after traffic:

```bash
curl -s http://localhost:8000/metrics | grep ai_api_http_requests_total
curl -s http://localhost:8000/metrics | grep ai_api_inference_latency_seconds
```

## Docker Deep Dive

### Build the image

```bash
docker build -t ai-inference-platform:latest ./api
```

Run the worker from the same image:

```bash
docker run --rm \
  -e ENABLE_ASYNC_QUEUE=true \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -v "$(pwd)/models:/models" \
  ai-inference-platform:latest python worker.py
```

### Run with explicit resource limits

```bash
docker run --rm -p 8000:8000 \
  --cpus="1" \
  --memory="512m" \
  -v "$(pwd)/models:/models" \
  ai-inference-platform:latest
```

### If you have an NVIDIA GPU

```bash
docker run --rm -p 8000:8000 \
  --gpus all \
  -e INFERENCE_DEVICE=cuda \
  -v "$(pwd)/models:/models" \
  ai-inference-platform:latest
```

### Concepts to explain in interviews

- Image layers: every `Dockerfile` instruction creates a layer, and the multi-stage build keeps compilers out of the final image.
- Filesystem: the final container sees read-only image layers plus one writable layer mounted on top through OverlayFS.
- Namespaces: the container has isolated process, network, and mount views even though it shares the host kernel.
- Cgroups: flags like `--cpus` and `--memory` become kernel-enforced resource controls.
- Networking: Docker bridge mode gives the container a private IP and port mapping publishes it on the host.

Useful inspection commands:

```bash
docker history ai-inference-platform:latest
docker inspect ai-inference-platform:latest
docker network ls
docker stats
```

## Kubernetes Deep Dive

### Apply the baseline manifests

```bash
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Redis-backed cache and queue

```bash
kubectl apply -f k8s/redis.yaml
kubectl set env deployment/ai-api ENABLE_REDIS_CACHE=true REDIS_URL=redis://redis:6379/0
```

The worker consumes jobs from Redis and writes results back so the API can return `202 Accepted` immediately for long-running requests.

### Why the scheduler picks a node

The baseline deployment includes:

- `nodeSelector.workload=general`, so only nodes with that label are eligible.
- CPU and memory requests, which the scheduler uses to determine whether a node has enough allocatable capacity.
- No GPU request, so regular worker nodes remain valid targets.

To see the actual scheduling decision:

```bash
kubectl get pods -o wide
kubectl describe pod <pod-name>
```

When explaining a placement decision, mention all of these:

1. Which nodes matched the pod's label constraints.
2. Whether each matching node had enough requested CPU and memory available.
3. Whether taints blocked the pod.
4. Which final node the scheduler bound the pod to.

### Debugging pod lifecycle

```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

Common states to be able to explain:

- `Pending`: not enough resources, unsatisfied selectors, or missing GPU capacity
- `Running`: probes passed and the app is serving traffic
- `CrashLoopBackOff`: process exits repeatedly or readiness never stabilizes

### Horizontal autoscaling

You can use the included HPA or create one imperatively:

```bash
kubectl autoscale deployment ai-api --cpu-percent=50 --min=1 --max=5
```

This relies on CPU utilization relative to the pod's requested CPU.

## Observability

### What the API exports

The service now exposes application metrics directly from `/metrics`, including:

- `ai_api_http_requests_total`
- `ai_api_http_request_duration_seconds`
- `ai_api_inference_requests_total`
- `ai_api_inference_latency_seconds`
- `ai_api_model_ready`
- `ai_api_cache_enabled`

Because the Python Prometheus client also exports process metrics, Grafana can show CPU and memory without adding a second exporter:

- `process_cpu_seconds_total`
- `process_resident_memory_bytes`

### Deploy Prometheus and Grafana

```bash
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml
```

Forward the UIs locally:

```bash
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/grafana 3000:3000
```

Open:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Grafana credentials for this demo setup:

- username: `admin`
- password: `admin`

### Dashboard signals you can talk through

- Request rate: `sum(rate(ai_api_http_requests_total{path="/v1/infer"}[5m]))`
- P95 latency by version: `histogram_quantile(0.95, sum(rate(ai_api_inference_latency_seconds_bucket[5m])) by (le, version))`
- Cache hit ratio: `sum(rate(ai_api_inference_requests_total{cached="true"}[5m])) / sum(rate(ai_api_inference_requests_total[5m]))`
- Memory usage: `process_resident_memory_bytes`
- CPU usage: `rate(process_cpu_seconds_total[5m])`

### Why this is strong portfolio signal

- You are not treating the model service as a black box; you are exposing latency and serving behavior from the application layer.
- You can correlate infrastructure limits with real application metrics during load tests.
- You can explain the difference between request throughput, model latency, and host resource pressure.

## GPU Workloads

### Kubernetes GPU deployment

The manifest in `k8s/gpu-deployment.yaml` adds:

- `nodeSelector.accelerator=nvidia`
- a `gpu` toleration
- `limits.nvidia.com/gpu: 1`
- `INFERENCE_DEVICE=cuda`

Apply it with:

```bash
kubectl apply -f k8s/gpu-deployment.yaml
```

That file also includes `ai-worker-gpu`, which requests a GPU for background jobs the same way the API deployment does.

### How Kubernetes knows which node has a GPU

1. The NVIDIA device plugin runs on GPU-capable nodes.
2. It advertises GPU capacity to the kubelet.
3. Kubernetes exposes that capacity as the schedulable resource `nvidia.com/gpu`.
4. A pod that requests `nvidia.com/gpu: 1` can only bind to nodes reporting at least one free GPU.

If the pod stays `Pending`, inspect:

```bash
kubectl describe pod <gpu-pod-name>
kubectl get nodes -o json | jq '.items[].status.allocatable'
```

## Suggested Portfolio Talking Points

- "I used multi-stage Docker builds so the runtime image stays slimmer and avoids shipping toolchains."
- "I can explain the difference between Kubernetes resource requests and limits, and why scheduling decisions are based on requests."
- "I understand why GPU pods remain pending when the device plugin is missing or nodes do not advertise `nvidia.com/gpu`."
- "I designed the image so model weights are pulled into a mounted cache at runtime rather than inflating the image layer set."
- "I exposed two model versions behind separate API routes so I can compare behavior and evolve the service without breaking clients."
- "I added a Redis-backed worker queue so the API can accept asynchronous inference jobs and hand them off to background workers."
- "I instrumented the API with Prometheus metrics and built dashboards for request rate, p95 latency, cache hit ratio, CPU, and memory."
