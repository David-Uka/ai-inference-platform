from time import perf_counter

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


HTTP_REQUESTS_TOTAL = Counter(
    "ai_api_http_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "ai_api_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "ai_api_inference_requests_total",
    "Total inference requests served.",
    ["cached", "device", "label"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "ai_api_inference_latency_seconds",
    "Model inference latency in seconds.",
    ["device", "cached"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

MODEL_READY = Gauge(
    "ai_api_model_ready",
    "Whether the model is fully loaded and ready to serve traffic.",
)

CACHE_ENABLED = Gauge(
    "ai_api_cache_enabled",
    "Whether Redis response caching is enabled.",
)


def setup_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_http_middleware(request: Request, call_next):
        started = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - started
        path = request.url.path
        method = request.method
        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            path=path,
            status_code=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
