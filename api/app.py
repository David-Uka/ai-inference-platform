from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from metrics import (
    CACHE_ENABLED,
    INFERENCE_LATENCY_SECONDS,
    INFERENCE_REQUESTS_TOTAL,
    MODEL_READY,
    setup_metrics,
)
from model import ModelRegistry, get_registry
from queue import get_queue


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry = get_registry()
    registry.warm_up_all()
    MODEL_READY.set(1 if registry.readiness() else 0)
    CACHE_ENABLED.set(1 if registry.cache_enabled else 0)
    yield
    await registry.close()
    await get_queue().close()
    MODEL_READY.set(0)


app = FastAPI(
    title="Inference Runtime",
    version="2.0.0",
    description="Versioned sentiment inference API designed for Docker, Kubernetes, and GPU-aware deployments.",
    lifespan=lifespan,
)
setup_metrics(app)


class InferenceRequest(BaseModel):
    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4000,
            description="Input text for sentiment analysis.",
        ),
    ]


class InferenceJobRequest(InferenceRequest):
    version: Annotated[
        str,
        Field(
            pattern=r"^v[12]$",
            description="Model version to run asynchronously.",
        ),
    ] = "v1"


class InferenceResponse(BaseModel):
    version: str
    label: str
    score: float
    cached: bool
    model_name: str
    device: str
    latency_ms: float


class InferenceJobAccepted(BaseModel):
    job_id: str
    version: str
    status: str
    created_at: str


class InferenceJobStatus(BaseModel):
    job_id: str
    version: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    cached: bool | None = None
    model_name: str | None = None
    device: str | None = None
    result: dict[str, str | float] | None = None
    error: str | None = None


def _registry() -> ModelRegistry:
    return get_registry()


def _build_inference_response(version: str, payload: InferenceRequest) -> InferenceResponse:
    registry = _registry()
    started = perf_counter()
    result, cached, engine = registry.predict(version, payload.text)
    latency_seconds = perf_counter() - started
    cached_label = "true" if cached else "false"
    INFERENCE_LATENCY_SECONDS.labels(
        version=version,
        device=engine.device,
        cached=cached_label,
    ).observe(latency_seconds)
    INFERENCE_REQUESTS_TOTAL.labels(
        version=version,
        cached=cached_label,
        device=engine.device,
        label=result["label"],
    ).inc()
    return InferenceResponse(
        version=version,
        label=result["label"],
        score=float(result["score"]),
        cached=cached,
        model_name=engine.model_name,
        device=engine.device,
        latency_ms=round(latency_seconds * 1000, 2),
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    registry = _registry()
    if not registry.readiness():
        MODEL_READY.set(0)
        raise HTTPException(status_code=503, detail="one or more models are not loaded")
    MODEL_READY.set(1)
    return {"status": "ready"}


@app.get("/metadata")
async def metadata() -> dict[str, object]:
    registry = _registry()
    CACHE_ENABLED.set(1 if registry.cache_enabled else 0)
    MODEL_READY.set(1 if registry.readiness() else 0)
    return registry.metadata()


@app.get("/versions")
async def versions() -> dict[str, object]:
    return _registry().metadata()["versions"]


@app.post("/v1/infer", response_model=InferenceResponse)
async def infer_v1(payload: InferenceRequest) -> InferenceResponse:
    return await run_in_threadpool(_build_inference_response, "v1", payload)


@app.post("/v2/infer", response_model=InferenceResponse)
async def infer_v2(payload: InferenceRequest) -> InferenceResponse:
    return await run_in_threadpool(_build_inference_response, "v2", payload)


@app.post("/jobs", response_model=InferenceJobAccepted, status_code=202)
async def create_job(payload: InferenceJobRequest) -> InferenceJobAccepted:
    queue = get_queue()
    if not queue.enabled:
        raise HTTPException(status_code=503, detail="async queue is disabled")
    if payload.version not in _registry().engines:
        raise HTTPException(status_code=404, detail=f"unknown model version '{payload.version}'")

    try:
        job = await run_in_threadpool(queue.enqueue, payload.text, payload.version)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}") from exc

    return InferenceJobAccepted(
        job_id=job["job_id"],
        version=job["version"],
        status=job["status"],
        created_at=job["created_at"],
    )


@app.get("/jobs/{job_id}", response_model=InferenceJobStatus)
async def get_job(job_id: str) -> InferenceJobStatus:
    queue = get_queue()
    if not queue.enabled:
        raise HTTPException(status_code=503, detail="async queue is disabled")

    try:
        job = await run_in_threadpool(queue.get_job, job_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}") from exc

    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    return InferenceJobStatus(**job)
