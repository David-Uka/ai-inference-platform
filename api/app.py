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
from model import InferenceEngine, get_engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine = get_engine()
    engine.warm_up()
    MODEL_READY.set(1 if engine.is_ready else 0)
    CACHE_ENABLED.set(1 if engine.cache_enabled else 0)
    yield
    await engine.close()
    MODEL_READY.set(0)


app = FastAPI(
    title="AI Inference Platform",
    version="1.0.0",
    description="Sentiment inference API designed for Docker, Kubernetes, and GPU-aware deployments.",
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


class InferenceResponse(BaseModel):
    label: str
    score: float
    cached: bool
    model_name: str
    device: str
    latency_ms: float


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    engine = get_engine()
    if not engine.is_ready:
        MODEL_READY.set(0)
        raise HTTPException(status_code=503, detail="model not loaded")
    MODEL_READY.set(1)
    return {"status": "ready"}


@app.get("/metadata")
async def metadata() -> dict[str, str | bool]:
    engine = get_engine()
    CACHE_ENABLED.set(1 if engine.cache_enabled else 0)
    MODEL_READY.set(1 if engine.is_ready else 0)
    return {
        "model_name": engine.model_name,
        "device": engine.device,
        "redis_cache_enabled": engine.cache_enabled,
        "ready": engine.is_ready,
    }


@app.post("/v1/infer", response_model=InferenceResponse)
async def infer(payload: InferenceRequest) -> InferenceResponse:
    engine = get_engine()
    started = perf_counter()
    result, cached = await run_in_threadpool(engine.predict, payload.text)
    latency_seconds = perf_counter() - started
    cached_label = "true" if cached else "false"
    INFERENCE_LATENCY_SECONDS.labels(
        device=engine.device,
        cached=cached_label,
    ).observe(latency_seconds)
    INFERENCE_REQUESTS_TOTAL.labels(
        cached=cached_label,
        device=engine.device,
        label=result["label"],
    ).inc()
    return InferenceResponse(
        label=result["label"],
        score=float(result["score"]),
        cached=cached,
        model_name=engine.model_name,
        device=engine.device,
        latency_ms=round(latency_seconds * 1000, 2),
    )
