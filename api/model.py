import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

import redis
import torch
from transformers import pipeline


def _truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool
    url: str
    ttl_seconds: int


@dataclass(frozen=True)
class ModelSpec:
    version: str
    model_name: str
    task_name: str
    description: str


class InferenceEngine:
    def __init__(self, spec: ModelSpec, cache: CacheConfig, device: str) -> None:
        self.spec = spec
        self.model_name = spec.model_name
        self.task_name = spec.task_name
        self.device = device
        self.cache = cache
        self._pipe = None
        self._redis_client: redis.Redis | None = None
        self.is_ready = False

    @property
    def cache_enabled(self) -> bool:
        return self.cache.enabled

    def warm_up(self) -> None:
        if self._pipe is not None:
            return
        device_index = 0 if self.device == "cuda" else -1
        self._pipe = pipeline(
            self.task_name,
            model=self.model_name,
            device=device_index,
        )
        self.is_ready = True

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"inference:{self.spec.version}:{self.model_name}:{digest}"

    def _get_cached(self, text: str) -> dict[str, Any] | None:
        if not self.cache.enabled:
            return None
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.cache.url, decode_responses=True)
        payload = self._redis_client.get(self._cache_key(text))
        if not payload:
            return None
        return json.loads(payload)

    def _set_cached(self, text: str, value: dict[str, Any]) -> None:
        if not self.cache.enabled:
            return
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.cache.url, decode_responses=True)
        self._redis_client.set(
            self._cache_key(text),
            json.dumps(value),
            ex=self.cache.ttl_seconds,
        )

    def predict(self, text: str) -> tuple[dict[str, Any], bool]:
        self.warm_up()
        cached = self._get_cached(text)
        if cached:
            return cached, True
        result = self._pipe(text)[0]
        self._set_cached(text, result)
        return result, False

    def metadata(self) -> dict[str, str | bool]:
        return {
            "version": self.spec.version,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "description": self.spec.description,
            "device": self.device,
            "redis_cache_enabled": self.cache_enabled,
            "ready": self.is_ready,
        }

    async def close(self) -> None:
        if self._redis_client is not None:
            self._redis_client.close()


class ModelRegistry:
    def __init__(self) -> None:
        self.device = self._resolve_device()
        self.cache = CacheConfig(
            enabled=_truthy(os.getenv("ENABLE_REDIS_CACHE")),
            url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
        )
        specs = [
            ModelSpec(
                version="v1",
                model_name=os.getenv(
                    "MODEL_V1_NAME",
                    "distilbert-base-uncased-finetuned-sst-2-english",
                ),
                task_name=os.getenv("MODEL_V1_TASK", "sentiment-analysis"),
                description=os.getenv(
                    "MODEL_V1_DESCRIPTION",
                    "Baseline sentiment model optimized for low-latency inference.",
                ),
            ),
            ModelSpec(
                version="v2",
                model_name=os.getenv(
                    "MODEL_V2_NAME",
                    "cardiffnlp/twitter-roberta-base-sentiment-latest",
                ),
                task_name=os.getenv("MODEL_V2_TASK", "sentiment-analysis"),
                description=os.getenv(
                    "MODEL_V2_DESCRIPTION",
                    "Alternate sentiment model tuned for social and conversational text.",
                ),
            ),
        ]
        self.engines = {
            spec.version: InferenceEngine(spec=spec, cache=self.cache, device=self.device)
            for spec in specs
        }

    def _resolve_device(self) -> str:
        requested = os.getenv("INFERENCE_DEVICE", "auto").lower()
        cuda_available = torch.cuda.is_available()
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" and not cuda_available:
            return "cpu"
        if requested in {"auto", "cuda"} and cuda_available:
            return "cuda"
        return "cpu"

    @property
    def cache_enabled(self) -> bool:
        return self.cache.enabled

    def warm_up_all(self) -> None:
        for engine in self.engines.values():
            engine.warm_up()

    def get(self, version: str) -> InferenceEngine:
        try:
            return self.engines[version]
        except KeyError as exc:
            raise KeyError(f"unknown model version '{version}'") from exc

    def predict(self, version: str, text: str) -> tuple[dict[str, Any], bool, InferenceEngine]:
        engine = self.get(version)
        result, cached = engine.predict(text)
        return result, cached, engine

    def metadata(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "redis_cache_enabled": self.cache_enabled,
            "versions": {
                version: engine.metadata() for version, engine in self.engines.items()
            },
        }

    def readiness(self) -> bool:
        return all(engine.is_ready for engine in self.engines.values())

    async def close(self) -> None:
        for engine in self.engines.values():
            await engine.close()


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
