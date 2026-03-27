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


@dataclass
class CacheConfig:
    enabled: bool
    url: str
    ttl_seconds: int


class InferenceEngine:
    def __init__(self) -> None:
        self.model_name = os.getenv(
            "MODEL_NAME",
            "distilbert-base-uncased-finetuned-sst-2-english",
        )
        self.task_name = os.getenv("TASK_NAME", "sentiment-analysis")
        self.device = self._resolve_device()
        self.cache = CacheConfig(
            enabled=_truthy(os.getenv("ENABLE_REDIS_CACHE")),
            url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
        )
        self._pipe = None
        self._redis_client: redis.Redis | None = None
        self.is_ready = False

    @property
    def cache_enabled(self) -> bool:
        return self.cache.enabled

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
        return f"inference:{self.model_name}:{digest}"

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

    async def close(self) -> None:
        if self._redis_client is not None:
            self._redis_client.close()


_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
