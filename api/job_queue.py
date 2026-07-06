import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import redis


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class QueueConfig:
    enabled: bool
    url: str
    pending_key: str
    processing_key: str
    result_ttl_seconds: int
    blocking_timeout_seconds: int


class InferenceJobQueue:
    def __init__(self) -> None:
        self.config = QueueConfig(
            enabled=str(os.getenv("ENABLE_ASYNC_QUEUE", "true")).lower()
            in {"1", "true", "yes", "on"},
            url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            pending_key=os.getenv("JOB_QUEUE_PENDING_KEY", "inference_jobs:pending"),
            processing_key=os.getenv("JOB_QUEUE_PROCESSING_KEY", "inference_jobs:processing"),
            result_ttl_seconds=int(os.getenv("JOB_RESULT_TTL_SECONDS", "3600")),
            blocking_timeout_seconds=int(os.getenv("JOB_QUEUE_BLOCKING_TIMEOUT", "5")),
        )
        self._redis_client: redis.Redis | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _client(self) -> redis.Redis:
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.config.url, decode_responses=True)
        return self._redis_client

    def _job_key(self, job_id: str) -> str:
        return f"inference_jobs:result:{job_id}"

    def enqueue(self, text: str, version: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("async queue is disabled")
        job_id = str(uuid.uuid4())
        record = {
            "job_id": job_id,
            "version": version,
            "text": text,
            "status": "queued",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        payload = json.dumps(record)
        client = self._client()
        client.set(self._job_key(job_id), payload, ex=self.config.result_ttl_seconds)
        client.lpush(self.config.pending_key, payload)
        return record

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        payload = self._client().get(self._job_key(job_id))
        if not payload:
            return None
        return json.loads(payload)

    def reserve(self) -> dict[str, Any] | None:
        payload = self._client().brpoplpush(
            self.config.pending_key,
            self.config.processing_key,
            timeout=self.config.blocking_timeout_seconds,
        )
        if not payload:
            return None
        return json.loads(payload)

    def mark_processing(self, job: dict[str, Any]) -> None:
        updated = {
            **job,
            "status": "processing",
            "updated_at": _utc_now(),
        }
        self._write_job(updated)

    def mark_completed(
        self,
        job: dict[str, Any],
        result: dict[str, Any],
        cached: bool,
        model_name: str,
        device: str,
    ) -> None:
        updated = {
            **job,
            "status": "completed",
            "cached": cached,
            "model_name": model_name,
            "device": device,
            "result": {
                "label": result["label"],
                "score": float(result["score"]),
            },
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._write_job(updated)
        self._ack(job)

    def mark_failed(self, job: dict[str, Any], error: str) -> None:
        updated = {
            **job,
            "status": "failed",
            "error": error,
            "updated_at": _utc_now(),
        }
        self._write_job(updated)
        self._ack(job)

    def queue_depth(self) -> int:
        return int(self._client().llen(self.config.pending_key))

    def _write_job(self, job: dict[str, Any]) -> None:
        self._client().set(
            self._job_key(job["job_id"]),
            json.dumps(job),
            ex=self.config.result_ttl_seconds,
        )

    def _ack(self, job: dict[str, Any]) -> None:
        self._client().lrem(
            self.config.processing_key,
            1,
            json.dumps(job),
        )

    async def close(self) -> None:
        if self._redis_client is not None:
            self._redis_client.close()


_queue: InferenceJobQueue | None = None


def get_queue() -> InferenceJobQueue:
    global _queue
    if _queue is None:
        _queue = InferenceJobQueue()
    return _queue
