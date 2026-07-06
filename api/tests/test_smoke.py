"""
Smoke tests for the inference API.

Scope: catch import/wiring breakage (the class of bug that prompted this
suite — a local module shadowing Python's stdlib `queue`) and confirm the
FastAPI app boots and its core routes respond. This is NOT a model-quality
or infra-integration suite (torch/transformers/redis are stubbed — see
conftest.py).
"""
import queue as stdlib_queue
import sys

import pytest
from fastapi.testclient import TestClient


def test_stdlib_queue_module_is_not_shadowed():
    """
    Regression test for the bug that caused this rename: api/queue.py used
    to sit next to app.py in /app (the container WORKDIR), which is on
    sys.path when uvicorn runs `app:app` from there. That shadowed the
    stdlib `queue` module for anything importing it by name.
    """
    assert hasattr(stdlib_queue, "Queue"), "stdlib queue module has been shadowed"
    assert "job_queue" not in stdlib_queue.__name__


def test_app_and_worker_modules_import_cleanly():
    # Import errors (e.g. `from queue import get_queue` after the rename)
    # would raise here.
    import app  # noqa: F401
    import worker  # noqa: F401
    import job_queue  # noqa: F401
    import model  # noqa: F401

    assert "app" in sys.modules
    assert "worker" in sys.modules


@pytest.fixture()
def client():
    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_after_warmup(client):
    # lifespan calls registry.warm_up_all(); with the stubbed transformers
    # pipeline this is instant, so both engines should report ready.
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_metadata_lists_both_versions(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["versions"].keys()) == {"v1", "v2"}
    assert body["versions"]["v1"]["ready"] is True


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_infer_endpoints_return_sentiment(client, version):
    resp = client.post(f"/{version}/infer", json={"text": "great product"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["version"] == version


def test_infer_rejects_empty_text(client):
    resp = client.post("/v1/infer", json={"text": ""})
    assert resp.status_code == 422


def test_job_lifecycle_create_and_fetch(client):
    created = client.post("/jobs", json={"text": "not bad at all", "version": "v1"})
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    fetched = client.get(f"/jobs/{job_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "queued"


def test_job_rejects_unknown_version(client):
    resp = client.post("/jobs", json={"text": "hi", "version": "v9"})
    assert resp.status_code in (404, 422)


def test_job_not_found(client):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404
