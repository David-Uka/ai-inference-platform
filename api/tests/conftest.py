"""
Test scaffolding for the inference API.

torch / transformers / redis are heavy or require infra we don't have in CI
sandboxing (GPU-capable wheels, network access, a running Redis). We stub
them at the sys.modules level so app.py / model.py / job_queue.py import
and run their real code paths, just against fakes instead of real backends.
This is a smoke suite, not a model-accuracy or infra-integration suite.
"""
import pathlib
import sys
import types

API_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    stub.cuda = _Cuda()
    sys.modules["torch"] = stub


def _install_transformers_stub() -> None:
    if "transformers" in sys.modules:
        return
    stub = types.ModuleType("transformers")

    def _fake_pipeline(task_name, model=None, device=None):
        def _run(text: str):
            label = "POSITIVE" if len(text) % 2 == 0 else "NEGATIVE"
            return [{"label": label, "score": 0.987654}]

        return _run

    stub.pipeline = _fake_pipeline
    sys.modules["transformers"] = stub


def _install_redis_stub() -> None:
    if "redis" in sys.modules:
        return
    stub = types.ModuleType("redis")

    class FakeRedis:
        """Tiny in-memory stand-in covering the subset job_queue.py/model.py use."""

        def __init__(self) -> None:
            self._kv: dict[str, str] = {}
            self._lists: dict[str, list[str]] = {}

        def get(self, key):
            return self._kv.get(key)

        def set(self, key, value, ex=None):
            self._kv[key] = value
            return True

        def lpush(self, key, value):
            self._lists.setdefault(key, []).insert(0, value)

        def brpoplpush(self, src, dst, timeout=0):
            lst = self._lists.get(src, [])
            if not lst:
                return None
            value = lst.pop()
            self._lists.setdefault(dst, []).insert(0, value)
            return value

        def lrem(self, key, count, value):
            lst = self._lists.get(key, [])
            if value in lst:
                lst.remove(value)

        def llen(self, key):
            return len(self._lists.get(key, []))

        def close(self):
            pass

    class Redis:  # noqa: N801 - matches real redis.Redis symbol used for typing
        ...

    def _from_url(url, decode_responses=True):
        return FakeRedis()

    stub.Redis = Redis
    stub.from_url = _from_url
    sys.modules["redis"] = stub


_install_torch_stub()
_install_transformers_stub()
_install_redis_stub()
