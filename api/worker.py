import logging
import time

from model import get_registry
from queue import get_queue


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("inference-worker")


def main() -> None:
    registry = get_registry()
    queue = get_queue()

    if not queue.enabled:
        raise RuntimeError("ENABLE_ASYNC_QUEUE must be enabled for the worker")

    registry.warm_up_all()
    logger.info("worker started with versions=%s", ",".join(registry.engines.keys()))

    while True:
        job = queue.reserve()
        if job is None:
            continue

        job_id = job["job_id"]
        version = job["version"]
        logger.info("processing job_id=%s version=%s", job_id, version)
        queue.mark_processing(job)

        try:
            result, cached, engine = registry.predict(version, job["text"])
            queue.mark_completed(
                job=job,
                result=result,
                cached=cached,
                model_name=engine.model_name,
                device=engine.device,
            )
            logger.info("completed job_id=%s version=%s cached=%s", job_id, version, cached)
        except Exception as exc:
            queue.mark_failed(job, str(exc))
            logger.exception("failed job_id=%s version=%s", job_id, version)
            time.sleep(1)


if __name__ == "__main__":
    main()
