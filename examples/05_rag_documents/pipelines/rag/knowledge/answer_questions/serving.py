"""Call a Databricks serving endpoint over plain HTTPS.

Why not the SDK, or mlflow.deployments? Both were tried on serverless and both
hung — the SDK's client retried inside `_BaseClient._perform` until the task
timed out, five minutes for a request the endpoint answers in three seconds. The
endpoint was never the problem: querying it from a laptop with the same service
principal works instantly.

What is known to work from serverless is an ordinary HTTPS request to the
workspace host with a bearer token — that is what `scripts/verify_egress.py`
proved, and it is all this module does. Fewer layers, and each one visible.

The notebook supplies DATABRICKS_HOST and DATABRICKS_TOKEN, which it reads from
its own runtime context.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List

import requests

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 120

# A small gap between calls, and a real backoff when the endpoint pushes back.
#
# Note that REQUEST_LIMIT_EXCEEDED from these endpoints is usually NOT about rate:
# it is the batch being too large (see BATCH_SIZE in transformations.py). Backoff
# will not save you from a batch of 64 — it will just retry it six times and fail.
# Keep both: the throttle for genuine rate limits, the batch size for the real cap.
MIN_SECONDS_BETWEEN_CALLS = 0.2
MAX_RETRIES = 6
RETRY_ON = {429, 500, 502, 503, 504}

_last_call_at = 0.0


def _endpoint_url(endpoint: str) -> str:
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if not host:
        raise RuntimeError(
            "DATABRICKS_HOST is not set. The notebook must export it — "
            "spark.conf.get('spark.databricks.workspaceUrl')."
        )
    if not host.startswith("https://"):
        host = "https://" + host
    return f"{host}/serving-endpoints/{endpoint}/invocations"


def _headers() -> Dict[str, str]:
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError(
            "DATABRICKS_TOKEN is not set. The notebook must export it from its "
            "own context (dbutils ... apiToken)."
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def invoke(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to a serving endpoint, throttled, with backoff on rate limits."""
    global _last_call_at

    url = _endpoint_url(endpoint)
    headers = _headers()

    for attempt in range(MAX_RETRIES + 1):
        # Throttle *before* the call, not after a failure. Waiting only once you
        # have been refused means every batch pays a rejection first.
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)

        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        _last_call_at = time.monotonic()

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRY_ON and attempt < MAX_RETRIES:
            # Exponential backoff with jitter. Without the jitter, every task that
            # got rate-limited at the same moment retries at the same moment, and
            # they rate-limit each other all over again.
            delay = (2**attempt) + random.uniform(0, 1)
            log.warning(
                "%s returned HTTP %s — backing off %.1fs (attempt %d/%d)",
                endpoint,
                response.status_code,
                delay,
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(delay)
            continue

        raise RuntimeError(
            f"{endpoint} returned HTTP {response.status_code}: {response.text[:200]}"
        )

    raise RuntimeError(f"{endpoint}: still rate-limited after {MAX_RETRIES} retries")


# --- provider selection ---------------------------------------------------------
#
# MODEL_BACKEND=databricks (default) -> the serving endpoints above.
# MODEL_BACKEND=local                -> open-source models on the CPU, no endpoint,
#                                       no API key, no account. See
#                                       examples/_shared/local_models.py.
#
# This is the ONLY thing that changes between a Databricks run and a run on a laptop,
# in a container, or on a Kubernetes pod. The config.yaml does not move; the pipeline
# does not move. Which model answers is business logic, and business logic lives here.
#
# The endpoint argument is kept in the signature and ignored by the local backend, so
# nothing above this file has to know which provider it got.


def _local():
    import sys
    from pathlib import Path

    shared = Path(__file__).resolve().parents[5] / "_shared"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    import local_models

    return local_models


def _use_local() -> bool:
    return os.environ.get("MODEL_BACKEND", "databricks").lower() == "local"


def embed(endpoint: str, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, preserving order.

    Order is load-bearing: these vectors get zipped back onto their chunks by
    position. A response that came back out of order would attach every chunk's
    meaning to a different chunk, and nothing would look wrong until an answer
    cited a source with nothing to do with the question.
    """
    if _use_local():
        return _local().embed(texts)

    body = invoke(endpoint, {"input": texts})
    data = body["data"]
    ordered = sorted(data, key=lambda item: item.get("index", 0))

    if len(ordered) != len(texts):
        raise RuntimeError(
            f"{endpoint} returned {len(ordered)} embeddings for {len(texts)} inputs"
        )
    return [[float(x) for x in item["embedding"]] for item in ordered]


def chat(endpoint: str, system: str, user: str, max_tokens: int = 250) -> str:
    """Ask a chat model and insist on getting text back."""
    if _use_local():
        return _local().chat(system, user, max_tokens=max_tokens)

    body = invoke(
        endpoint,
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,  # a RAG answer should not be creative
            "max_tokens": max_tokens,
        },
    )
    content = body["choices"][0]["message"]["content"]

    if not isinstance(content, str):
        # gpt-oss returns structured *reasoning* objects rather than a string. A
        # naive choices[0].message.content would write a mangled dict into the
        # answer column and nothing would notice.
        raise RuntimeError(
            f"{endpoint} returned {type(content).__name__}, not text. "
            "Use an endpoint that returns a string."
        )
    return content.strip()
