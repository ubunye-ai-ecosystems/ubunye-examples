"""Open-source models that run anywhere — the same two functions, no endpoint.

Examples 05 (RAG) and 06 (distillation) were the two that could not leave Databricks,
because they call Databricks-hosted serving endpoints: `databricks-bge-large-en` for
embeddings and `databricks-meta-llama-3-1-8b-instruct` for generation. Take the
workspace away and there is nothing to call.

That was never a limitation of the *pipeline*. It was a limitation of one dependency —
a hosted model — and a hosted model has an open-source substitute that runs on a CPU
with no API key and no account:

    embeddings   sentence-transformers/all-MiniLM-L6-v2      ~90 MB
    generation   Qwen/Qwen2.5-0.5B-Instruct                  ~1 GB

Both come from HuggingFace, which serverless Databricks *can* reach (measured:
`huggingface.co` and `pypi.org` are allowed; `raw.githubusercontent.com` is not). So
this backend works off Databricks **and on it** — it is not a downgrade for the
unlucky, it is a second provider.

This module implements exactly the two functions `serving.py` exposes, with the same
signatures, so nothing above it changes:

    embed(texts)                 -> list[list[float]]
    chat(system, user, max_tokens) -> str

The models are cached per process. Loading a transformer for every row is the single
most common way a "slow model" turns out to be a slow *loop*.

**The numbers will differ from Databricks, and they are supposed to.** A 0.5B model is
not an 8B model, and MiniLM is not BGE-large. What is identical is the *process*:
chunk, embed, retrieve by cosine, ground the answer in the retrieved text, cite the
sources. That is what the example teaches, and it is what has to survive the move.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

log = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.environ.get("LOCAL_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Loaded once, on first use. Not at import: importing this module must stay cheap, and
# a task that only embeds must not pay to load a language model it never calls.
_CACHE: Dict[str, Any] = {}


def _embedder():
    if "embedder" not in _CACHE:
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model %s (CPU)", EMBED_MODEL)
        _CACHE["embedder"] = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _CACHE["embedder"]


def _chat_model():
    if "chat" not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading chat model %s (CPU)", CHAT_MODEL)
        tokeniser = AutoTokenizer.from_pretrained(CHAT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, torch_dtype=torch.float32)
        model.eval()

        # Cap the threads. Unbounded, torch spawns a worker per core, each with its own
        # arena — no faster for a model this size, and it is how TensorFlow OOM'd the
        # driver in example 10. Same lesson, different library.
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

        _CACHE["chat"] = (tokeniser, model)
    return _CACHE["chat"]


def embed(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, preserving order.

    Order is load-bearing: these vectors get zipped back onto their chunks by position.
    Vectors that came back out of order would attach every chunk's meaning to a
    different chunk, and nothing would look wrong until an answer cited a source with
    nothing to do with the question.
    """
    vectors = _embedder().encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=False,  # the task computes cosine itself; don't do it twice
        show_progress_bar=False,
    )
    if len(vectors) != len(texts):
        raise RuntimeError(f"got {len(vectors)} embeddings for {len(texts)} inputs")
    return [[float(x) for x in v] for v in vectors]


def chat(system: str, user: str, max_tokens: int = 250) -> str:
    """Ask the local model and insist on getting text back."""
    import torch

    tokeniser, model = _chat_model()

    prompt = tokeniser.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokeniser(prompt, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # a RAG answer should not be creative
            pad_token_id=tokeniser.eos_token_id,
        )

    # Only the NEW tokens. Decoding the whole sequence returns the prompt back with the
    # answer glued on the end, and the answer column quietly fills up with the question.
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokeniser.decode(generated, skip_special_tokens=True).strip()
