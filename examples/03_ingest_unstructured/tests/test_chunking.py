"""The chunker is pure Python, so it is tested without Spark, without a cluster,
and without a workspace — in milliseconds, on every pull request."""

from __future__ import annotations

from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "pipelines/docs/ingestion/document_index"
CORPUS = Path(__file__).resolve().parents[1] / "data/corpus"


def _load_chunker():
    """Pull chunk_text out of transformations.py without importing pyspark.

    That module imports pyspark at the top and PR runners have no JVM. The
    function under test is pure Python, so lift it out rather than dragging Spark
    into a unit test that does not need it.
    """
    src = (TASK / "transformations.py").read_text(encoding="utf-8")
    consts = [ln for ln in src.splitlines() if ln.startswith(("CHUNK_WORDS", "OVERLAP_WORDS"))]
    body = src[src.index("def chunk_text") : src.index("_chunk_udf =")]

    namespace: dict = {}
    exec("from typing import List\n" + "\n".join(consts) + "\n" + body, namespace)
    return namespace["chunk_text"], namespace["CHUNK_WORDS"], namespace["OVERLAP_WORDS"]


chunk_text, CHUNK_WORDS, OVERLAP_WORDS = _load_chunker()


def test_short_text_is_one_chunk():
    assert len(chunk_text("hello world")) == 1


def test_empty_text_yields_nothing():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_windows_overlap():
    """The overlap is the point. Without it, a sentence split across a boundary is
    a sentence that neither chunk can answer."""
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text)

    assert len(chunks) > 1
    tail = chunks[0]["chunk_text"].split()[-OVERLAP_WORDS:]
    head = chunks[1]["chunk_text"].split()[:OVERLAP_WORDS]
    assert tail == head


def test_no_words_are_lost():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text)
    assert chunks[-1]["chunk_text"].split()[-1] == "w399"


def test_the_real_corpus_actually_chunks():
    """A window bigger than every document means the chunking never runs — the
    example would pass, prove nothing, and mislead whoever copied it.

    This is not hypothetical: the window was 200 words and no document reached
    200, so every document produced exactly one chunk.
    """
    docs = sorted(CORPUS.glob("*.txt"))
    assert docs, "corpus is missing"

    for doc in docs:
        chunks = chunk_text(doc.read_text(encoding="utf-8"))
        assert len(chunks) > 1, f"{doc.name} produced one chunk — the window is too big"
