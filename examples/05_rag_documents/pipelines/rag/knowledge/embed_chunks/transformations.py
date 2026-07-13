"""Embed text chunks with an open-source model served in the workspace.

BGE-large is open source (BAAI) and already deployed as a Databricks serving
endpoint, so this needs no weights downloaded, no GPU, and no outbound internet —
all three of which serverless makes difficult.

The batching is not an optimisation, it is the difference between working and not.
Embedding 262 chunks one HTTP request at a time is 262 round trips; the endpoint
takes a list, so send it a list.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from ubunye.core.interfaces import Task

log = logging.getLogger(__name__)

# An open-source embedding model, served in-workspace. Nothing here is specific to
# it — point EMBEDDING_ENDPOINT at another and the pipeline does not change.
DEFAULT_ENDPOINT = "databricks-bge-large-en"
# Sixteen, and the number is not arbitrary. The endpoint rejects a batch of 32 with
# REQUEST_LIMIT_EXCEEDED and the message "Exceeded workspace QPS rate limit" — which
# is misleading, because it is not about rate at all: six calls of 16 fired
# back-to-back with no delay all succeed, while a single call of 32 always fails.
# The cap is on INPUTS PER REQUEST, and the error names the wrong thing.
#
# Measured against the live endpoint, not read in a document:
#     batch 8  -> OK      batch 32 -> 429
#     batch 16 -> OK      batch 64 -> 429
BATCH_SIZE = 16


class EmbedChunks(Task):
    """document_chunks -> chunk_embeddings."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        chunks: DataFrame = sources["chunks"]
        endpoint = os.environ.get("EMBEDDING_ENDPOINT", DEFAULT_ENDPOINT)

        # Collect to the driver: this is a few hundred chunks and the work is an
        # HTTP call, not computation. A Spark UDF would open a connection pool per
        # executor to make the same number of requests, slower and harder to debug.
        # At a million chunks this would become a pandas_udf; at 262 it would be
        # cargo cult.
        rows = chunks.select("chunk_id", "doc_id", "source", "title", "chunk_text").collect()
        log.info("embedding %d chunks via %s", len(rows), endpoint)

        texts = [r["chunk_text"] for r in rows]
        vectors = _embed_all(texts, endpoint)

        embedded = [
            (
                r["chunk_id"],
                r["doc_id"],
                r["source"],
                r["title"],
                r["chunk_text"],
                [float(x) for x in vec],
                endpoint,
                len(vec),
            )
            for r, vec in zip(rows, vectors)
        ]

        schema = T.StructType(
            [
                T.StructField("chunk_id", T.StringType()),
                T.StructField("doc_id", T.StringType()),
                T.StructField("source", T.StringType()),
                T.StructField("title", T.StringType()),
                T.StructField("chunk_text", T.StringType()),
                T.StructField("embedding", T.ArrayType(T.FloatType())),
                T.StructField("embedding_model", T.StringType()),
                T.StructField("embedding_dim", T.IntegerType()),
            ]
        )

        spark = chunks.sparkSession
        return {
            "chunk_embeddings": spark.createDataFrame(embedded, schema).withColumn(
                "embedded_at", F.current_timestamp()
            )
        }


def _batches(items: List[str], size: int) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _embed_all(texts: List[str], endpoint: str) -> List[List[float]]:
    """Embed in batches, keeping order."""
    from serving import embed  # sits next to this file; the task dir is on sys.path

    vectors: List[List[float]] = []
    for batch in _batches(texts, BATCH_SIZE):
        vectors.extend(embed(endpoint, batch))
    return vectors
