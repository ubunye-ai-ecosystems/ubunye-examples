"""Turn free text and raw files into a chunked document index.

The rule is the same for both sources once the bytes are decoded: normalise into
(doc_id, source, title, text), then cut the text into overlapping windows. What
comes out is what a vector index wants — the embedding step is deliberately not
here, because embedding needs a model endpoint and this example is about getting
unstructured data *into the lakehouse*, not about which embedding model you like.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from ubunye.core.interfaces import Task

# 120-word windows with a 30-word overlap. The overlap is not decoration: a
# sentence split across a boundary is a sentence neither chunk can answer, and
# the answer to a question lives in exactly one of them.
#
# The window is smaller than the documents on purpose. A window larger than your
# corpus means every document is one chunk and the chunking never runs — the
# example would pass, prove nothing, and quietly mislead the person copying it.
CHUNK_WORDS = 120
OVERLAP_WORDS = 30

_CHUNK_SCHEMA = T.ArrayType(
    T.StructType(
        [
            T.StructField("chunk_index", T.IntegerType()),
            T.StructField("chunk_text", T.StringType()),
            T.StructField("word_count", T.IntegerType()),
        ]
    )
)


def chunk_text(text: str) -> List[dict]:
    """Cut text into overlapping word windows. Pure Python — unit-testable offline."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    step = max(CHUNK_WORDS - OVERLAP_WORDS, 1)
    chunks = []
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + CHUNK_WORDS]
        if not window:
            break
        chunks.append(
            {
                "chunk_index": index,
                "chunk_text": " ".join(window),
                "word_count": len(window),
            }
        )
        if start + CHUNK_WORDS >= len(words):
            break  # the last window already ran to the end; another would repeat it
    return chunks


_chunk_udf = F.udf(chunk_text, _CHUNK_SCHEMA)


class DocumentIndex(Task):
    """Reviews + files -> documents + chunks."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        documents = self._reviews(sources["reviews"]).unionByName(self._files(sources["files"]))
        documents = documents.withColumn("ingested_at", F.current_timestamp())

        return {
            "documents": documents,
            "document_chunks": self._chunks(documents),
        }

    def _reviews(self, reviews: DataFrame) -> DataFrame:
        """Text that already lives in a table."""
        return reviews.select(
            F.sha2(F.concat_ws("::", F.lit("review"), F.col("new_id").cast("string")), 256).alias(
                "doc_id"
            ),
            F.lit("review").alias("source"),
            F.concat_ws(" ", F.lit("Review"), F.col("new_id").cast("string")).alias("title"),
            F.col("review").alias("text"),
        ).filter(F.col("text").isNotNull())

    def _files(self, files: DataFrame) -> DataFrame:
        """Raw bytes off a volume.

        `content` is the file. Decoding it is the caller's job — Spark will not
        guess an encoding for you, and for a PDF you would parse here instead.
        """
        return files.select(
            F.sha2(F.col("path"), 256).alias("doc_id"),
            F.lit("file").alias("source"),
            F.element_at(F.split(F.col("path"), "/"), -1).alias("title"),
            F.decode(F.col("content"), "utf-8").alias("text"),
        ).filter(F.length(F.col("text")) > 0)

    def _chunks(self, documents: DataFrame) -> DataFrame:
        """Explode each document into its overlapping windows."""
        return (
            documents.withColumn("chunk", F.explode(_chunk_udf(F.col("text"))))
            .select(
                F.concat_ws(
                    "::", F.col("doc_id"), F.col("chunk.chunk_index").cast("string")
                ).alias("chunk_id"),
                F.col("doc_id"),
                F.col("source"),
                F.col("title"),
                F.col("chunk.chunk_index").alias("chunk_index"),
                F.col("chunk.chunk_text").alias("chunk_text"),
                F.col("chunk.word_count").alias("word_count"),
                F.col("ingested_at"),
            )
            .filter(F.col("word_count") > 0)
        )
