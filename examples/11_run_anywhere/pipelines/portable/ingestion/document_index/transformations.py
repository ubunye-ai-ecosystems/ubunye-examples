"""Files in, documents and chunks out — and the output is bit-for-bit deterministic.

That determinism is the whole point of this task. The claim "define once, run
anywhere" is only worth anything if you can *check* it, and you can only check it if
the same input produces the same output everywhere. So there is deliberately **no
`current_timestamp()` and no random anything** in here.

Every other example in this repo stamps its rows with an ingest time, which is the
right thing to do in a real pipeline and exactly the wrong thing here: a timestamp
column would make the local, Docker, Kubernetes and Databricks outputs differ on
every run, and the portability check would have nothing to compare. The fingerprint
in `platforms/fingerprint.py` hashes these rows, and the CI matrix asserts every
platform produced the *same hash*. A wall-clock column would quietly make that
assertion untestable — it would still pass, because the hash would only ever be
compared against itself.
"""

from __future__ import annotations

from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task

# Small enough that every document produces several chunks — otherwise the chunker
# never actually runs and the example proves nothing about it.
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30


class DocumentIndex(Task):
    """Text files -> one row per document, plus overlapping chunks ready to embed."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        documents = self._documents(sources["files"])
        return {
            "documents": documents,
            "document_chunks": self._chunks(documents),
        }

    def _documents(self, files: DataFrame) -> DataFrame:
        """Spark's binaryFile source gives path/modificationTime/length/content.

        `modificationTime` and `length` are dropped on purpose. A file checked out
        by git has whatever mtime the checkout gave it, so it differs between a
        laptop, a container and a cluster — carrying it through would make the
        output platform-dependent for a reason that has nothing to do with the data.
        """
        text = F.decode(F.col("content"), "utf-8")

        return (
            files.select(
                # The id is derived from the CONTENT, not the path. The same document
                # is the same document whether it arrived at file:///tmp, s3a://, or
                # /Volumes — which is precisely what makes `mode: merge` idempotent
                # across platforms as well as across runs.
                F.sha2(text, 256).alias("doc_id"),
                F.element_at(F.split(F.col("path"), "/"), -1).alias("source"),
                text.alias("text"),
            )
            .withColumn("word_count", F.size(F.split(F.trim(F.col("text")), r"\s+")))
            .filter(F.col("word_count") > 0)
        )

    def _chunks(self, documents: DataFrame) -> DataFrame:
        """Overlapping word windows.

        The overlap is the point of chunking: a sentence that straddles a boundary
        would otherwise be cut in half and both halves made meaningless. Windows that
        overlap mean every sentence survives whole in at least one of them.
        """
        words = F.split(F.trim(F.col("text")), r"\s+")
        step = CHUNK_WORDS - CHUNK_OVERLAP

        # One row per window start: 0, 90, 180, ... up to the last word.
        starts = F.sequence(
            F.lit(0),
            F.greatest(F.size(words) - F.lit(1), F.lit(0)),
            F.lit(step),
        )

        return (
            documents.withColumn("chunk_start", F.explode(starts))
            .withColumn(
                "chunk_text",
                F.array_join(F.slice(words, F.col("chunk_start") + 1, CHUNK_WORDS), " "),
            )
            .withColumn("chunk_index", (F.col("chunk_start") / F.lit(step)).cast("int"))
            .withColumn(
                "chunk_id",
                F.sha2(F.concat_ws("#", F.col("doc_id"), F.col("chunk_index").cast("string")), 256),
            )
            .select("chunk_id", "doc_id", "source", "chunk_index", "chunk_text")
            .filter(F.length(F.col("chunk_text")) > 0)
        )
