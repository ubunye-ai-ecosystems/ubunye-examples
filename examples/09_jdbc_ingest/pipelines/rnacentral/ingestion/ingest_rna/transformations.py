"""Profile what came back over the wire.

The business rule here is deliberately thin. JDBC is the lesson, not the arithmetic:
the interesting decisions in this task were all made in ``config.yaml`` — pushing the
projection down, slicing the read across four connections, and capping the fetch size
so the driver does not swallow the result set whole.

The one thing worth asserting in code is that the parallel read actually happened.
``numPartitions: 4`` is a *hint*: Spark silently ignores it if ``partitionColumn`` is
missing or the bounds do not make sense, and you get the single-threaded read you were
trying to avoid — with no warning, no error, and a job that is simply slow forever.
So the task checks, and says so out loud.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task

log = logging.getLogger(__name__)

# The bands are the biology, roughly: micro-RNAs are tiny, transfer RNAs are small,
# ribosomal RNAs and long non-coding RNAs run to thousands of bases.
LENGTH_BANDS = [
    (0, 50, "tiny (<50)"),
    (50, 200, "small (50-200)"),
    (200, 1000, "medium (200-1k)"),
    (1000, 10000, "long (1k-10k)"),
    (10000, 10**9, "very long (>10k)"),
]


class IngestRna(Task):
    """Ingest the RNAcentral dimension table and a slice of its sequences."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        databases: DataFrame = sources["databases"]
        sequences: DataFrame = sources["sequences"]

        partitions = sequences.rdd.getNumPartitions()
        log.info("the sequence read came back in %d partitions", partitions)
        if partitions < 2:
            # Not a crash — the data is fine. But the read was single-threaded, which
            # on the full 54M-row table is the difference between minutes and hours,
            # and it is invisible unless somebody looks.
            log.warning(
                "The JDBC read used ONE connection. partitionColumn/lowerBound/"
                "upperBound/numPartitions were ignored — check they are all set and "
                "that partitionColumn is numeric."
            )

        return {
            "rna_databases": self._databases(databases),
            "rna_sequence_profile": self._profile(sequences, partitions),
        }

    def _databases(self, databases: DataFrame) -> DataFrame:
        """The contributing databases, trimmed to the columns worth keeping.

        `alive` is 'Y'/'N' text in the source. Carrying a character flag into a
        lakehouse table and making every downstream query remember to compare it to
        the string 'Y' is how a boolean quietly becomes a footgun.
        """
        return (
            databases.select(
                F.col("id").cast("long").alias("id"),
                F.col("descr").alias("code"),
                F.col("display_name"),
                (F.upper(F.col("alive")) == "Y").alias("is_active"),
                F.col("num_sequences").cast("long").alias("num_sequences"),
                F.col("avg_length").cast("long").alias("avg_length"),
                F.col("current_release").cast("int").alias("current_release"),
            )
            .withColumn("ingested_at", F.current_timestamp())
        )

    def _profile(self, sequences: DataFrame, partitions: int) -> DataFrame:
        """Length distribution of the slice we pulled."""
        band = F.lit(None).cast("string")
        for low, high, label in reversed(LENGTH_BANDS):
            band = F.when(
                (F.col("len") >= low) & (F.col("len") < high), F.lit(label)
            ).otherwise(band)

        return (
            sequences.withColumn("length_band", band)
            .groupBy("length_band")
            .agg(
                F.count("*").alias("sequences"),
                F.min("len").alias("min_length"),
                F.round(F.avg("len"), 1).alias("avg_length"),
                F.max("len").alias("max_length"),
            )
            .withColumn("read_partitions", F.lit(partitions))
            .withColumn("ingested_at", F.current_timestamp())
        )
