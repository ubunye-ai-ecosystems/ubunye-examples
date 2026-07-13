"""Label the reviews with an open-source LLM — the teacher.

This is the expensive half of distillation, and it happens once. A 70B model
labelling every review forever is a bill that never stops; a 70B model labelling
your corpus once is a fixed cost you pay to create a training set.

The prompt is constrained hard — one word, from a closed set — because an LLM
asked for a label will otherwise give you a label *and* a paragraph explaining
itself, and the paragraph will end up in your training data.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from ubunye.core.interfaces import Task

log = logging.getLogger(__name__)

TEACHER = os.environ.get("TEACHER_ENDPOINT", "databricks-meta-llama-3-1-8b-instruct")

LABELS = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = (
    "You are a sentiment classifier. Read the customer review and reply with "
    "EXACTLY ONE WORD from this list: positive, negative, neutral. "
    "No punctuation, no explanation, no other words."
)


class LabelReviews(Task):
    """Reviews -> reviews + a sentiment label from the teacher."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        reviews: DataFrame = sources["reviews"]

        rows = (
            reviews.select(
                F.col("new_id").cast("string").alias("review_id"),
                F.col("review").alias("review_text"),
            )
            .filter(F.col("review_text").isNotNull())
            .collect()
        )
        log.info("labelling %d reviews with %s", len(rows), TEACHER)

        labelled = []
        for i, row in enumerate(rows):
            label = _label_one(row["review_text"])
            labelled.append((row["review_id"], row["review_text"], label, TEACHER))
            if (i + 1) % 25 == 0:
                log.info("  %d/%d", i + 1, len(rows))

        schema = T.StructType(
            [
                T.StructField("review_id", T.StringType()),
                T.StructField("review_text", T.StringType()),
                T.StructField("label", T.StringType()),
                T.StructField("teacher_model", T.StringType()),
            ]
        )

        spark = reviews.sparkSession
        return {
            "labelled_reviews": spark.createDataFrame(labelled, schema)
            .withColumn("labelled_at", F.current_timestamp())
            # A deterministic split, decided here and written down, so the training
            # task cannot accidentally test on rows it trained on.
            .withColumn(
                "split",
                F.when(F.abs(F.hash(F.col("review_id"))) % 10 < 8, "train").otherwise("test"),
            )
        }


def _label_one(text: str) -> str:
    """Ask the teacher for one word, and refuse to accept anything else.

    A model told to answer with one word will still sometimes answer with a
    sentence. Taking the first word that matches a known label — rather than
    trusting the whole response — is the difference between a clean training set
    and one where 3% of the labels are the string "Sure! The sentiment is".
    """
    from serving import chat  # sits next to this file; the task dir is on sys.path

    # Long reviews waste tokens and add nothing: sentiment lives in the first
    # couple of hundred words.
    snippet = " ".join(text.split()[:220])
    answer = chat(TEACHER, system=SYSTEM_PROMPT, user=snippet, max_tokens=5).lower()

    for label in LABELS:
        if label in answer:
            return label

    log.warning("teacher returned an unusable label %r — defaulting to neutral", answer[:40])
    return "neutral"


def label_from_response(answer: str) -> str:
    """The parsing rule, exposed so it can be unit-tested without a cluster."""
    answer = (answer or "").lower()
    for label in LABELS:
        if label in answer:
            return label
    return "neutral"
