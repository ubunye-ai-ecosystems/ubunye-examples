"""Clean, derive, split. One table out, with every row labelled train/test/score.

Why the split is written to the table rather than decided at training time: a
split that is recomputed is a split you cannot trust. `randomSplit()` reshuffles
on every run, so a model's test score moves even when nothing about the model
changed, and the quality gate becomes a coin toss. Worse, the "unseen" rows the
scoring task uses might be rows the model was fitted on.

Deciding it once, deterministically, and writing it down means the training task
and the scoring task cannot disagree about what the model has seen.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task

# The feature derivation lives with the model, and every task that needs features
# calls it. Two copies of this logic is how a training task and a scoring task
# quietly stop agreeing about what column 3 means.
_MODELS = Path(__file__).resolve().parents[4] / "models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from taxi_fare_model import FEATURES, TARGET, engineer_features  # noqa: E402

# 70% to fit on, 15% to be judged on, 15% the model never sees at all until it is
# in production and scoring for real.
TRAIN_PCT = 70
TEST_PCT = 15


class TaxiFeatureEngineering(Task):
    """Raw trips -> model-ready features, split three ways."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        features = engineer_features(sources["trips"]).select(*FEATURES, TARGET)
        return {"taxi_features": self._split(features)}

    def _split(self, features: DataFrame) -> DataFrame:
        """Deterministic three-way split, hashed on the feature values themselves.

        The same trip always lands in the same bucket, on every run, on every
        cluster — so re-running this task does not silently move a row from the
        training set into the test set and flatter the model.
        """
        bucket = F.abs(F.hash(*[F.col(c) for c in FEATURES])) % 100

        return (
            features.withColumn(
                "split",
                F.when(bucket < TRAIN_PCT, "train")
                .when(bucket < TRAIN_PCT + TEST_PCT, "test")
                .otherwise("score"),
            )
            .withColumn("feature_set_version", F.lit("1.0.0"))
            .withColumn("engineered_at", F.current_timestamp())
        )
