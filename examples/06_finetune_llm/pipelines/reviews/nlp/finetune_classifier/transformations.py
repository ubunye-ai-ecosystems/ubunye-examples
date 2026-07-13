"""Fine-tune the student, judge it against the teacher, and register it if it earns it.

The lifecycle is identical to the sklearn model in example 04 — split, fit, judge
on unseen data, gate, register, promote — because it *is* identical. That the
learner is a transformer rather than a gradient-boosted tree changes how it fits,
and nothing about how it must be governed.

The gate here asks a distillation-specific question: does the student agree with
the teacher on reviews it has never seen? A student that cannot reproduce its
teacher out of sample has not compressed anything. It is merely cheaper, and wrong.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task
from ubunye.models.gates import PromotionGate
from ubunye.models.registry import ModelRegistry, ModelStage

_MODELS = Path(__file__).resolve().parents[4] / "models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from review_sentiment_model import ReviewSentimentModel  # noqa: E402

log = logging.getLogger(__name__)

USE_CASE = "review_sentiment"

# The bar. Two gates, and the second is the one that matters.
#
# Accuracy alone is not enough: if the teacher had labelled 90% of the corpus
# "positive", a student that answered "positive" to everything would score 0.90
# and be worthless. Per-class recall catches exactly that — a collapsed model
# scores ~1.0 on the majority class and ~0.0 on the minority one, and this gate
# refuses it.
GATES = {
    "min_test_accuracy": float(os.environ.get("STUDENT_MIN_ACCURACY", "0.70")),
    "min_test_recall_negative": float(os.environ.get("STUDENT_MIN_RECALL_NEG", "0.50")),
}


class FinetuneClassifier(Task):
    """Fine-tune DistilBERT on the teacher's labels."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        labelled: DataFrame = sources["labelled"]

        train_df = labelled.filter(F.col("split") == "train").select("review_text", "label")
        test_df = labelled.filter(F.col("split") == "test").select("review_text", "label")

        n_train, n_test = train_df.count(), test_df.count()
        log.info("fine-tuning on %d reviews, judging on %d unseen", n_train, n_test)
        if n_train < 20 or n_test < 5:
            raise RuntimeError(
                f"Not enough labelled data to fine-tune ({n_train} train / {n_test} test). "
                "Run the label_reviews task first."
            )

        model = ReviewSentimentModel()
        train_metrics = model.train(train_df)
        test_metrics = model.validate(test_df)  # agreement with the teacher, out of sample
        metrics = {**train_metrics, **test_metrics}

        log.info("train: %s", train_metrics)
        log.info("test : %s", test_metrics)

        self._gate(test_metrics)
        version = self._register(model, metrics)
        self._log_to_mlflow(model, metrics, version)

        return {"student_metrics": self._metrics_table(labelled, metrics, version)}

    def _gate(self, test_metrics: Dict[str, float]) -> None:
        failed = [r for r in PromotionGate(GATES).evaluate(test_metrics) if not r.passed]
        if failed:
            raise RuntimeError(
                "The student failed to reproduce the teacher on held-out reviews and "
                "was NOT registered:\n  " + "\n  ".join(r.message for r in failed)
            )

    def _register(self, model: ReviewSentimentModel, metrics: Dict[str, float]) -> str:
        registry = ModelRegistry(_model_store())
        version = registry.register(
            use_case=USE_CASE,
            model_name=ReviewSentimentModel.__name__,
            version=None,
            model=model,
            metrics=metrics,
        )
        registry.promote(
            use_case=USE_CASE,
            model_name=ReviewSentimentModel.__name__,
            version=version.version,
            to_stage=ModelStage.PRODUCTION,
        )
        log.info("registered %s v%s -> production", ReviewSentimentModel.__name__, version.version)
        return version.version

    def _log_to_mlflow(
        self, model: ReviewSentimentModel, metrics: Dict[str, float], version: str
    ) -> None:
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if not experiment:
            return
        try:
            import mlflow

            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=f"distilbert-student-v{version}"):
                mlflow.log_params({k: str(v) for k, v in model.metadata()["params"].items()})
                mlflow.log_metrics(metrics)
                mlflow.set_tags(
                    {
                        "use_case": USE_CASE,
                        "teacher": os.environ.get("TEACHER_ENDPOINT", "llama-3.1-8b"),
                        "student": "distilbert-base-uncased",
                        "registry_version": version,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow logging failed (the model is registered regardless): %s", exc)

    def _metrics_table(
        self, any_df: DataFrame, metrics: Dict[str, float], version: str
    ) -> DataFrame:
        spark = any_df.sparkSession
        rows = [
            (USE_CASE, ReviewSentimentModel.__name__, version, name, float(value))
            for name, value in metrics.items()
        ]
        return spark.createDataFrame(
            rows,
            "use_case string, model_name string, version string, metric string, value double",
        ).withColumn("recorded_at", F.current_timestamp())


def _model_store() -> str:
    store = os.environ.get("STUDENT_MODEL_STORE")
    if not store:
        raise RuntimeError(
            "STUDENT_MODEL_STORE is not set. It must be a Unity Catalog volume — "
            "the fine-tuned weights are a few hundred megabytes and /tmp is not "
            "writable from serverless."
        )
    return store
