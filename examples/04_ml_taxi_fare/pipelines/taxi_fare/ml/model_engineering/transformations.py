"""Model engineering: clean -> features -> split -> train -> validate -> register.

This is an ordinary Ubunye task. It has a config.yaml that says where the data
comes from and where its output goes, and this file, which holds the business
rule. That the business rule happens to train a model rather than aggregate a
table is not something the engine needs to know.

The rule ends with a decision: the model is only registered if it clears the
gate on the *test* split — data it never saw during fitting. A model that cannot
clear the gate is not an artifact, it is a failed experiment, and it should not
be in the registry pretending otherwise.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# The model lives once, at the example root, and both tasks import it from there.
# The engine puts each *task* dir on sys.path, so a plain `from model import ...`
# would mean a copy of model.py in every task dir — and two copies of the feature
# list is how a training task and an inference task quietly stop agreeing about
# what column 3 means.
_MODELS = Path(__file__).resolve().parents[4] / "models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task
from ubunye.models.gates import PromotionGate
from ubunye.models.registry import ModelRegistry, ModelStage

from taxi_fare_model import (  # noqa: E402 — after the sys.path shim above
    FEATURES,
    TARGET,
    TaxiFareModel,
    engineer_features,
)

log = logging.getLogger(__name__)

USE_CASE = "taxi_fare"

# The bar a model must clear on unseen data before anyone is allowed to use it.
# Config, not code, in the sense that it lives in one obvious place — change it
# here and the next run is judged by the new standard.
GATES = {
    "max_test_rmse": float(os.environ.get("TAXI_MAX_RMSE", "6.0")),
    "min_test_r2": float(os.environ.get("TAXI_MIN_R2", "0.55")),
}

TRAIN_FRACTION = 0.8


class TaxiFareModelEngineering(Task):
    """Produce a taxi-fare model, or fail loudly trying."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        trips = sources["trips"]

        features = engineer_features(trips).select(*FEATURES, TARGET)
        train_df, test_df = self._split(features)

        model = TaxiFareModel()
        train_metrics = model.train(train_df)
        test_metrics = model.validate(test_df)

        log.info("train: %s", train_metrics)
        log.info("test:  %s", test_metrics)

        self._gate(test_metrics)  # raises if the model is not good enough
        version = self._register(model, {**train_metrics, **test_metrics})
        self._log_to_mlflow(model, {**train_metrics, **test_metrics}, version)

        return {"model_metrics": self._metrics_table(train_df, {**train_metrics, **test_metrics}, version)}

    # -- the rule ------------------------------------------------------------

    def _split(self, features: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Deterministic split — the same rows land in the same side every run.

        randomSplit() would reshuffle on every run, so a model's test score would
        move even when nothing about the model changed, and the gate would become
        a coin toss.
        """
        bucket = F.abs(F.hash(*[F.col(c) for c in FEATURES])) % 100
        train = features.filter(bucket < int(TRAIN_FRACTION * 100))
        test = features.filter(bucket >= int(TRAIN_FRACTION * 100))
        return train, test

    def _gate(self, test_metrics: Dict[str, float]) -> None:
        """Refuse to register a model that fails on data it has not seen."""
        results = PromotionGate(GATES).evaluate(test_metrics)
        failed = [r for r in results if not r.passed]
        if failed:
            raise RuntimeError(
                "Model failed the quality gate on the test split and was not "
                "registered:\n  " + "\n  ".join(r.message for r in failed)
            )

    def _register(self, model: TaxiFareModel, metrics: Dict[str, float]) -> str:
        """Save the artifact where the inference task can find it.

        Straight to production: it already cleared the gate on unseen data, which
        is the only evidence a staging step would have waited for. Promotion
        auto-archives the incumbent, so `ubunye models rollback` restores it.
        """
        registry = ModelRegistry(_model_store())
        version = registry.register(
            use_case=USE_CASE,
            model_name=TaxiFareModel.__name__,
            version=None,  # auto patch-bump
            model=model,
            metrics=metrics,
        )
        registry.promote(
            use_case=USE_CASE,
            model_name=TaxiFareModel.__name__,
            version=version.version,
            to_stage=ModelStage.PRODUCTION,
        )
        log.info("registered %s v%s -> production", TaxiFareModel.__name__, version.version)
        return version.version

    def _log_to_mlflow(self, model: TaxiFareModel, metrics: Dict[str, float], version: str) -> None:
        """Best-effort: an MLflow outage must not lose you a good model."""
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if not experiment:
            return
        try:
            import mlflow

            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=f"taxi-fare-v{version}"):
                mlflow.log_params({k: str(v) for k, v in model.metadata()["params"].items()})
                mlflow.log_metrics(metrics)
                mlflow.set_tags({"use_case": USE_CASE, "version": version})
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow logging failed (model is registered regardless): %s", exc)

    def _metrics_table(self, any_df: DataFrame, metrics: Dict[str, float], version: str) -> DataFrame:
        """Long-form: one row per metric.

        A column per metric would change the table's schema the day someone adds
        one, and this table is appended to forever.
        """
        spark = any_df.sparkSession
        rows = [(USE_CASE, TaxiFareModel.__name__, version, k, float(v)) for k, v in metrics.items()]
        return spark.createDataFrame(
            rows, "use_case string, model_name string, version string, metric string, value double"
        ).withColumn("recorded_at", F.current_timestamp())


def _model_store() -> str:
    """The registry lives on a UC volume — the only writable path on serverless."""
    store = os.environ.get("TAXI_MODEL_STORE")
    if not store:
        raise RuntimeError(
            "TAXI_MODEL_STORE is not set. It must point at a Unity Catalog volume, "
            "e.g. /Volumes/workspace/ubunye_examples/model_store — /tmp is not "
            "writable from serverless compute."
        )
    return store
