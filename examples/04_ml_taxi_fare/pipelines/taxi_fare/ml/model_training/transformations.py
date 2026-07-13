"""Train, validate, gate, register, promote — and log the lot to MLflow.

The order is the argument. Fit on `train`. Score on `test`, which the model has
never seen. Judge *those* numbers against a bar agreed in advance. Only then
register, and only then promote.

Judging a model on its training metrics is the model marking its own homework: it
always passes, and the gate becomes decoration. So the gate here reads `test_*`
and nothing else.

A model that cannot clear the bar is not an artifact with a warning label attached
— it is a failed experiment, and it does not enter the registry at all, because
somebody downstream will eventually load "the production model" without reading
the label.
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

from taxi_fare_model import TaxiFareModel  # noqa: E402

log = logging.getLogger(__name__)

USE_CASE = "taxi_fare"

# The bar, in one obvious place. Change it here and the next run is judged by the
# new standard — no code to read, no hidden threshold buried in a function.
GATES = {
    "max_test_rmse": float(os.environ.get("TAXI_MAX_RMSE", "6.0")),
    "min_test_r2": float(os.environ.get("TAXI_MIN_R2", "0.55")),
}


class TaxiModelTraining(Task):
    """Fit -> validate -> gate -> register -> promote."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        train_df, test_df = sources["train"], sources["test"]

        model = TaxiFareModel()
        train_metrics = model.train(train_df)
        test_metrics = model.validate(test_df)  # rows the model has never seen
        metrics = {**train_metrics, **test_metrics}

        log.info("train: %s", train_metrics)
        log.info("test : %s", test_metrics)

        self._gate(test_metrics)  # raises, and nothing below runs
        version = self._register_and_promote(model, metrics)
        self._log_to_mlflow(model, metrics, version)

        return {"model_metrics": self._metrics_table(train_df, metrics, version)}

    # ------------------------------------------------------------------

    def _gate(self, test_metrics: Dict[str, float]) -> None:
        """The gate. Note it reads test_*, never train_*."""
        failed = [r for r in PromotionGate(GATES).evaluate(test_metrics) if not r.passed]
        if failed:
            raise RuntimeError(
                "The model failed its quality gate on the test split and was NOT "
                "registered. Nothing downstream will pick it up.\n  "
                + "\n  ".join(r.message for r in failed)
            )

    def _register_and_promote(self, model: TaxiFareModel, metrics: Dict[str, float]) -> str:
        """Register the artifact, then promote it to production.

        Straight to production, because it already cleared the bar on unseen data
        — which is the only evidence a staging soak would have been waiting for.
        Promotion auto-archives the incumbent, so `ubunye models rollback` puts the
        old one back if this turns out to be a mistake.
        """
        registry = ModelRegistry(_model_store())

        version = registry.register(
            use_case=USE_CASE,
            model_name=TaxiFareModel.__name__,
            version=None,  # auto patch-bump: 1.0.0, 1.0.1, ...
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
        """Best effort. An MLflow outage must not lose you a model that passed."""
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if not experiment:
            return
        try:
            import mlflow

            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=f"{TaxiFareModel.__name__}-v{version}"):
                mlflow.log_params({k: str(v) for k, v in model.metadata()["params"].items()})
                mlflow.log_metrics(metrics)
                mlflow.set_tags(
                    {
                        "use_case": USE_CASE,
                        "registry_version": version,
                        "stage": "production",
                        "gates": str(GATES),
                    }
                )
            log.info("logged run to MLflow experiment %s", experiment)
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow logging failed (the model is registered regardless): %s", exc)

    def _metrics_table(self, any_df: DataFrame, metrics: Dict[str, float], version: str) -> DataFrame:
        spark = any_df.sparkSession
        rows = [
            (USE_CASE, TaxiFareModel.__name__, version, name, float(value))
            for name, value in metrics.items()
        ]
        return spark.createDataFrame(
            rows,
            "use_case string, model_name string, version string, metric string, value double",
        ).withColumn("recorded_at", F.current_timestamp())


def _model_store() -> str:
    store = os.environ.get("TAXI_MODEL_STORE")
    if not store:
        raise RuntimeError(
            "TAXI_MODEL_STORE is not set. It must be a Unity Catalog volume — "
            "/tmp is the driver's disk and the executors cannot see it."
        )
    return store
