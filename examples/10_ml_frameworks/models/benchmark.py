"""The training run, once — shared by all three frameworks.

Fit on `train`. Judge on `test`, which the model has never seen. Gate on *those*
numbers. Register only if it earns it. Then score the `score` split — rows that were
in neither the fitting nor the gating — and record that too, because that is the
number the leaderboard is decided on.

Judging the frameworks on the split the gate already looked at would be picking the
winner on the exam it was allowed to resit.

**These are plain functions, deliberately not a Task subclass.** The engine's
``_load_task_class`` returns *the first Task subclass it finds in the module*, so a
``transformations.py`` that imports a shared Task base would hand the engine the base
class instead of its own task — and the engine would run the wrong thing, silently
and successfully. So the sharing happens through functions, and each task declares
its own Task.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.models.gates import PromotionGate
from ubunye.models.registry import ModelRegistry, ModelStage

log = logging.getLogger(__name__)

USE_CASE = "fare_benchmark"

# One bar, and all three frameworks are held to it. A per-framework gate would let a
# weak model through by lowering the bar it was judged against, which is not a gate,
# it is a formality.
GATES = {
    "max_test_rmse": float(os.environ.get("BENCH_MAX_RMSE", "6.0")),
    "min_test_r2": float(os.environ.get("BENCH_MIN_R2", "0.55")),
}


def model_store() -> str:
    store = os.environ.get("BENCH_MODEL_STORE")
    if not store:
        raise RuntimeError(
            "BENCH_MODEL_STORE is not set. It must be a Unity Catalog volume — /tmp is "
            "the driver's local disk and nothing else can see it."
        )
    return store


def run(model: Any, sources: Dict[str, Any]) -> Dict[str, DataFrame]:
    """Train -> validate -> gate -> register -> promote -> score. One framework."""
    train_df, test_df, score_df = sources["train"], sources["test"], sources["score"]
    framework = model.metadata()["framework"]
    name = type(model).__name__

    started = time.time()
    train_metrics = model.train(train_df)
    train_seconds = time.time() - started

    test_metrics = model.validate(test_df)  # rows it has never seen

    log.info("[%s] train: %s", framework, train_metrics)
    log.info("[%s] test : %s  (%.1fs to fit)", framework, test_metrics, train_seconds)

    _gate(framework, test_metrics)  # raises — nothing below runs

    version = _register(model, name, {**train_metrics, **test_metrics})

    # The honest tiebreak: a split neither fitted on nor gated against.
    score_metrics = _score(model, score_df)
    log.info("[%s] score: %s", framework, score_metrics)

    _log_to_mlflow(model, framework, name, {**test_metrics, **score_metrics}, version)

    return {
        "fare_bench_metrics": _metrics_table(
            train_df,
            framework=framework,
            model_name=name,
            version=version,
            metrics={**train_metrics, **test_metrics, **score_metrics},
            train_seconds=train_seconds,
        )
    }


# ---------------------------------------------------------------------------


def _gate(framework: str, test_metrics: Dict[str, float]) -> None:
    """The gate reads test_*, never train_*. A model marking its own homework always passes."""
    failed = [r for r in PromotionGate(GATES).evaluate(test_metrics) if not r.passed]
    if failed:
        raise RuntimeError(
            f"The {framework} model failed its quality gate on the test split and was "
            "NOT registered. It cannot appear on the leaderboard, because a leaderboard "
            "of models nobody would ship is a leaderboard of nothing.\n  "
            + "\n  ".join(r.message for r in failed)
        )


def _register(model: Any, name: str, metrics: Dict[str, float]) -> str:
    """Each framework is its own model_name, under one shared use_case.

    So they version independently — retraining the Keras model does not bump the
    sklearn one — while still living in one registry you can list and compare.
    """
    registry = ModelRegistry(model_store())

    version = registry.register(
        use_case=USE_CASE,
        model_name=name,
        version=None,  # auto patch-bump
        model=model,
        metrics=metrics,
    )
    registry.promote(
        use_case=USE_CASE,
        model_name=name,
        version=version.version,
        to_stage=ModelStage.PRODUCTION,
    )
    log.info("registered %s v%s -> production", name, version.version)
    return version.version


def _score(model: Any, score_df: DataFrame) -> Dict[str, float]:
    """Score the ranking split — collecting the rows exactly ONCE.

    The obvious way to write this is a bug:

        _X, y = xy(score_df)              # toPandas() -> one collect
        scored = model.predict(score_df)  # toPandas() -> ANOTHER collect

    Two independent collects of the same Spark DataFrame. Spark guarantees no row
    order without an ORDER BY, so the truths and the predictions can come back in
    different orders — and then every score_* metric is computed by comparing row 1's
    fare against row 900's prediction. Nothing raises. The numbers look entirely
    plausible, and they are meaningless.

    So: collect once, and pass the *same* pandas frame to both. The models take
    pandas or Spark, which is exactly what that flexibility is for.
    """
    from fare_common import TARGET, metrics as compute_metrics, to_pandas  # noqa: PLC0415

    pdf = to_pandas(score_df)  # the one and only collect
    scored = model.predict(pdf)

    return compute_metrics(
        pdf[TARGET].to_numpy(),
        scored["predicted_fare"].to_numpy(),
        prefix="score_",
    )


def _log_to_mlflow(
    model: Any, framework: str, name: str, metrics: Dict[str, float], version: str
) -> None:
    """Best effort. An MLflow outage must not lose a model that passed its gate."""
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not experiment:
        return
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"{name}-v{version}"):
            mlflow.log_params(model.metadata()["params"])
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "use_case": USE_CASE,
                    "framework": framework,
                    "registry_version": version,
                    "stage": "production",
                }
            )
        log.info("[%s] logged to MLflow experiment %s", framework, experiment)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] MLflow logging failed (the model is registered anyway): %s", framework, exc)


def _metrics_table(
    any_df: DataFrame,
    *,
    framework: str,
    model_name: str,
    version: str,
    metrics: Dict[str, float],
    train_seconds: float,
) -> DataFrame:
    """Long form: one row per metric. A column per metric changes the table's schema
    the day somebody adds one, and this table is appended to forever."""
    rows: list[Tuple[str, str, str, str, float, float]] = [
        (framework, model_name, version, metric, float(value), float(train_seconds))
        for metric, value in metrics.items()
    ]
    return any_df.sparkSession.createDataFrame(
        rows,
        "framework string, model_name string, version string, metric string, "
        "value double, train_seconds double",
    ).withColumn("recorded_at", F.current_timestamp())
