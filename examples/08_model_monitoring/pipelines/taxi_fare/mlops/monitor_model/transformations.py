"""Watch the live model, work out what is actually wrong, and do the right thing.

Three questions, and they are three because no two of them catch the same failure:

**Has the world changed?** — drift. Compare what the model is scoring now against
what it was trained on (PSI). Split the answer in two, because it matters which
half moved:

  * *input* drift — the questions changed. Distances, durations, times of day.
  * *target* drift — the answers changed. The same trip now costs more.

**Has the model got worse than it was?** — decay. Its live error against the error
it earned on its own test set when it was promoted. The baseline comes from the
registry, so a model is always judged against what it actually proved it could do.

**Is the live model still the best model I have?** — champion vs challengers. Load
every registered version, score them all on *the same live window*, and compare.
This is the only question whose answer justifies a rollback, and it is the one
almost nobody asks.

That last point is the whole example. **A rollback only ever fixes one thing: a
model that is worse than a model you already had.** It cannot fix a changed world
— every version in the registry learned the old world, so they are all equally
wrong, and rolling between them is theatre with a changelog. A monitor that
answers every alarm with "roll back" is as useless as one that answers with
nothing.

So this task measures which case it is, and then acts:

    challenger is materially better  ->  ROLL BACK to it. A real, measured fix.
    otherwise, but drift or decay    ->  RETRAIN. And say so, with the numbers
                                         that prove a rollback would be futile.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task
from ubunye.models.registry import ModelRegistry, ModelStage

log = logging.getLogger(__name__)

USE_CASE = "taxi_fare"
MODEL_NAME = "TaxiFareModel"

# Drift is measured on the model's numeric inputs and on its target, kept apart on
# purpose — see the module docstring, and the truth table in the README.
#
# pickup_zip and dropoff_zip are deliberately absent. They are categorical: a zip
# code is a label, not a quantity, and cutting one into deciles asks whether zip
# 10023 is "bigger" than 10012. Categorical drift needs a comparison of category
# shares, which is a different calculation and not this example's point.
INPUTS = ["trip_distance", "trip_minutes", "pickup_hour", "pickup_dow"]
TARGET = "fare_amount"

# PSI — Population Stability Index. The industry rule of thumb, and it is only a
# rule of thumb:
#   < 0.10     the world is the same
#   0.10-0.25  it has moved; watch it
#   > 0.25     it has moved enough that the model is answering a different question
PSI_WARN = 0.10
PSI_BREACH = 0.25
BUCKETS = 10

# How much worse than its own test-set error the model may get before somebody is
# woken up. Twice is noise; three times is a different model.
MAX_ERROR_MULTIPLE = 3.0

# How much better a challenger must be before it is worth churning production.
# Rolling back is not free — it is a deployment, on evidence from one window. A
# challenger that is 2% better is noise wearing a rosette.
MIN_IMPROVEMENT = 0.10


class MonitorModel(Task):
    """Drift, decay, champion-vs-challengers — and the right remedy for each."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        spark = sources["live"].sparkSession
        reference = sources["reference"].toPandas()
        live = sources["live"].toPandas()

        registry = ModelRegistry(_model_store())
        _, champion = registry.get_model(
            use_case=USE_CASE, model_name=MODEL_NAME, stage=ModelStage.PRODUCTION
        )

        drift = self._drift(reference, live)
        champion_mae = self._live_mae(registry, champion.version, live)
        challengers = self._challengers(registry, champion.version, live)
        decay = self._decay(champion, champion_mae)

        action = self._respond(registry, champion, champion_mae, challengers, drift, decay)

        return {
            "model_monitoring": self._monitoring_table(spark, drift, champion, champion_mae, decay),
            "model_candidates": self._candidate_table(
                spark, champion, champion_mae, challengers, action
            ),
            "monitoring_incidents": self._incident_table(
                spark, drift, decay, champion, champion_mae, challengers, action
            ),
        }

    # -- has the world changed? ------------------------------------------------

    def _drift(self, reference: pd.DataFrame, live: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """PSI per column, bucketed on the REFERENCE distribution.

        The buckets must come from the reference. Bucket each side on its own
        quantiles and you are measuring two things with two different rulers —
        every feature then looks perfectly stable, forever, including the ones
        that have moved.
        """
        results: List[Tuple[str, str, float]] = []
        for column, kind in [(c, "input") for c in INPUTS] + [(TARGET, "target")]:
            edges = np.quantile(
                reference[column].dropna().astype(float), np.linspace(0, 1, BUCKETS + 1)
            )
            # Live values outside the training range have to land somewhere. Without
            # this they fall outside every bucket, are silently dropped, and the
            # most alarming rows in the batch — the ones the model has never seen
            # anything like — are the exact rows that never reach the alarm.
            edges[0], edges[-1] = -np.inf, np.inf

            psi = _psi(reference[column], live[column], edges)
            results.append((column, kind, psi))
            log.info("PSI %-14s %-6s %.4f", column, kind, psi)

        return results

    # -- has the model got worse than it was? ----------------------------------

    def _decay(self, champion: Any, champion_mae: float) -> Dict[str, Any]:
        """Live error vs the error THIS version earned on its own test set.

        The baseline is read from the registry, not hard-coded here, so the model
        is judged against what it actually proved — not against a number somebody
        typed in a config a year ago.

        Note what this check cannot see: a model that was always bad. Its live
        error matches its (bad) baseline, so nothing has "decayed" and this check
        stays quiet. That is not a bug — it was born broken, it did not rot — and
        it is precisely why the champion-vs-challenger check below exists.
        """
        baseline = float(champion.metrics.get("test_mae") or 0.0)
        limit = baseline * MAX_ERROR_MULTIPLE
        breached = baseline > 0 and champion_mae > limit

        log.info(
            "MAE: live %.3f vs baseline %.3f (limit %.3f) -> %s",
            champion_mae,
            baseline,
            limit,
            "BREACH" if breached else "ok",
        )
        return {
            "baseline_mae": baseline,
            "live_mae": champion_mae,
            "limit": limit,
            "breached": bool(breached),
        }

    # -- is the live model still the best one I have? --------------------------

    def _challengers(
        self, registry: ModelRegistry, champion_version: str, live: pd.DataFrame
    ) -> Dict[str, float]:
        """Score every OTHER registered version on the same live window.

        Not on its own test set — on today's data. A version's recorded metrics
        describe the world it was trained in; the only way to know whether it would
        do better *now* is to run it *now*.
        """
        scores: Dict[str, float] = {}
        for version in registry.list_versions(USE_CASE, MODEL_NAME):
            if version.version == champion_version:
                continue
            scores[version.version] = self._live_mae(registry, version.version, live)
            log.info("challenger %-8s live MAE %.3f", version.version, scores[version.version])
        return scores

    def _live_mae(self, registry: ModelRegistry, version: str, live: pd.DataFrame) -> float:
        """Load a version and score it on the live window.

        ``validate()`` is the model's own "score data you have never seen" method.
        The live window is exactly that, so this is what it is for.
        """
        model_class = _model_class()
        path, _ = registry.get_model(use_case=USE_CASE, model_name=MODEL_NAME, version=version)
        return float(model_class.load(path).validate(live)["test_mae"])

    # -- do the right thing about it -------------------------------------------

    def _respond(
        self,
        registry: ModelRegistry,
        champion: Any,
        champion_mae: float,
        challengers: Dict[str, float],
        drift: List[Tuple[str, str, float]],
        decay: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Choose the remedy that fits the diagnosis — and refuse the one that doesn't.

        This is the step MLflow cannot take for you: it does not know what is
        serving, and it cannot change it.
        """
        best_version, best_mae = _best(challengers)
        # The margin is what makes this a decision and not a coin toss. A rollback
        # is a deployment, justified by one window of data.
        wins = best_mae is not None and best_mae < champion_mae * (1.0 - MIN_IMPROVEMENT)

        breached = [f for f, _kind, psi in drift if psi > PSI_BREACH]
        alarm = bool(breached) or decay["breached"] or wins

        if not alarm:
            return {"action": "none", "diagnosis": "healthy", "detail": "within thresholds"}

        if wins and os.environ.get("MONITOR_ROLLBACK", "true").lower() != "true":
            return {
                "action": "alert_only",
                "diagnosis": "better_version_available",
                "detail": (
                    f"{best_version} scores {best_mae:.3f} vs production {champion_mae:.3f} on "
                    "today's data, but MONITOR_ROLLBACK=false — a human decides"
                ),
            }

        if wins:
            registry.rollback(
                use_case=USE_CASE, model_name=MODEL_NAME, to_version=str(best_version)
            )
            log.warning("ROLLED BACK production: %s -> %s", champion.version, best_version)
            return {
                "action": "rollback",
                "diagnosis": "regression",
                "detail": (
                    f"production {champion.version} scores {champion_mae:.3f} on today's data; "
                    f"{best_version} scores {best_mae:.3f} on the same rows. Rolled back — the "
                    "live model was worse than one we already had."
                ),
            }

        # Something breached, and nothing in the registry does better. Rolling back
        # would move production for no gain: every version learned the same old
        # world. Say so, with the number that proves it.
        best_note = (
            f"the best of {len(challengers)} other version(s) scores {best_mae:.3f} vs "
            f"production {champion_mae:.3f} — no better"
            if best_mae is not None
            else "there is no other version to roll back to"
        )
        return {
            "action": "retrain",
            "diagnosis": "world_moved" if breached else "decayed",
            "detail": (
                f"drift on {breached or 'nothing'}, live MAE {champion_mae:.3f} vs baseline "
                f"{decay['baseline_mae']:.3f}. A rollback would not help: {best_note}. "
                "Retrain on recent data."
            ),
        }

    # -- what we write down ----------------------------------------------------

    def _monitoring_table(
        self,
        spark: Any,
        drift: List[Tuple[str, str, float]],
        champion: Any,
        champion_mae: float,
        decay: Dict[str, Any],
    ) -> DataFrame:
        rows = [
            (
                feature,
                kind,
                float(psi),
                "breach" if psi > PSI_BREACH else ("warn" if psi > PSI_WARN else "stable"),
                champion.version,
                decay["baseline_mae"],
                champion_mae,
                decay["breached"],
            )
            for feature, kind, psi in drift
        ]
        return spark.createDataFrame(
            rows,
            "feature string, kind string, psi double, drift_status string, model_version string, "
            "baseline_mae double, live_mae double, decay_detected boolean",
        ).withColumn("monitored_at", F.current_timestamp())

    def _candidate_table(
        self,
        spark: Any,
        champion: Any,
        champion_mae: float,
        challengers: Dict[str, float],
        action: Dict[str, Any],
    ) -> DataFrame:
        best_version, _ = _best(challengers)
        rows = [(champion.version, "production", float(champion_mae), False)] + [
            (version, "challenger", float(mae), version == best_version)
            for version, mae in challengers.items()
        ]
        return spark.createDataFrame(
            rows,
            "model_version string, role string, live_mae double, best_challenger boolean",
        ).withColumn("action", F.lit(action["action"])).withColumn(
            "evaluated_at", F.current_timestamp()
        )

    def _incident_table(
        self,
        spark: Any,
        drift: List[Tuple[str, str, float]],
        decay: Dict[str, Any],
        champion: Any,
        champion_mae: float,
        challengers: Dict[str, float],
        action: Dict[str, Any],
    ) -> DataFrame:
        drifted_inputs = [f for f, kind, psi in drift if kind == "input" and psi > PSI_BREACH]
        drifted_target = any(kind == "target" and psi > PSI_BREACH for _f, kind, psi in drift)
        best_version, best_mae = _best(challengers)

        rows = [
            (
                champion.version,
                bool(drifted_inputs),
                drifted_inputs,
                bool(drifted_target),
                decay["breached"],
                decay["baseline_mae"],
                float(champion_mae),
                best_version,
                float(best_mae) if best_mae is not None else None,
                action["diagnosis"],
                action["action"],
                action["detail"],
            )
        ]
        return spark.createDataFrame(
            rows,
            "model_version string, input_drift boolean, drifted_inputs array<string>, "
            "target_drift boolean, decay_detected boolean, baseline_mae double, live_mae double, "
            "best_challenger string, best_challenger_mae double, diagnosis string, "
            "action string, detail string",
        ).withColumn("detected_at", F.current_timestamp())


# ---------------------------------------------------------------------------


def _psi(reference: pd.Series, current: pd.Series, edges: np.ndarray) -> float:
    """Population Stability Index between two distributions over fixed buckets."""
    ref_counts, _ = np.histogram(reference.dropna().astype(float), bins=edges)
    cur_counts, _ = np.histogram(current.dropna().astype(float), bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # An empty bucket sends the log to infinity. Floor both sides: the alternative
    # is an alarm about arithmetic rather than about the data.
    floor = 1e-6
    ref_pct = np.clip(ref_pct, floor, None)
    cur_pct = np.clip(cur_pct, floor, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _best(challengers: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    if not challengers:
        return None, None
    version = min(challengers, key=lambda v: challengers[v])
    return version, challengers[version]


def _model_class():
    """The model class from example 04 — this task monitors 04's model.

    Imported from 04 rather than copied. Two copies of a model class is how a
    registry starts loading an artifact into a class that no longer matches it:
    the joblib still deserialises, the columns still line up, and the predictions
    quietly mean something else.
    """
    models_dir = os.environ.get("TAXI_MODELS_DIR")
    if not models_dir:
        raise RuntimeError(
            "TAXI_MODELS_DIR is not set. The monitor scores every registered version on "
            "today's data, so it needs example 04's model class to load them with."
        )
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    from taxi_fare_model import TaxiFareModel  # noqa: PLC0415 — path injected just above

    return TaxiFareModel


def _model_store() -> str:
    store = os.environ.get("TAXI_MODEL_STORE")
    if not store:
        raise RuntimeError(
            "TAXI_MODEL_STORE is not set. The monitor reads the production model's own "
            "test metrics from the registry — the baseline it is judged against is the "
            "number it earned when it was promoted."
        )
    return store
