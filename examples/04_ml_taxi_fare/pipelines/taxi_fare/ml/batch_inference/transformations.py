"""Batch inference: use the production model on new instances.

The counterpart to model engineering. That task decides *whether there is a model
worth using*; this one *uses it*. Nothing here knows how the model was fitted, and
nothing here can change which model is live — that is the registry's job, so
shipping a better model is a promotion, not a deploy of this code.

The same class the training task registered is loaded here from the artifact, so
the features are computed identically by construction: both import them from the
same `model.py`.
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
from ubunye.models.registry import ModelRegistry, ModelStage

from taxi_fare_model import FEATURES, TaxiFareModel  # noqa: E402

log = logging.getLogger(__name__)

USE_CASE = "taxi_fare"


class TaxiFareBatchInference(Task):
    """Score new trips with whatever model is currently in production."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        new_trips = sources["new_trips"]

        model, version = self._load_production_model()
        # Already engineered by the feature task, and already labelled `score`.
        # Re-deriving them here is how a scoring task drifts from the model it is
        # scoring with.
        features = new_trips.select(*FEATURES, "fare_amount")

        # Score on the driver: this is a batch of tens of thousands of rows and a
        # scikit-learn model. Distributing it would mean shipping the model to
        # every executor to save milliseconds. If this table grew to millions of
        # rows, the honest fix is a Spark UDF or a served endpoint — not a bigger
        # driver.
        scored = model.predict(features.toPandas())
        spark = new_trips.sparkSession

        predictions = (
            spark.createDataFrame(scored)
            .withColumn("model_version", F.lit(version))
            .withColumn("scored_date", F.current_date())
            .withColumn("error", F.col("predicted_fare") - F.col("fare_amount"))
        )

        log.info("scored %s trips with %s v%s", predictions.count(), TaxiFareModel.__name__, version)
        return {"predictions": predictions}

    def _load_production_model(self) -> tuple[TaxiFareModel, str]:
        """Ask the registry for what is live. Fail loudly if nothing is."""
        registry = ModelRegistry(_model_store())
        try:
            artifact_path, model_version = registry.get_model(
                use_case=USE_CASE,
                model_name=TaxiFareModel.__name__,
                stage=ModelStage.PRODUCTION,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "No production model found in the registry. Run the "
                "model_engineering task first — it registers a model only if it "
                f"clears the quality gate.\n  registry: {_model_store()}\n  cause: {exc}"
            ) from exc

        log.info("loaded %s v%s from production", TaxiFareModel.__name__, model_version.version)
        return TaxiFareModel.load(artifact_path), model_version.version


def _model_store() -> str:
    store = os.environ.get("TAXI_MODEL_STORE")
    if not store:
        raise RuntimeError(
            "TAXI_MODEL_STORE is not set. It must point at the same Unity Catalog "
            "volume the model_engineering task wrote to, e.g. "
            "/Volumes/workspace/ubunye_examples/model_store"
        )
    return store
