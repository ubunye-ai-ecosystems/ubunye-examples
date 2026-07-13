"""The model itself — fit, predict, score. Nothing else.

An ``UbunyeModel`` knows how to learn and how to predict. It does not know what
a registry is, what MLflow is, or which stage it is deployed to. That is the
engine's and the task's business, not the model's, which is why this file has no
imports from ubunye beyond the contract it implements.

The same class is loaded by the inference task to score new instances — it is the
artifact, not a script.
"""

from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ubunye.models.base import UbunyeModel

FEATURES: List[str] = [
    "trip_distance",
    "trip_minutes",
    "pickup_hour",
    "pickup_dow",
    "pickup_zip",
    "dropoff_zip",
]
TARGET = "fare_amount"


def engineer_features(trips: Any) -> Any:
    """Derive the model's features from raw trips. Spark in, Spark out.

    This lives with the model, and both the training task and the inference task
    call it. Two copies of this function is how a model silently starts scoring
    garbage: the columns still line up, they just mean something else.
    """
    from pyspark.sql import functions as F

    return (
        trips.withColumn(
            "trip_minutes",
            (
                F.col("tpep_dropoff_datetime").cast("long")
                - F.col("tpep_pickup_datetime").cast("long")
            )
            / 60.0,
        )
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow", F.dayofweek("tpep_pickup_datetime"))
        # A trip that took no time, or half a day, is a data-entry artefact, not a
        # taxi ride. Left in, they teach the model nonsense.
        .filter(F.col("trip_minutes").between(1, 180))
        .dropna(subset=FEATURES + [TARGET])
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}rows": float(len(y_true)),
    }


class TaxiFareModel(UbunyeModel):
    """Predicts a NYC taxi fare from distance, duration and time of day."""

    def __init__(self) -> None:
        self._model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
        )

    # -- learn ---------------------------------------------------------------

    def train(self, df: Any) -> Dict[str, Any]:
        """Fit on the training split. Returns the metrics scored on that split.

        These are training metrics — the model has seen every row of this data.
        They tell you the model *can* learn. They tell you nothing about whether
        it will hold up, which is what :meth:`validate` is for.
        """
        pdf = _to_pandas(df)
        self._model.fit(pdf[FEATURES], pdf[TARGET])
        preds = self._model.predict(pdf[FEATURES])
        return _metrics(pdf[TARGET].to_numpy(), preds, prefix="train_")

    # -- score honestly ------------------------------------------------------

    def validate(self, df: Any) -> Dict[str, Any]:
        """Score a holdout the model has never seen. This is the number that counts."""
        pdf = _to_pandas(df)
        preds = self._model.predict(pdf[FEATURES])
        return _metrics(pdf[TARGET].to_numpy(), preds, prefix="test_")

    # -- use -----------------------------------------------------------------

    def predict(self, df: Any) -> Any:
        """Score new instances. Returns the input plus a `predicted_fare` column."""
        pdf = _to_pandas(df)
        pdf = pdf.copy()
        pdf["predicted_fare"] = self._model.predict(pdf[FEATURES])
        return pdf

    # -- persist -------------------------------------------------------------

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(Path(path) / "model.joblib"))

    @classmethod
    def load(cls, path: str) -> "TaxiFareModel":
        from pathlib import Path

        model = cls()
        model._model = joblib.load(str(Path(path) / "model.joblib"))
        return model

    def metadata(self) -> Dict[str, Any]:
        import sklearn

        return {
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
            "features": FEATURES,
            "target": TARGET,
            "params": self._model.get_params(),
        }


def _to_pandas(df: Any) -> pd.DataFrame:
    """Accept a Spark DataFrame or a pandas one, so the model is testable offline."""
    if isinstance(df, pd.DataFrame):
        return df
    return df.toPandas()
