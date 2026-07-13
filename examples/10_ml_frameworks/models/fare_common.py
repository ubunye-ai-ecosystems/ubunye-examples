"""What the three frameworks share: the features, the metrics, and the scaling.

Everything in here is deliberately **pure numpy**. Not because sklearn is unavailable
— it is — but because the three models must be judged by *identical* arithmetic. If
the tree measured its RMSE with sklearn and the neural nets measured theirs with
Keras's internal metric, the leaderboard would be comparing two different numbers and
calling one of them a winner.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

FEATURES: List[str] = [
    "trip_distance",
    "trip_minutes",
    "pickup_hour",
    "pickup_dow",
    "pickup_zip",
    "dropoff_zip",
]
TARGET = "fare_amount"


def to_pandas(df: Any) -> pd.DataFrame:
    """Accept a Spark DataFrame or a pandas one, so the models are testable offline."""
    if isinstance(df, pd.DataFrame):
        return df
    return df.toPandas()


def xy(df: Any) -> tuple[np.ndarray, np.ndarray]:
    pdf = to_pandas(df)
    return (
        pdf[FEATURES].to_numpy(dtype="float32"),
        pdf[TARGET].to_numpy(dtype="float32"),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """RMSE, MAE, R², rows — computed once, the same way, for every framework."""
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()

    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}mae": float(np.mean(np.abs(err))),
        # Guard the degenerate case: a constant target makes ss_tot zero, and R²
        # becomes a division by zero rather than a bad score.
        f"{prefix}r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        f"{prefix}rows": float(len(y_true)),
    }


class Standardiser:
    """Zero-mean, unit-variance scaling — for the neural nets, not the tree.

    A gradient-boosted tree splits on thresholds; the scale of a column is
    irrelevant to it. A neural net multiplies every input by a weight, so a feature
    running to tens of thousands (``pickup_zip``) drowns one running from 0 to 24
    (``pickup_hour``), the gradients blow up, and the loss becomes NaN. That is not
    a hyperparameter problem, it is an arithmetic one.

    **The fitted mean and std are part of the model** and get saved with it. Fit the
    scaler on training data, save it, and apply the *same* one at inference. Refitting
    a scaler on the data you are scoring is the classic silent failure: nothing raises,
    the predictions are simply wrong, and they are wrong in a way that looks plausible.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mean = X.mean(axis=0)
        # A constant column has std 0. Dividing by it yields inf, then NaN weights,
        # and a model that trains "successfully" to a loss of nan.
        std = X.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "The scaler was never fitted — or was not restored when the model was "
                "loaded. Predictions would be computed on unscaled inputs and would be "
                "silently, confidently wrong."
            )
        return ((X - self.mean) / self.std).astype("float32")

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "Standardiser":
        s = cls()
        s.mean = np.asarray(data["mean"], dtype="float32")
        s.std = np.asarray(data["std"], dtype="float32")
        return s
