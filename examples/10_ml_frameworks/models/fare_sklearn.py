"""scikit-learn: a gradient-boosted tree.

The boring one, and — spoiler, and the whole point of this example — the one that
wins. On tabular data with six features and fifteen thousand rows, a GBM is not a
baseline you improve on with a neural net. It is the answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from fare_common import FEATURES, TARGET, metrics, to_pandas, xy
from sklearn.ensemble import HistGradientBoostingRegressor
from ubunye.models.base import UbunyeModel

FRAMEWORK = "scikit-learn"


class SklearnFareModel(UbunyeModel):
    """HistGradientBoostingRegressor. No scaling — a tree does not care about scale."""

    def __init__(self) -> None:
        self._model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
        )

    def train(self, df: Any) -> Dict[str, Any]:
        X, y = xy(df)
        self._model.fit(X, y)
        return metrics(y, self._model.predict(X), prefix="train_")

    def validate(self, df: Any) -> Dict[str, Any]:
        X, y = xy(df)
        return metrics(y, self._model.predict(X), prefix="test_")

    def predict(self, df: Any) -> Any:
        pdf = to_pandas(df).copy()
        pdf["predicted_fare"] = self._model.predict(pdf[FEATURES].to_numpy(dtype="float32"))
        return pdf

    def save(self, path: str) -> None:
        # joblib writes a plain sequential stream, which a FUSE-mounted UC volume
        # handles fine. The other two frameworks write ZIP archives, and those do not
        # — see the note in fare_keras.py.
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(Path(path) / "model.joblib"))

    @classmethod
    def load(cls, path: str) -> "SklearnFareModel":
        model = cls()
        model._model = joblib.load(str(Path(path) / "model.joblib"))
        return model

    def metadata(self) -> Dict[str, Any]:
        import sklearn

        return {
            "framework": FRAMEWORK,
            "library_version": sklearn.__version__,
            "features": FEATURES,
            "target": TARGET,
            "params": {k: str(v) for k, v in self._model.get_params().items()},
        }
