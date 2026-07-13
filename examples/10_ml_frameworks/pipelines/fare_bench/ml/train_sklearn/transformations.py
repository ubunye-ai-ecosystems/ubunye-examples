"""scikit-learn — the whole task.

Every line that matters is shared with the other two frameworks. What is left is
this: name a model class, hand it to the same training run. Fit, judge on unseen
rows, gate, register, promote, score.

The engine has no idea a gradient-boosted tree is involved, and that is the point. `config.yaml` is
byte-for-byte identical to the other two trainers'.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from ubunye.core.interfaces import Task

_MODELS = Path(__file__).resolve().parents[4] / "models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

import benchmark  # noqa: E402
from fare_sklearn import SklearnFareModel  # noqa: E402


class TrainSklearn(Task):
    """Train the scikit-learn model and register it."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        return benchmark.run(SklearnFareModel(), sources)
