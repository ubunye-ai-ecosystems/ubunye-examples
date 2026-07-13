"""TensorFlow/Keras — the whole task.

Every line that matters is shared with the other two frameworks. What is left is
this: name a model class, hand it to the same training run. Fit, judge on unseen
rows, gate, register, promote, score.

The engine has no idea a neural network is involved, and that is the point. `config.yaml` is
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
from fare_keras import KerasFareModel  # noqa: E402


class TrainKeras(Task):
    """Train the TensorFlow/Keras model and register it."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        return benchmark.run(KerasFareModel(), sources)
