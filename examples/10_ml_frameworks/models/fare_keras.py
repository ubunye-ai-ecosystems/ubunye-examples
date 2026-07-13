"""TensorFlow / Keras: the same MLP again, in the third framework.

This file is where the environment bites, and both bites are silent. Everything here
was measured on the workspace, not read in a doc.

**1. Cap the thread pools before you build anything.**
Serverless hands you a many-core box with a modest driver (16 GB total, ~6 GB free).
TensorFlow's default ``intra_op`` parallelism is "one thread per core", and each
thread gets its own allocation arena. For a model this small that buys no speed at
all and it OOMs the driver — the job dies with *Execution ran out of memory*, which
reads like the data was too big. It was not. Capped, this trains 15k rows in about
seven seconds at a flat 2.3 GB.

**2. Never call ``model.save()`` straight onto a Unity Catalog volume.**
A ``.keras`` file is a ZIP archive, and writing a zip requires seeking. A UC volume
is a FUSE mount that does not support it, and the call fails with
``OSError: [Errno 95] Operation not supported``. This matters because the engine's
``ModelRegistry`` hands ``save()`` a path *on the volume*. So: write to local disk,
then copy the finished file across. Reading back is fine — it is only the write.

**3. TensorFlow's install is not free either.**
Serverless is ``aarch64``. ``tensorflow-cpu`` publishes no ARM wheels at all
("No matching distribution found"). Plain ``tensorflow`` installs — and drags in
numpy 2 and protobuf 7, which quietly break scipy, scikit-learn and MLflow in the
same environment while pip cheerfully prints "Successfully installed". Pin
``numpy<2`` and ``protobuf<5`` and everything coexists. See the notebook.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from fare_common import FEATURES, TARGET, Standardiser, metrics, to_pandas, xy
from ubunye.models.base import UbunyeModel

FRAMEWORK = "tensorflow"

EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HIDDEN = (64, 32)
SEED = 42

# See the module docstring. This is not tuning — uncapped, the driver runs out of
# memory and the job dies.
INTRA_OP_THREADS = 2
INTER_OP_THREADS = 1

tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)


class KerasFareModel(UbunyeModel):
    """The same 6 -> 64 -> 32 -> 1 MLP as the PyTorch model, in Keras."""

    def __init__(self) -> None:
        tf.keras.utils.set_random_seed(SEED)
        self._model: Any = None
        self._scaler = Standardiser()

    def _build(self) -> Any:
        layers = [tf.keras.layers.Input(shape=(len(FEATURES),))]
        for width in HIDDEN:
            layers.append(tf.keras.layers.Dense(width, activation="relu"))
        layers.append(tf.keras.layers.Dense(1))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="mse",
        )
        return model

    # -- learn ---------------------------------------------------------------

    def train(self, df: Any) -> Dict[str, Any]:
        X, y = xy(df)
        Xs = self._scaler.fit(X).transform(X)

        self._model = self._build()
        self._model.fit(Xs, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)

        return metrics(y, self._raw_predict(X), prefix="train_")

    def validate(self, df: Any) -> Dict[str, Any]:
        X, y = xy(df)
        return metrics(y, self._raw_predict(X), prefix="test_")

    def predict(self, df: Any) -> Any:
        pdf = to_pandas(df).copy()
        X = pdf[FEATURES].to_numpy(dtype="float32")
        pdf["predicted_fare"] = self._raw_predict(X)
        return pdf

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(self._scaler.transform(X), verbose=0)
        return np.asarray(preds).ravel()

    # -- persist -------------------------------------------------------------

    def save(self, path: str) -> None:
        """Local disk first, then copy. See point 2 in the module docstring."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "model.keras")
            self._model.save(local)
            shutil.copyfile(local, str(target / "model.keras"))

        (target / "scaler.json").write_text(json.dumps(self._scaler.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "KerasFareModel":
        source = Path(path)
        model = cls()
        # Reading a zip off the volume is fine — it is only writing one that is not.
        model._model = tf.keras.models.load_model(str(source / "model.keras"))
        model._scaler = Standardiser.from_dict(
            json.loads((source / "scaler.json").read_text(encoding="utf-8"))
        )
        return model

    def metadata(self) -> Dict[str, Any]:
        return {
            "framework": FRAMEWORK,
            "library_version": tf.__version__,
            "features": FEATURES,
            "target": TARGET,
            "params": {
                "epochs": str(EPOCHS),
                "batch_size": str(BATCH_SIZE),
                "learning_rate": str(LEARNING_RATE),
                "hidden": str(HIDDEN),
                "parameters": str(int(self._model.count_params())) if self._model else "0",
            },
        }
