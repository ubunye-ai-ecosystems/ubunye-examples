"""PyTorch: a small multi-layer perceptron, trained on CPU.

Same contract, same features, same metrics as the sklearn model — so the leaderboard
is comparing the *frameworks*, not two different experiments wearing the same name.

Two things here are not optional, and both are silent when you get them wrong:

* **The inputs are standardised.** ``pickup_zip`` runs to tens of thousands;
  ``pickup_hour`` runs 0-23. Unscaled, the zip code dominates every dot product, the
  gradients explode, and the loss goes to NaN. The tree does not care. The net does.
* **The scaler is saved with the weights.** It is part of the model. Restore the
  weights without it and inference runs on unscaled inputs — no error, just
  confidently wrong numbers.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from fare_common import FEATURES, TARGET, Standardiser, metrics, to_pandas, xy
from torch import nn
from ubunye.models.base import UbunyeModel

FRAMEWORK = "pytorch"

EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HIDDEN = (64, 32)
SEED = 42

# Serverless gives us a many-core box with a modest driver. Left alone, torch spins
# up a worker per core, each with its own arena, for a model this small — all cost,
# no speed. See the same cap, for the same reason, in fare_keras.py.
TORCH_THREADS = 2


def _net() -> nn.Module:
    layers: list[nn.Module] = []
    prev = len(FEATURES)
    for width in HIDDEN:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


class TorchFareModel(UbunyeModel):
    """A 6 -> 64 -> 32 -> 1 MLP in PyTorch."""

    def __init__(self) -> None:
        torch.manual_seed(SEED)
        torch.set_num_threads(TORCH_THREADS)
        self._net = _net()
        self._scaler = Standardiser()

    # -- learn ---------------------------------------------------------------

    def train(self, df: Any) -> Dict[str, Any]:
        X, y = xy(df)
        Xs = self._scaler.fit(X).transform(X)

        features = torch.from_numpy(Xs)
        target = torch.from_numpy(y).unsqueeze(1)

        optimiser = torch.optim.Adam(self._net.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()

        self._net.train()
        n = len(features)
        for _epoch in range(EPOCHS):
            # Shuffle each epoch. Without it the net sees the rows in the same order
            # every pass and happily learns the order along with the data.
            order = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                optimiser.zero_grad()
                loss = loss_fn(self._net(features[batch]), target[batch])
                loss.backward()
                optimiser.step()

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
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.from_numpy(self._scaler.transform(X)))
        return out.squeeze(1).numpy()

    # -- persist -------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write locally, then copy onto the volume.

        ``torch.save`` writes a ZIP container, and a Unity Catalog volume is a FUSE
        mount that does not support the seeks a zip needs. Writing straight to it
        fails with ``OSError: [Errno 95] Operation not supported`` — measured, not
        assumed. Local disk first, then a plain sequential copy.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "model.pt")
            torch.save(self._net.state_dict(), local)
            shutil.copyfile(local, str(target / "model.pt"))

        # The scaler is part of the model, not a note about it. JSON is a sequential
        # write, so this one can go straight onto the volume.
        (target / "scaler.json").write_text(json.dumps(self._scaler.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TorchFareModel":
        source = Path(path)
        model = cls()
        model._net.load_state_dict(torch.load(str(source / "model.pt"), map_location="cpu"))
        model._scaler = Standardiser.from_dict(
            json.loads((source / "scaler.json").read_text(encoding="utf-8"))
        )
        return model

    def metadata(self) -> Dict[str, Any]:
        return {
            "framework": FRAMEWORK,
            "library_version": torch.__version__,
            "features": FEATURES,
            "target": TARGET,
            "params": {
                "epochs": str(EPOCHS),
                "batch_size": str(BATCH_SIZE),
                "learning_rate": str(LEARNING_RATE),
                "hidden": str(HIDDEN),
                "parameters": str(sum(p.numel() for p in self._net.parameters())),
            },
        }
