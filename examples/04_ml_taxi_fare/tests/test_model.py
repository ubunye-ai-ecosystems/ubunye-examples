"""The model and its quality gate, tested on synthetic trips — no Spark, no workspace.

The gate is the part that matters: a model that cannot predict must not be
registered, because something downstream will load "the production model" and
trust it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from taxi_fare_model import FEATURES, TARGET, TaxiFareModel  # noqa: E402
from ubunye.models.gates import PromotionGate  # noqa: E402

GATES = {"max_test_rmse": 6.0, "min_test_r2": 0.55}


def _trips(n=3000, seed=42, learnable=True):
    rng = np.random.default_rng(seed)
    dist = rng.gamma(2.0, 1.5, n).clip(0.1, 40)
    df = pd.DataFrame({
        "trip_distance": dist,
        "trip_minutes": dist * rng.normal(3.2, 0.5, n).clip(1.5, 8),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_dow": rng.integers(1, 8, n),
        "pickup_zip": rng.integers(10001, 10100, n),
        "dropoff_zip": rng.integers(10001, 10100, n),
    })
    if learnable:
        df[TARGET] = (2.5 + 2.6 * df.trip_distance + 0.35 * df.trip_minutes
                      + rng.normal(0, 1.2, n)).clip(2.5, 200)
    else:
        df[TARGET] = rng.uniform(2.5, 200, n)   # fare unrelated to anything
    return df


@pytest.fixture
def split():
    df = _trips()
    return df.iloc[:2400], df.iloc[2400:]


def test_a_learnable_fare_is_learned(split):
    train, test = split
    model = TaxiFareModel()
    model.train(train)
    metrics = model.validate(test)

    assert metrics["test_r2"] > 0.8
    assert all(r.passed for r in PromotionGate(GATES).evaluate(metrics))


def test_noise_does_not_pass_the_gate():
    """If the target is unpredictable, the model must FAIL the gate — otherwise
    the gate is decoration and the registry fills up with junk."""
    df = _trips(learnable=False)
    model = TaxiFareModel()
    model.train(df.iloc[:2400])
    metrics = model.validate(df.iloc[2400:])

    assert not all(r.passed for r in PromotionGate(GATES).evaluate(metrics))


def test_save_load_roundtrip_predicts_identically(split):
    """The training task saves; the inference task loads. If they disagree, every
    prediction in production is quietly wrong."""
    train, test = split
    model = TaxiFareModel()
    model.train(train)
    before = model.predict(test)["predicted_fare"].to_numpy()

    with tempfile.TemporaryDirectory() as d:
        model.save(d)
        after = TaxiFareModel.load(d).predict(test)["predicted_fare"].to_numpy()

    np.testing.assert_allclose(before, after)


def test_metadata_declares_the_feature_contract():
    meta = TaxiFareModel().metadata()
    assert meta["features"] == FEATURES
    assert meta["library"] == "scikit-learn"
