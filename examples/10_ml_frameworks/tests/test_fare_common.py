"""The shared arithmetic, tested without a cluster.

Everything here guards a failure that is **silent**. None of these bugs raise: they
produce numbers that look entirely reasonable and are wrong — which is the only kind
of bug a leaderboard cannot survive, because the whole artifact *is* a number.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"


def _load(name: str):
    """Load by path, under a unique name.

    Every example in this repo has same-named modules; under one pytest process the
    first import wins the name in sys.modules and later tests silently get someone
    else's module.
    """
    spec = importlib.util.spec_from_file_location(f"bench_{name}", MODELS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fare_common = _load("fare_common")
metrics = fare_common.metrics
Standardiser = fare_common.Standardiser


# -- the metrics ------------------------------------------------------------


def test_a_perfect_prediction_scores_perfectly():
    y = np.array([1.0, 5.0, 9.0])
    m = metrics(y, y, prefix="test_")
    assert m["test_rmse"] == pytest.approx(0.0)
    assert m["test_mae"] == pytest.approx(0.0)
    assert m["test_r2"] == pytest.approx(1.0)
    assert m["test_rows"] == 3


def test_rmse_punishes_a_single_big_miss_harder_than_mae():
    """The reason the leaderboard ranks on RMSE and not MAE.

    One catastrophic fare is worse than a hundred small errors, and RMSE is the metric
    that says so. If these two ever agree, the squaring has been lost somewhere.
    """
    y = np.zeros(100)
    y_pred = np.zeros(100)
    y_pred[0] = 50.0  # one wild miss

    m = metrics(y, y_pred)
    assert m["rmse"] > m["mae"] * 5


def test_r2_of_a_model_that_just_predicts_the_mean_is_zero():
    """R² is measured against "predict the average and go home". A model that cannot
    beat that has learned nothing, and must not score above zero."""
    y = np.array([2.0, 4.0, 6.0, 8.0])
    m = metrics(y, np.full(4, y.mean()))
    assert m["r2"] == pytest.approx(0.0)


def test_r2_does_not_divide_by_zero_on_a_constant_target():
    """A constant target makes the denominator zero. Unguarded this is a ZeroDivisionError
    (or a silent nan) rather than a score."""
    y = np.full(10, 7.0)
    m = metrics(y, np.full(10, 7.0))
    assert np.isfinite(m["r2"])


def test_misaligned_predictions_are_caught_by_the_metric():
    """Why `_score` collects the rows exactly once.

    If truths and predictions come back from two separate Spark collects in different
    orders, this is what the metric sees. It must NOT look fine — and it does not: a
    shuffled prediction scores about as well as guessing.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, size=1000)
    perfect = metrics(y, y)
    shuffled = metrics(y, rng.permutation(y))

    assert perfect["r2"] == pytest.approx(1.0)
    assert shuffled["r2"] < 0.1, "a misaligned prediction must not score like a good one"


# -- the scaler -------------------------------------------------------------


def test_standardiser_centres_and_scales():
    X = np.array([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]], dtype="float32")
    Xs = Standardiser().fit(X).transform(X)
    assert Xs.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-5)
    assert Xs.std(axis=0) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_a_constant_column_does_not_produce_nan():
    """std = 0 -> divide by zero -> inf -> NaN weights -> a net that trains to a loss
    of nan and reports it as a number. The guard is why that cannot happen."""
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]], dtype="float32")
    Xs = Standardiser().fit(X).transform(X)
    assert np.isfinite(Xs).all()


def test_the_scaler_round_trips_so_it_can_be_saved_with_the_model():
    """The scaler is part of the model. If it does not survive save/load, inference
    runs on unscaled inputs — no error, just confidently wrong numbers."""
    rng = np.random.default_rng(1)
    X = rng.normal(50, 20, size=(100, 3)).astype("float32")

    fitted = Standardiser().fit(X)
    restored = Standardiser.from_dict(fitted.to_dict())

    assert restored.transform(X) == pytest.approx(fitted.transform(X), abs=1e-5)


def test_transforming_before_fitting_raises_rather_than_guessing():
    """An unfitted scaler must not quietly pass the data through. That would be a model
    scoring raw zip codes against weights trained on standardised ones."""
    with pytest.raises(RuntimeError, match="never fitted"):
        Standardiser().transform(np.zeros((2, 3), dtype="float32"))


def test_the_scaler_must_be_fitted_on_train_and_reused_on_new_data():
    """Refitting the scaler on the data you are scoring is the classic silent bug.

    Two batches with different distributions must NOT map to the same standardised
    values — if they do, the model is being handed inputs that mean something
    different from what it learned.
    """
    train = np.array([[10.0], [20.0], [30.0]], dtype="float32")
    live = np.array([[110.0], [120.0], [130.0]], dtype="float32")  # the world moved

    scaler = Standardiser().fit(train)

    reused = scaler.transform(live)  # correct: the model's own scaler
    refitted = Standardiser().fit(live).transform(live)  # the bug

    assert not np.allclose(reused, refitted), (
        "refitting the scaler on live data hid a real shift — the model would never "
        "see that anything had changed"
    )
    assert reused.mean() > 4, "the shifted batch must land far from zero, and it does"
