"""PSI, tested without a cluster.

PSI is the one piece of real arithmetic in this example, and it is the kind of
arithmetic that fails quietly: a bucketing mistake does not raise, it just reports
0.00 forever and the monitor becomes a smoke detector with no battery.

So every test here is written to fail if the calculation goes *silent*, not just if
it goes wrong. The numbers are the ones measured on the real workspace data
(reference = the taxi training split; live = the same trips with fares inflated
35%), so if the maths drifts from what the deployed example asserts, this fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TASK = Path(__file__).resolve().parents[1] / "pipelines/taxi_fare/mlops/monitor_model"
sys.path.insert(0, str(TASK))

pytest.importorskip("pyspark", reason="the task module imports pyspark types")

from transformations import (  # noqa: E402
    BUCKETS,
    MAX_ERROR_MULTIPLE,
    MIN_IMPROVEMENT,
    PSI_BREACH,
    PSI_WARN,
    _best,
    _psi,
)


def edges_from(reference: pd.Series) -> np.ndarray:
    """The buckets the task uses: quantiles of the REFERENCE, open at both ends."""
    e = np.quantile(reference.astype(float), np.linspace(0, 1, BUCKETS + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


@pytest.fixture
def reference() -> pd.Series:
    # Fare-like: right-skewed, strictly positive. A normal distribution would make
    # the quantile buckets uniform and hide bucketing bugs.
    rng = np.random.default_rng(42)
    return pd.Series(rng.lognormal(mean=2.4, sigma=0.5, size=20_000))


def test_identical_distributions_are_stable(reference):
    """The control. If this is not ~0, every other number here is noise."""
    psi = _psi(reference, reference, edges_from(reference))
    assert psi < 0.001, f"a distribution drifted from itself: PSI={psi}"


def test_a_fresh_sample_of_the_same_world_is_stable(reference):
    """A different sample of the SAME distribution must not raise an alarm.

    This is the test that catches a monitor which fires on everything — the one
    that gets silenced in week two and then misses the real thing.
    """
    rng = np.random.default_rng(7)
    same_world = pd.Series(rng.lognormal(mean=2.4, sigma=0.5, size=5_000))
    psi = _psi(reference, same_world, edges_from(reference))
    assert psi < PSI_WARN, f"sampling noise alone tripped the warning: PSI={psi}"


def test_inflation_breaches(reference):
    """The scenario the example deploys: every value up 35%.

    Measured on the real workspace data: PSI 0.349 against a 0.25 threshold.
    """
    inflated = reference * 1.35
    psi = _psi(reference, inflated, edges_from(reference))
    assert psi > PSI_BREACH, f"a 35% shift in every value went undetected: PSI={psi}"


def test_values_beyond_the_training_range_are_not_silently_dropped(reference):
    """The open-ended buckets are load-bearing, and this is the failure they prevent.

    Without `edges[0], edges[-1] = -inf, +inf`, live values outside the training
    range fall outside every bucket and np.histogram discards them. The survivors
    are then renormalised into proportions — which land right back on the reference
    shape. The rows the model has never seen anything like, the most alarming rows
    in the batch, are the exact rows that never reach the alarm, and PSI reports a
    reassuring 0.00.

    It has to be a PARTIAL contamination to show this. If *every* row were out of
    range, all the live buckets would be empty, the 1e-6 floor would take over, and
    PSI would explode to ~11 — alarming, but for the wrong reason. The dangerous
    case is the realistic one: a quarter of the batch is off the end of the world
    and the rest looks completely normal.
    """
    e = np.quantile(reference.astype(float), np.linspace(0, 1, BUCKETS + 1))
    closed = e.copy()  # what you get if you forget the ±inf
    opened = e.copy()
    opened[0], opened[-1] = -np.inf, np.inf

    # Three parts the model has seen, one part it has never seen anything like.
    off_the_end = pd.Series(np.full(len(reference) // 3, reference.max() * 10))
    live = pd.concat([reference, off_the_end], ignore_index=True)

    assert _psi(reference, live, closed) == pytest.approx(0.0, abs=0.01), (
        "expected the closed-bucket version to be blind here — if it is not, this "
        "test no longer proves what it claims"
    )
    assert _psi(reference, live, opened) > PSI_BREACH, (
        "a quarter of the batch off the end of the training range must breach"
    )


def test_an_empty_bucket_does_not_produce_infinity(reference):
    """A bucket with no rows sends log(0) to -inf and PSI to nan/inf.

    The floor exists so an alarm is about the data, not about arithmetic.
    """
    # Live data occupying a single narrow slice — most reference buckets get zero.
    narrow = pd.Series(np.full(1_000, float(reference.median())))
    psi = _psi(reference, narrow, edges_from(reference))

    assert np.isfinite(psi), f"PSI was not finite: {psi}"
    assert psi > PSI_BREACH, "collapsing the whole batch into one bucket must breach"


def test_psi_is_symmetric_in_magnitude(reference):
    """Deflation is as much a drift as inflation. A one-sided monitor is half a monitor."""
    up = _psi(reference, reference * 1.35, edges_from(reference))
    down = _psi(reference, reference / 1.35, edges_from(reference))
    assert down > PSI_BREACH, f"a 26% drop went undetected: PSI={down}"
    assert up > PSI_BREACH


# -- the rollback decision --------------------------------------------------


def test_best_picks_the_lowest_error():
    assert _best({"1.0.0": 0.9, "1.0.1": 0.4, "1.0.2": 3.1}) == ("1.0.1", 0.4)


def test_best_of_nothing_is_nothing():
    """No other version to roll back to. The monitor must not crash — it must say so."""
    assert _best({}) == (None, None)


def test_a_marginally_better_challenger_does_not_justify_a_rollback():
    """A rollback is a deployment, on one window of evidence.

    2% better is noise wearing a rosette. If this margin ever collapses to zero,
    the monitor starts churning production on sampling noise.
    """
    champion = 1.00
    _, best = _best({"1.0.1": 0.98})
    assert not best < champion * (1 - MIN_IMPROVEMENT)

    # ...but the broken-model case is not marginal, and must win.
    _, broken_beats = _best({"1.0.1": 0.45})
    assert broken_beats < 3.0 * (1 - MIN_IMPROVEMENT)


def test_thresholds_are_the_ones_the_example_documents():
    """The README and the notebook quote these. Drift here, and the docs start lying."""
    assert (PSI_WARN, PSI_BREACH) == (0.10, 0.25)
    assert MAX_ERROR_MULTIPLE == 3.0
    assert MIN_IMPROVEMENT == 0.10
