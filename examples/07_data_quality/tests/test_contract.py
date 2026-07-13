"""The contract, tested without a cluster.

The pipeline deliberately never triggers a FATAL rule — injecting a null primary
key would mean the job could never pass. So the fatal path is proven here, where a
deliberate failure IS the assertion.

These are Spark-free: the rules are Column expressions, so they are checked by
reading the contract rather than executing it. What is asserted is the *shape* of
the contract — that the severities mean what the pipeline assumes, that no rule
was left unclassified, and that the thresholds are what the code enforces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1] / "pipelines/quality/contracts/validate_transactions"
sys.path.insert(0, str(TASK))

pytest.importorskip("pyspark", reason="the contract is built from pyspark Columns")

from contract import (  # noqa: E402
    MAX_QUARANTINE_RATE,
    PRICE_TOLERANCE,
    VALID_PAYMENT_METHODS,
    rules,
)

SEVERITIES = {"fatal", "quarantine"}


def test_every_rule_has_a_severity_the_pipeline_understands():
    """A rule with an unrecognised severity is a rule that is silently ignored:
    it would never be fatal and never quarantine, and it would look fine."""
    for rule in rules():
        assert rule.severity in SEVERITIES, f"{rule.name} has severity {rule.severity!r}"


def test_identity_rules_are_fatal():
    """A row that cannot be identified or attributed is not a bad row — it is a
    broken source. Ingesting around it corrupts every downstream aggregate
    silently, so it must stop the run rather than be set aside."""
    fatal = {r.name for r in rules() if r.severity == "fatal"}

    assert "transaction_id_present" in fatal
    assert "customer_id_present" in fatal


def test_value_rules_are_quarantine_not_fatal():
    """A negative price is Tuesday. It should not take down the pipeline — the row
    goes to quarantine with the reason attached and the other 3,000 rows land."""
    quarantine = {r.name for r in rules() if r.severity == "quarantine"}

    for name in (
        "quantity_positive",
        "unit_price_not_negative",
        "total_price_reconciles",
        "payment_method_known",
        "timestamp_present",
    ):
        assert name in quarantine


def test_every_rule_explains_itself():
    """A rule whose failure you cannot explain is a rule nobody will fix. The
    description ends up in the contract_results table, where somebody reads it at
    3am without the code in front of them."""
    for rule in rules():
        assert rule.description, f"{rule.name} has no description"
        assert len(rule.description) > 30, f"{rule.name}'s description explains nothing"


def test_rule_names_are_unique():
    """Duplicate names would collide in the report and one would be invisible."""
    names = [r.name for r in rules()]
    assert len(names) == len(set(names))


def test_price_tolerance_is_a_cent_not_a_bit():
    """Cross-field arithmetic is never exact in floating point: 3 x 3.30 is not
    always 9.90. A tolerance of zero would quarantine honest rows forever."""
    assert 0 < PRICE_TOLERANCE <= 0.01


def test_quarantine_limit_is_a_limit():
    """The point of the limit is to distinguish 'some bad rows' from 'the source
    changed'. Set at 1.0 it can never trip; at 0.0 a single bad row kills the run."""
    assert 0 < MAX_QUARANTINE_RATE < 0.5


def test_payment_methods_are_a_closed_set():
    assert VALID_PAYMENT_METHODS
    assert all(m == m.lower() for m in VALID_PAYMENT_METHODS), "the rule lowercases the column"
