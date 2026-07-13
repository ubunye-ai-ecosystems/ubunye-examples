"""The contract: what a valid transaction *is*.

Kept in one file, separate from the pipeline that enforces it, because a contract
is a statement about the data and not a step in a job. You should be able to read
this file and know what the table promises without reading any Spark.

Each rule is a name, a Spark condition that TRUE means valid, and a severity:

  ``fatal``     a breach means the source is broken in a way you cannot ingest
                around — the run stops. A missing primary key is not a row
                problem, it is a schema problem.
  ``quarantine`` the row is wrong but the pipeline is fine. Set it aside with the
                reason attached and keep going.

The severity is the interesting decision, and it is a judgement, not a technicality:
how bad does this have to be before nobody downstream should see *any* of today's
data?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from pyspark.sql import Column

# Cross-field arithmetic is never exact in floating point: 3 x 3.30 is not always
# 9.90. A tolerance of a cent says "the books balance", not "the bits match".
PRICE_TOLERANCE = 0.01

VALID_PAYMENT_METHODS = ["visa", "mastercard", "amex", "discover"]

# If more than this share of rows breaks the contract, something has changed at
# the source and quietly quarantining a third of the day's data is not a fix.
MAX_QUARANTINE_RATE = 0.05


@dataclass(frozen=True)
class Rule:
    name: str
    condition: Callable[[], Column]  # returns a Column; TRUE = the row is valid
    severity: str  # "fatal" | "quarantine"
    description: str

    def check(self) -> Column:
        """Build the Spark expression. Called only when there is a session."""
        return self.condition()


# The conditions are CALLABLES, not Columns. Building `F.col("x") > 0` at import
# time needs a live SparkContext — which would mean the contract could not be read,
# reviewed, or unit-tested without a cluster. A contract nobody can inspect without
# spinning up Spark is a contract nobody will review.


def rules() -> List[Rule]:
    """The contract, as executable rules."""
    from pyspark.sql import functions as F  # noqa: PLC0415 — needs a live session

    return [
        # --- fatal: the row cannot be identified or joined ------------------
        Rule(
            name="transaction_id_present",
            condition=lambda: F.col("transactionID").isNotNull(),
            severity="fatal",
            description="Every transaction must have an id. Without one it cannot "
            "be merged, deduplicated, or corrected later.",
        ),
        Rule(
            name="customer_id_present",
            condition=lambda: F.col("customerID").isNotNull(),
            severity="fatal",
            description="A transaction with no customer cannot be attributed. "
            "Revenue per customer would silently under-count.",
        ),
        # --- quarantine: the row is wrong, the pipeline is not ---------------
        Rule(
            name="quantity_positive",
            condition=lambda: F.col("quantity") > 0,
            severity="quarantine",
            description="You cannot sell zero or minus one croissant.",
        ),
        Rule(
            name="unit_price_not_negative",
            condition=lambda: F.col("unitPrice") >= 0,
            severity="quarantine",
            description="A negative price is a refund wearing a sale's clothes, and "
            "it belongs in a different table.",
        ),
        Rule(
            name="total_price_reconciles",
            condition=lambda: F.abs(
                F.col("totalPrice") - (F.col("quantity") * F.col("unitPrice"))
            )
            <= PRICE_TOLERANCE,
            severity="quarantine",
            description="totalPrice must equal quantity x unitPrice. This is the rule "
            "that catches a source system whose columns drifted apart — each field is "
            "individually plausible and together they do not add up.",
        ),
        Rule(
            name="payment_method_known",
            condition=lambda: F.lower(F.col("paymentMethod")).isin(VALID_PAYMENT_METHODS),
            severity="quarantine",
            description="A payment method nobody has seen before is either a new "
            "product decision or a corrupted field, and you want to know which.",
        ),
        Rule(
            name="timestamp_present",
            condition=lambda: F.col("dateTime").isNotNull(),
            severity="quarantine",
            description="A transaction with no time cannot be partitioned or "
            "reported on by period.",
        ),
    ]
