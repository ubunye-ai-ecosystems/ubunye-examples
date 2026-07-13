"""Enforce the contract: split, quarantine, report, and fail when it matters.

Three principles, and each one is an argument against something people actually do:

**Quarantine, do not drop.** A dropped row is a silent data loss — the counts move,
nobody knows why, and the evidence is gone. A quarantined row is a bug report with
the payload attached, and somebody can fix the source.

**Fail loudly when the contract is broken, not when the data is merely bad.** A few
malformed rows is Tuesday. A missing primary key column, or a third of the day's
rows failing, is a source system that has changed under you — and quietly setting
aside a third of the data is not a fix, it is a cover-up.

**Report every rule, every run, even the ones that passed.** A rule that fails zero
rows today and 4,000 tomorrow is the earliest possible warning that something
upstream moved. You only see that if you write down the zero.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task

from contract import MAX_QUARANTINE_RATE, Rule, rules  # noqa: E402 — sits next to this file

log = logging.getLogger(__name__)


class ValidateTransactions(Task):
    """Apply the data contract to incoming transactions."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        # The faults are deliberately broken rows (see config.yaml). Without them
        # the quarantine path would never execute on this sample data, and an
        # example whose failure path never runs proves nothing.
        incoming = sources["transactions"].unionByName(sources["faults"])
        total = incoming.count()

        contract = rules()
        checked = self._apply(incoming, contract)

        clean = checked.filter(F.size("broken_rules") == 0).drop("broken_rules", "fatal_rules")
        quarantined = checked.filter(F.size("broken_rules") > 0)

        results = self._report(incoming, contract, total)
        self._enforce(quarantined, contract, total)

        n_quarantined = quarantined.count()
        log.info("contract: %d rows in, %d quarantined", total, n_quarantined)

        return {
            "clean_transactions": clean.withColumn("validated_at", F.current_timestamp()),
            "quarantined_transactions": quarantined.withColumn(
                "quarantined_at", F.current_timestamp()
            ),
            "contract_results": results,
        }

    # ------------------------------------------------------------------

    def _apply(self, df: DataFrame, contract: List[Rule]) -> DataFrame:
        """Tag every row with the rules it breaks.

        One pass, and the row carries its own verdict. Filtering the DataFrame once
        per rule would read the data seven times and lose the ability to say "this
        row broke three rules" — which is exactly what you want to know, because a
        row that breaks three rules is usually one bug, not three.
        """
        broken = F.array_compact(
            F.array(
                *[
                    F.when(~rule.check() | rule.check().isNull(), F.lit(rule.name))
                    for rule in contract
                ]
            )
        )
        fatal_names = [r.name for r in contract if r.severity == "fatal"]

        return df.withColumn("broken_rules", broken).withColumn(
            "fatal_rules", F.array_intersect(F.col("broken_rules"), F.array(*map(F.lit, fatal_names)))
        )

    def _report(self, df: DataFrame, contract: List[Rule], total: int) -> DataFrame:
        """One row per rule per run — including the rules that passed.

        The zeros are the point. A rule that fails nothing today and 4,000 rows
        tomorrow is the earliest warning you will get that a source system changed,
        and you cannot see that change if you only record failures.
        """
        rows = []
        for rule in contract:
            failed = df.filter(~rule.check() | rule.check().isNull()).count()
            rows.append(
                (
                    rule.name,
                    rule.severity,
                    rule.description,
                    int(total),
                    int(failed),
                    float(failed / total) if total else 0.0,
                    failed == 0,
                )
            )

        spark = df.sparkSession
        return spark.createDataFrame(
            rows,
            "rule string, severity string, description string, rows_checked long, "
            "rows_failed long, failure_rate double, passed boolean",
        ).withColumn("checked_at", F.current_timestamp())

    def _enforce(self, quarantined: DataFrame, contract: List[Rule], total: int) -> None:
        """Stop the run when the breach is structural rather than incidental."""
        fatal = quarantined.filter(F.size("fatal_rules") > 0)
        n_fatal = fatal.count()

        if n_fatal:
            broken = [r.name for r in contract if r.severity == "fatal"]
            raise RuntimeError(
                f"{n_fatal} row(s) broke a FATAL rule ({', '.join(broken)}). "
                "A row that cannot be identified or attributed is not a bad row, it is a "
                "broken source — ingesting around it would corrupt every downstream "
                "aggregate silently. Fix the source, or amend the contract on purpose."
            )

        n_quarantined = quarantined.count()
        rate = n_quarantined / total if total else 0.0

        if rate > MAX_QUARANTINE_RATE:
            raise RuntimeError(
                f"{n_quarantined}/{total} rows ({rate:.1%}) failed the contract, over the "
                f"{MAX_QUARANTINE_RATE:.0%} limit. At this scale it is not bad rows, it is a "
                "changed source — and quietly setting aside a fifth of the day's data is "
                "not a fix, it is a cover-up."
            )
