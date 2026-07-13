"""Rank the frameworks, name a champion.

Ranked on **score_rmse** — the split that no model was fitted on and no gate was
judged against. Ranking on `test_rmse` would be picking the winner on the exam it was
allowed to resit: the gate already read those numbers, and a model that squeaked
through by overfitting the test split would be rewarded for it.

Nothing here loads a model. It pivots a table of numbers that three tasks already
wrote — which is why this task needs neither torch, nor tensorflow, nor sklearn
installed. See config.yaml.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task

log = logging.getLogger(__name__)

# The metric the championship is decided on. One line, in the open, so nobody has to
# go looking for where the winner was really chosen.
RANK_BY = "score_rmse"

# How far ahead the winner has to be before "winner" means anything.
#
# This is the honest part of a leaderboard, and the part leaderboards leave out. Sort
# any three models by any metric and one of them comes first — that is arithmetic, not
# evidence. If the gap to the runner-up is a couple of percent, what you have measured
# is the seed, the split, and the weather. Say so, rather than crowning it.
DECISIVE_MARGIN_PCT = 5.0

REPORTED = [
    "train_rmse",
    "test_rmse",
    "test_mae",
    "test_r2",
    "score_rmse",
    "score_mae",
    "score_r2",
]


class SelectChampion(Task):
    """Pivot the metrics into a leaderboard and flag the winner."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        metrics: DataFrame = sources["metrics"]

        board = (
            metrics.groupBy("framework", "model_name", "version", "train_seconds")
            .pivot("metric", REPORTED)
            .agg(F.first("value"))
        )

        # A framework that trained but never wrote score_rmse did not finish. Ranking
        # it as "best" on a null would hand it the trophy for not turning up.
        ranked = board.filter(F.col(RANK_BY).isNotNull())

        scores = sorted(r[RANK_BY] for r in ranked.select(RANK_BY).collect())
        if not scores:
            raise RuntimeError(
                "No framework produced a " + RANK_BY + ". Every trainer either failed its "
                "quality gate or never ran — there is nothing to rank."
            )

        best = scores[0]
        runner_up = scores[1] if len(scores) > 1 else None

        # How much better is the winner than whoever came second? That is the only
        # number that says whether "first place" means anything at all.
        margin_pct = 100.0 * (runner_up - best) / runner_up if runner_up else 100.0
        decisive = margin_pct >= DECISIVE_MARGIN_PCT

        leaderboard = (
            ranked.withColumn("is_champion", F.col(RANK_BY) == F.lit(best))
            .withColumn("rank", F.row_number().over(Window.orderBy(RANK_BY)))
            .withColumn("ranked_on", F.lit(RANK_BY))
            .withColumn("margin_pct", F.lit(float(margin_pct)))
            .withColumn(
                "verdict",
                F.lit("clear win" if decisive else "too close to call"),
            )
            .withColumn("decided_at", F.current_timestamp())
        )

        for row in leaderboard.orderBy("rank").collect():
            log.info(
                "%d. %-14s %s v%s  %s=%.4f  (%.1fs to fit)%s",
                row["rank"],
                row["framework"],
                row["model_name"],
                row["version"],
                RANK_BY,
                row[RANK_BY],
                row["train_seconds"],
                "  <- champion" if row["is_champion"] else "",
            )

        if not decisive:
            log.warning(
                "The winner is %.1f%% ahead of the runner-up, under the %.0f%% needed to "
                "call it. These models are TIED. Sorting three numbers always produces a "
                "first place; that is arithmetic, not evidence. Choose on what actually "
                "differs — training cost, dependency weight, who has to maintain it.",
                margin_pct,
                DECISIVE_MARGIN_PCT,
            )

        return {"fare_bench_leaderboard": leaderboard}
