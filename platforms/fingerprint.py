"""Fingerprint the output of a task, so "it ran" can be checked instead of claimed.

A badge saying *runs on 6 platforms* is decoration. This makes it a test: every
platform runs the same task and prints the same hash, and CI fails if any of them
disagrees. Without this, "portable" means "it did not crash" — which is a much
weaker claim than the one the README makes.

The hash covers the rows and the schema, sorted, so it does not depend on partition
count, executor count, file layout, or the order Spark happened to read things in.
Those all differ between a laptop and a Kubernetes cluster and none of them is a
difference in the *data*.

Usage:
    python platforms/fingerprint.py <delta-path-or-table> [more...]
"""

from __future__ import annotations

import hashlib
import json
import sys


def fingerprint(spark, target: str) -> dict:
    """Hash a Delta target's contents, independent of how it is laid out on disk."""
    df = (
        spark.read.format("delta").load(target)
        if "://" in target or target.startswith("/")
        else spark.table(target)
    )

    columns = sorted(df.columns)
    rows = [
        # Round-trip through JSON with sorted keys: two platforms must not disagree
        # because one printed a float as 1.0 and the other as 1.
        json.dumps({c: r[c] for c in columns}, sort_keys=True, default=str)
        for r in df.select(*columns).collect()
    ]
    rows.sort()  # row order is a property of the cluster, not of the data

    digest = hashlib.sha256()
    digest.update("|".join(columns).encode())
    for row in rows:
        digest.update(row.encode())

    return {"target": target, "rows": len(rows), "columns": columns, "sha256": digest.hexdigest()}


def main() -> int:
    from pyspark.sql import SparkSession

    # The verifier needs Delta switched on too — it is reading Delta tables.
    #
    # The first version of this script did a bare `.getOrCreate()` and died with
    # DELTA_CONFIGURE_SPARK_SESSION_WITH_EXTENSION_AND_CATALOG *after the pipeline
    # had already run perfectly*. The task was portable; the thing checking it was
    # not. On Databricks these two lines are a no-op — Delta is already on.
    spark = (
        SparkSession.builder.appName("ubunye:fingerprint")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    report = [fingerprint(spark, t) for t in sys.argv[1:]]
    for entry in report:
        print(f"{entry['target']}\n  rows   : {entry['rows']}\n  sha256 : {entry['sha256']}")

    # One line CI can diff across platforms.
    combined = hashlib.sha256("".join(e["sha256"] for e in report).encode()).hexdigest()
    print(f"\nFINGERPRINT={combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
