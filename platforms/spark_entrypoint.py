"""The entry point a CLOUD runs. Not the CLI.

This is the gap nobody notices until the first cloud bill.

Everything else in this repo invokes the engine through `ubunye run` — a console
script, in a shell, on a machine you can `ssh` into. **AWS EMR Serverless and GCP
Dataproc Serverless do not do that.** They take a *Python file*, hand it to
`spark-submit`, and run it. There is no shell, no console script on PATH, no `-t`
flags. If the engine can only be reached through its CLI, it cannot run on either of
them — and you would only find out after wiring up IAM, a bucket, and a billing
account.

So: this file, and the `spark-submit` job in CI that runs it *for free*, on the same
invocation shape the cloud uses. The one remaining unknown for AWS and GCP is then
"does the cloud accept our job", not "does our code work when submitted this way".

    spark-submit platforms/spark_entrypoint.py \\
        --task-dir examples/11_run_anywhere/pipelines/portable/ingestion/document_index \\
        --mode PROD --dt 2026-07-13

Note what it does NOT do: create a SparkSession. `spark-submit` has already made one,
and the engine attaches to it (`_detect_backend` finds the active session). Building a
second one here would either fail or — worse — quietly ignore the cluster's conf.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an Ubunye task under spark-submit.")
    parser.add_argument("--task-dir", required=True, help="the task directory to run")
    parser.add_argument("--mode", default="PROD")
    parser.add_argument("--dt", default="2026-07-13")
    parser.add_argument(
        "--fingerprint",
        nargs="*",
        default=[],
        help="Delta targets to fingerprint after the run (for the portability check).",
    )
    args = parser.parse_args()

    import ubunye

    print(f"running {args.task_dir} (mode={args.mode}, dt={args.dt})", flush=True)
    ubunye.run_task(task_dir=args.task_dir, dt=args.dt, mode=args.mode, lineage=False)

    if args.fingerprint:
        # Prove what landed, in the cloud, from inside the cloud. A cloud job that
        # "succeeded" and wrote nothing is the most convincing green build there is.
        import hashlib

        from pyspark.sql import SparkSession

        from fingerprint import fingerprint  # shipped alongside via --py-files

        spark = SparkSession.builder.getOrCreate()
        report = [fingerprint(spark, t) for t in args.fingerprint]
        for entry in report:
            print(f"{entry['target']}\n  rows   : {entry['rows']}\n  sha256 : {entry['sha256']}")

        combined = hashlib.sha256("".join(e["sha256"] for e in report).encode()).hexdigest()
        print(f"\nFINGERPRINT={combined}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
