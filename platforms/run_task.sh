#!/usr/bin/env bash
# Run ANY example's task on open-source Spark — local, in Docker, or in a Kubernetes
# pod. One script, because the platforms differ in where they stand, not in what they
# do.
#
#   platforms/run_task.sh <example-dir> <usecase> <package> <task> [more tasks...]
#
# e.g. platforms/run_task.sh examples/02_ingest_rest_api weather ingestion hourly_forecast
set -euo pipefail

EXAMPLE="${1:?example dir, e.g. examples/02_ingest_rest_api}"
USECASE="${2:?usecase}"
PACKAGE="${3:?package}"
shift 3
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || { echo "at least one task required" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-/tmp/ubunye}"

# --- the three variables that ARE the portability surface ----------------------
export SPARK_MASTER="${SPARK_MASTER:-local[*]}"
export UBUNYE_SINK="${UBUNYE_SINK:-s3}"        # generic path connector, nothing to do with AWS
export UBUNYE_DATA_ROOT="${UBUNYE_DATA_ROOT:-file://${DATA}}"

# --- JARs Spark needs, which pip cannot give it --------------------------------
# Delta and the JDBC drivers are JVM artifacts. `pip install delta-spark` ships the
# Python half and none of the JAR, and the engine builds its own SparkSession — so
# PYSPARK_SUBMIT_ARGS is the only channel that reaches the classpath. Get this wrong
# and you get ClassNotFoundException at RUNTIME, never at install time.
SPARK_MAJOR="$(python -c 'import pyspark;print(pyspark.__version__.split(".")[0])')"
case "$SPARK_MAJOR" in
  3) DELTA_PKG="io.delta:delta-spark_2.12:${DELTA_VERSION:-3.2.0}" ;;
  4) DELTA_PKG="io.delta:delta-spark_2.13:${DELTA_VERSION:-4.0.0}" ;;
  *) echo "unsupported pyspark major: $SPARK_MAJOR" >&2; exit 1 ;;
esac

# The Postgres driver, for the JDBC example. Note what this line means: adding a JDBC
# driver to open-source Spark is one Maven coordinate. It is Databricks SERVERLESS that
# cannot do it — which is why example 09 runs here and not there.
PACKAGES="${DELTA_PKG},org.postgresql:postgresql:42.7.4"
export PYSPARK_SUBMIT_ARGS="--packages ${PACKAGES} pyspark-shell"

mkdir -p "${DATA}"
echo "example   : ${EXAMPLE}"
echo "tasks     : ${TASKS[*]}"
echo "master    : ${SPARK_MASTER}"
echo "sink      : ${UBUNYE_SINK}"
echo "data root : ${UBUNYE_DATA_ROOT}"

# --- run the SAME task directory Databricks runs -------------------------------
cd "$ROOT"
TASK_ARGS=()
for t in "${TASKS[@]}"; do TASK_ARGS+=(-t "$t"); done

ubunye run \
  -d "${EXAMPLE}/pipelines" \
  -u "${USECASE}" \
  -p "${PACKAGE}" \
  "${TASK_ARGS[@]}" \
  -m PROD \
  -dt "${DT:-2026-07-13}"
