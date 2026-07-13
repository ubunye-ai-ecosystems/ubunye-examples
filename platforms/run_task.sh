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

# --- Spark conf the PLATFORM owns, not the pipeline ----------------------------
# Delta has to be switched on for open-source Spark. Databricks ships it already
# active, so this is pure OSS-plumbing — and that means it does NOT belong in
# config.yaml.
#
# I put it in a config first and it was wrong, and CI said so: example 11 carried
# these two lines and worked, example 02 did not carry them and died with
# DELTA_CONFIGURE_SPARK_SESSION_WITH_EXTENSION_AND_CATALOG. The fix is not "add the
# boilerplate to the other nine configs". It is to notice that a pipeline should not
# have to know which Spark distribution it landed on. The platform sets this; the
# config stays clean; and no config anywhere mentions Delta-on-OSS again.
#
# `--conf` here lands in the JVM's default SparkConf, which the engine's
# SparkSession.builder inherits — so it reaches the session the engine builds itself.
CONF=(
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension"
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"
  # A real metastore, so `format: unity` — which is only ever spark.table() and
  # saveAsTable() — behaves off Databricks exactly as it does on it, and tables
  # survive between separate `ubunye run` invocations.
  --conf "spark.sql.catalogImplementation=hive"
  --conf "spark.sql.warehouse.dir=${DATA}/warehouse"
  --conf "javax.jdo.option.ConnectionURL=jdbc:derby:;databaseName=${DATA}/metastore_db;create=true"
)
export PYSPARK_SUBMIT_ARGS="--packages ${PACKAGES} ${CONF[*]} pyspark-shell"

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
