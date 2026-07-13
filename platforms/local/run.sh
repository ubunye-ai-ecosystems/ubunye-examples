#!/usr/bin/env bash
# Run the portable task on plain open-source Spark. No Databricks, no cloud, no cluster.
#
# This is the baseline every other platform is compared against: whatever hash comes
# out of here is the hash Docker, Kubernetes and Databricks must also produce.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${UBUNYE_DATA_ROOT_DIR:-/tmp/ubunye}"

# --- the three variables that are the entire portability surface ---------------
export SPARK_MASTER="${SPARK_MASTER:-local[*]}"
export UBUNYE_SINK="${UBUNYE_SINK:-s3}"          # generic path connector, not AWS
export UBUNYE_DATA_ROOT="file://${DATA}"

# Delta on OSS Spark is a JAR, not a pip package. pyspark fetches it from Maven on
# first run; the version MUST match Spark's major or it fails at RUNTIME with
# ClassNotFoundException: delta.DefaultSource -- not at install time, which is the
# nastiest way for a version mismatch to present itself.
SPARK_MAJOR="$(python -c 'import pyspark;print(pyspark.__version__.split(".")[0])')"
case "$SPARK_MAJOR" in
  3) DELTA_PKG="io.delta:delta-spark_2.12:3.2.0" ;;
  4) DELTA_PKG="io.delta:delta-spark_2.13:4.0.0" ;;
  *) echo "unsupported pyspark major: $SPARK_MAJOR" >&2; exit 1 ;;
esac
export PYSPARK_SUBMIT_ARGS="--packages ${DELTA_PKG} pyspark-shell"

echo "master    : $SPARK_MASTER"
echo "sink      : $UBUNYE_SINK"
echo "data root : $UBUNYE_DATA_ROOT"
echo "delta     : $DELTA_PKG"

# --- stage the corpus where the config expects it ------------------------------
# The pipeline reads $UBUNYE_DATA_ROOT/corpus on EVERY platform. Putting the files
# there is the platform's job, not the pipeline's -- and that split is the honest
# one: the bootstrap differs, the pipeline does not.
rm -rf "${DATA}/corpus" "${DATA}/documents" "${DATA}/document_chunks"
mkdir -p "${DATA}/corpus"
cp "$ROOT"/examples/11_run_anywhere/data/corpus/*.txt "${DATA}/corpus/"

# --- run the SAME task directory every other platform runs ---------------------
cd "$ROOT"
ubunye run \
  -d examples/11_run_anywhere/pipelines \
  -u portable \
  -p ingestion \
  -t document_index \
  -m PROD \
  -dt "${DT:-2026-07-13}"

# --- prove it ------------------------------------------------------------------
python platforms/fingerprint.py \
  "${DATA}/documents" \
  "${DATA}/document_chunks"
