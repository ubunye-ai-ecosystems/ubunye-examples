#!/usr/bin/env bash
# Same three steps as every other platform: stage the corpus, run the task, print the
# fingerprint. The task directory is byte-identical to the one on Databricks.
set -euo pipefail

. /etc/java.env
export JAVA_HOME

DATA="${DATA_DIR:-/data}"
rm -rf "${DATA}/documents" "${DATA}/document_chunks"
mkdir -p "${DATA}/corpus"
cp /app/examples/11_run_anywhere/data/corpus/*.txt "${DATA}/corpus/"

echo "master    : ${SPARK_MASTER}"
echo "sink      : ${UBUNYE_SINK}"
echo "data root : ${UBUNYE_DATA_ROOT}"

cd /app
ubunye run \
  -d examples/11_run_anywhere/pipelines \
  -u portable \
  -p ingestion \
  -t document_index \
  -m PROD \
  -dt "${DT:-2026-07-13}"

python platforms/fingerprint.py "${DATA}/documents" "${DATA}/document_chunks"
