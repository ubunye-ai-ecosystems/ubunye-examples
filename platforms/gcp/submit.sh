#!/usr/bin/env bash
# Submit the task to GCP Dataproc Serverless.
#
# Same task directory as everywhere else. Dataproc, like EMR, runs a Python entry point
# through spark-submit — not `ubunye run` — which is why platforms/spark_entrypoint.py
# exists and why CI proves it under a real spark-submit before a cloud ever sees it.
#
# Requires: GCP_PROJECT, GCP_REGION, GCP_BUCKET, GCP_SERVICE_ACCOUNT.
# Run platforms/gcp/setup.sh once to create them.
set -euo pipefail

: "${GCP_PROJECT:?}" "${GCP_BUCKET:?}"
REGION="${GCP_REGION:-europe-west1}"
EXAMPLE="${1:-examples/11_run_anywhere}"
TASK="${2:-pipelines/portable/ingestion/document_index}"
GS="gs://${GCP_BUCKET}"

echo "staging code and corpus to ${GS}"
gsutil -q cp platforms/spark_entrypoint.py "${GS}/code/"
gsutil -q cp platforms/fingerprint.py      "${GS}/code/"
gsutil -q -m cp -r "${EXAMPLE}" "${GS}/code/examples/"
gsutil -q -m cp -r "${EXAMPLE}/data/corpus" "${GS}/data/" 2>/dev/null || true

# Dataproc Serverless sets the master itself. We do not pass spark.master.
gcloud dataproc batches submit pyspark "${GS}/code/spark_entrypoint.py" \
  --project "$GCP_PROJECT" --region "$REGION" \
  --deps-bucket "$GCP_BUCKET" \
  ${GCP_SERVICE_ACCOUNT:+--service-account "$GCP_SERVICE_ACCOUNT"} \
  --py-files "${GS}/code/fingerprint.py" \
  --version 2.2 \
  --properties="spark.jars.packages=io.delta:delta-spark_2.12:3.2.0,\
spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension,\
spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog,\
spark.executorEnv.UBUNYE_SINK=s3,\
spark.executorEnv.UBUNYE_DATA_ROOT=${GS}/data,\
spark.yarn.appMasterEnv.UBUNYE_SINK=s3,\
spark.yarn.appMasterEnv.UBUNYE_DATA_ROOT=${GS}/data" \
  -- \
  --task-dir "${GS}/code/${EXAMPLE}/${TASK}" \
  --mode PROD --dt 2026-07-13 \
  --fingerprint "${GS}/data/documents" "${GS}/data/document_chunks"
