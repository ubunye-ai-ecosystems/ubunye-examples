#!/usr/bin/env bash
# Seed, build the document index, then answer questions with open-source models.
#
# This is one script because it is one story, and because 05 CANNOT run alone: it reads
# the chunks that 03 produces. Handing a Kubernetes pod "run example 05" gave it an
# empty volume, no metastore and nothing to read — and it sat there failing until the
# job timed out. The bootstrap has to come with it.
#
# Docker and Kubernetes both run exactly this. Nothing in it is platform-specific.
set -euo pipefail

ROOT="${APP_ROOT:-/app}"
DATA="${DATA_DIR:-/data}"

# This script IS the off-Databricks bootstrap, so it carries its own defaults rather
# than trusting every caller to remember six variables. The first version did not, I
# dropped the -e flags when I switched the docker run to call it, and example 03 fell
# straight back to its Databricks default and went looking for
#   /Volumes/workspace/ubunye_examples/corpus
# inside a container. Anything the caller sets still wins.
export SOURCE_CATALOG="${SOURCE_CATALOG:-spark_catalog}"
export UBUNYE_CATALOG="${UBUNYE_CATALOG:-spark_catalog}"
export UBUNYE_SCHEMA="${UBUNYE_SCHEMA:-ubunye_examples}"
export UBUNYE_SINK="${UBUNYE_SINK:-unity}"
export UBUNYE_CORPUS="${UBUNYE_CORPUS:-file://${DATA}/corpus}"
export MODEL_BACKEND="${MODEL_BACKEND:-local}"   # the whole point: no endpoint
export TORCH_THREADS="${TORCH_THREADS:-2}"
export STUDENT_MODEL_STORE="${STUDENT_MODEL_STORE:-${DATA}/model_store}"
export DATA_DIR="${DATA}"

[ -f /etc/java.env ] && . /etc/java.env

mkdir -p "${DATA}/corpus"
cp "${ROOT}"/examples/11_run_anywhere/data/corpus/*.txt "${DATA}/corpus/"

cd "$ROOT"
# shellcheck source=/dev/null
. platforms/spark_env.sh

echo "=== seed (same schema as samples.*, different rows) ==="
python platforms/seed.py

echo "=== 03 · documents and chunks ==="
platforms/run_task.sh examples/03_ingest_unstructured docs ingestion document_index

echo "=== 05 · RAG, on open-source models, no endpoint ==="
platforms/run_task.sh examples/05_rag_documents rag knowledge embed_chunks answer_questions

if [ "${WITH_DISTILLATION:-false}" = "true" ]; then
  echo "=== 06 · distillation, an open teacher and an open student ==="
  platforms/run_task.sh examples/06_finetune_llm reviews nlp label_reviews finetune_classifier
fi

echo
echo "RAG ran with no Databricks, no serving endpoint, and no API key."
