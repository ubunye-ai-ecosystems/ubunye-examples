#!/usr/bin/env bash
# The portability spine, on plain open-source Spark. No Databricks, no cloud.
#
# Whatever hash comes out of here is the hash Docker, Kubernetes and Databricks must
# also produce. Everything platform-specific lives in run_task.sh, which every other
# platform calls too.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_DIR:-/tmp/ubunye}"
export DATA_DIR="$DATA"

# Stage the corpus where the config expects it. This is the BOOTSTRAP, and the
# bootstrap is allowed to differ per platform. The pipeline is not.
rm -rf "${DATA}/corpus" "${DATA}/documents" "${DATA}/document_chunks"
mkdir -p "${DATA}/corpus"
cp "$ROOT"/examples/11_run_anywhere/data/corpus/*.txt "${DATA}/corpus/"

"$ROOT/platforms/run_task.sh" examples/11_run_anywhere portable ingestion document_index

python "$ROOT/platforms/fingerprint.py" "${DATA}/documents" "${DATA}/document_chunks"
