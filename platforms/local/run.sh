#!/usr/bin/env bash
# The portability spine, on plain open-source Spark. No Databricks, no cloud.
#
# Whatever hash comes out of here is the hash Docker, Kubernetes and Databricks must
# also produce.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_DIR="${DATA_DIR:-/tmp/ubunye}"

# Stage the corpus where the config expects it. This is the BOOTSTRAP, and the
# bootstrap is allowed to differ per platform. The pipeline is not.
rm -rf "${DATA_DIR}/corpus" "${DATA_DIR}/documents" "${DATA_DIR}/document_chunks"
mkdir -p "${DATA_DIR}/corpus"
cp "$ROOT"/examples/11_run_anywhere/data/corpus/*.txt "${DATA_DIR}/corpus/"

"$ROOT/platforms/run_task.sh" examples/11_run_anywhere portable ingestion document_index

# The fingerprint needs Spark too — and it needs the SAME Spark. Sourcing rather than
# re-deriving is the whole point of spark_env.sh: the first version of this script let
# run_task.sh export PYSPARK_SUBMIT_ARGS inside its own subshell, and this line then
# died with ClassNotFoundException reading tables the pipeline had just written.
# shellcheck source=/dev/null
. "$ROOT/platforms/spark_env.sh"
python "$ROOT/platforms/fingerprint.py" "${DATA_DIR}/documents" "${DATA_DIR}/document_chunks"
