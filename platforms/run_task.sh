#!/usr/bin/env bash
# Run ANY example's task on open-source Spark — local, in Docker, or in a pod.
#
#   platforms/run_task.sh <example-dir> <usecase> <package> <task> [more tasks...]
#
# e.g. platforms/run_task.sh examples/02_ingest_rest_api weather ingestion hourly_forecast
#
# One script, because the platforms differ in where they stand, not in what they do.
set -euo pipefail

EXAMPLE="${1:?example dir, e.g. examples/02_ingest_rest_api}"
USECASE="${2:?usecase}"
PACKAGE="${3:?package}"
shift 3
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || { echo "at least one task required" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The Spark environment is defined in ONE place and sourced. See spark_env.sh for why.
# shellcheck source=/dev/null
. "$ROOT/platforms/spark_env.sh"

# --- the three variables that ARE the portability surface ----------------------
export SPARK_MASTER="${SPARK_MASTER:-local[*]}"
export UBUNYE_SINK="${UBUNYE_SINK:-s3}"
export UBUNYE_DATA_ROOT="${UBUNYE_DATA_ROOT:-file://${DATA}}"

echo "example   : ${EXAMPLE}"
echo "tasks     : ${TASKS[*]}"
echo "master    : ${SPARK_MASTER}"
echo "sink      : ${UBUNYE_SINK}"
echo "data root : ${UBUNYE_DATA_ROOT}"

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
