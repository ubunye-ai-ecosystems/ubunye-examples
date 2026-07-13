#!/usr/bin/env bash
# Stage whatever the task needs, run it, and (for the deterministic one) fingerprint it.
#
# Staging is the BOOTSTRAP, and the bootstrap is allowed to differ per platform. The
# pipeline is not. That split is the honest one, and confusing the two is why people
# conclude portability is impossible: they try to make the *launcher* portable, fail,
# and blame the pipeline.
set -euo pipefail

. /etc/java.env

DATA="${DATA_DIR:-/data}"
mkdir -p "${DATA}"

EXAMPLE="${1:-examples/11_run_anywhere}"

# Example 11 reads a corpus of files; put them where the config expects them.
if [ -d "/app/${EXAMPLE}/data/corpus" ]; then
  rm -rf "${DATA}/corpus"
  mkdir -p "${DATA}/corpus"
  cp /app/"${EXAMPLE}"/data/corpus/*.txt "${DATA}/corpus/"
fi

cd /app
platforms/run_task.sh "$@"

# The run-anywhere example is the one whose output is deterministic, so it is the one
# the portability matrix compares. The others are checked by their own assertions.
if [ "${EXAMPLE}" = "examples/11_run_anywhere" ]; then
  python platforms/fingerprint.py "${DATA}/documents" "${DATA}/document_chunks"
fi
