#!/usr/bin/env bash
# Create the minimum GCP needed to run an Ubunye task on Dataproc Serverless.
#
# YOU run this, not me. It prints what to put into GitHub secrets; CI holds the key,
# CI runs the job, and I read the run log. I never see it.
#
#   bash platforms/gcp/setup.sh
#
# Needs: gcloud, logged in as someone who can enable APIs and create service accounts.
#
# COST: Dataproc Serverless bills per DCU-second while a batch runs. One run of these
# examples is a couple of minutes on the smallest shape — cents. No free tier. Storage
# is a few MB.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-europe-west1}"
BUCKET="${BUCKET:-ubunye-examples-${PROJECT}}"
SA="${SA:-ubunye-examples}"

[ -n "$PROJECT" ] || { echo "set PROJECT or run: gcloud config set project <id>" >&2; exit 1; }

echo "project: $PROJECT"
echo "region : $REGION"
echo "bucket : gs://$BUCKET"

# --- 1. the APIs ----------------------------------------------------------------
gcloud services enable dataproc.googleapis.com storage.googleapis.com --project "$PROJECT"

# --- 2. a bucket for the data, the job staging and the logs ----------------------
gsutil mb -p "$PROJECT" -l "$REGION" "gs://${BUCKET}" 2>/dev/null || echo "  bucket exists"

# --- 3. the service account the BATCH runs as ------------------------------------
gcloud iam service-accounts create "$SA" --project "$PROJECT" \
  --display-name "Ubunye examples (Dataproc Serverless)" 2>/dev/null || echo "  service account exists"

SA_EMAIL="${SA}@${PROJECT}.iam.gserviceaccount.com"

# Dataproc Serverless needs the worker role, and access to exactly one bucket.
# Scoped on purpose: roles/storage.admin at project level is how a demo account ends
# up able to delete a production bucket.
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:${SA_EMAIL}" --role roles/dataproc.worker --quiet >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SA_EMAIL}" --role roles/storage.objectAdmin --quiet >/dev/null

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Put these into the repo's GitHub secrets.

  gh secret set GCP_PROJECT --body "${PROJECT}"
  gh secret set GCP_REGION  --body "${REGION}"
  gh secret set GCP_BUCKET  --body "${BUCKET}"
  gh secret set GCP_SERVICE_ACCOUNT --body "${SA_EMAIL}"

And the credential CI authenticates with. Prefer Workload Identity Federation —
no key file exists at all, so none can leak:

  gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body "projects/<n>/locations/global/workloadIdentityPools/<pool>/providers/<provider>"

Or, if you must, a JSON key (rotate it afterwards):

  gcloud iam service-accounts keys create key.json --iam-account "${SA_EMAIL}"
  gh secret set GCP_SA_KEY < key.json
  rm key.json      # <- do not leave this on disk, and never commit it
────────────────────────────────────────────────────────────────────────────
EOF
