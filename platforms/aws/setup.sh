#!/usr/bin/env bash
# Create the minimum AWS needed to run an Ubunye task on EMR Serverless.
#
# YOU run this, not me. It prints the four values to put into GitHub secrets, and I
# never see your keys — CI holds them, CI runs the job, and I read the run log.
#
#   bash platforms/aws/setup.sh
#
# Needs: aws CLI, logged in as someone who can create S3 buckets, IAM roles and EMR
# Serverless applications.
#
# COST: EMR Serverless bills per vCPU-second while a job runs, with no charge when
# idle. One run of these examples is a couple of minutes on 2-4 vCPUs — cents, not
# dollars. There is no free tier for it. The S3 storage is a few MB.
set -euo pipefail

REGION="${AWS_REGION:-eu-west-1}"
BUCKET="${BUCKET:-ubunye-examples-$(aws sts get-caller-identity --query Account --output text)}"
ROLE="${ROLE:-UbunyeEMRServerlessJobRole}"
APP="${APP:-ubunye-examples}"

echo "region : $REGION"
echo "bucket : $BUCKET"

# --- 1. a bucket for the data and the job logs ---------------------------------
aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
  --create-bucket-configuration LocationConstraint="$REGION" 2>/dev/null \
  || echo "  bucket exists"

# --- 2. the role the JOB runs as ------------------------------------------------
# Not the role that submits the job — the role the Spark job itself assumes. It needs
# to read and write exactly one bucket and write its logs. Nothing else.
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"emr-serverless.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" \
  >/dev/null 2>&1 || echo "  role exists"

# Scoped to the one bucket. A wildcard here is how a demo role becomes the thing that
# read your production data three years later.
POLICY=$(cat <<JSON
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],
   "Resource":["arn:aws:s3:::${BUCKET}","arn:aws:s3:::${BUCKET}/*"]},
  {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents",
                              "logs:DescribeLogStreams","logs:DescribeLogGroups"],
   "Resource":"arn:aws:logs:*:*:*"}
]}
JSON
)
aws iam put-role-policy --role-name "$ROLE" \
  --policy-name UbunyeExamplesAccess --policy-document "$POLICY"

ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

# --- 3. the EMR Serverless application ------------------------------------------
# EMR 7.x ships Spark 3.5, which is the pair the image pins. A mismatched Spark here
# is the same ClassNotFoundException trap as everywhere else.
APP_ID="$(aws emr-serverless list-applications --region "$REGION" \
  --query "applications[?name=='${APP}'].id | [0]" --output text 2>/dev/null || echo None)"

if [ "$APP_ID" = "None" ] || [ -z "$APP_ID" ]; then
  APP_ID="$(aws emr-serverless create-application \
    --region "$REGION" --name "$APP" --type SPARK --release-label emr-7.2.0 \
    --query applicationId --output text)"
  echo "  created application $APP_ID"
else
  echo "  application exists: $APP_ID"
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Put these into the repo's GitHub secrets. I never need to see the keys.

  gh secret set AWS_REGION            --body "${REGION}"
  gh secret set AWS_S3_BUCKET         --body "${BUCKET}"
  gh secret set AWS_EMR_APPLICATION_ID --body "${APP_ID}"
  gh secret set AWS_EMR_JOB_ROLE_ARN  --body "${ROLE_ARN}"

And the credentials CI submits jobs with. Prefer OIDC over a long-lived key:

  # Best: no static key at all — GitHub assumes a role directly.
  gh secret set AWS_ROLE_TO_ASSUME --body "arn:aws:iam::<account>:role/<gh-oidc-role>"

  # Or, if you must, a key for a user allowed ONLY to submit EMR jobs:
  gh secret set AWS_ACCESS_KEY_ID     --body "..."
  gh secret set AWS_SECRET_ACCESS_KEY --body "..."
────────────────────────────────────────────────────────────────────────────
EOF
