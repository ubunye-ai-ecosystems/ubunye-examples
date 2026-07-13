#!/usr/bin/env bash
# Submit the task to AWS EMR Serverless.
#
# Note what is NOT here: any change to the pipeline. The task directory is the same one
# local Spark, Docker, Kubernetes and Databricks run. What EMR needs is a Python entry
# point (platforms/spark_entrypoint.py) and its dependencies on S3 — that is the
# platform's business, and it lives here.
#
# Requires: AWS_S3_BUCKET, AWS_EMR_APPLICATION_ID, AWS_EMR_JOB_ROLE_ARN, AWS_REGION.
# Run platforms/aws/setup.sh once to create them.
set -euo pipefail

# --- running the MODEL examples (05 RAG, 06 distillation) on EMR ----------------
#
# They need torch, transformers and the weights. EMR Serverless takes a CUSTOM
# CONTAINER IMAGE, and platforms/docker/Dockerfile.ml is exactly that image -- the same
# one Docker and Kubernetes run. Push it to ECR and point the application at it:
#
#   aws ecr create-repository --repository-name ubunye-ml --region "$REGION"
#   docker build -f platforms/docker/Dockerfile   -t ubunye-portable:ci .
#   docker build -f platforms/docker/Dockerfile.ml --build-arg BASE=ubunye-portable:ci #          -t "$ECR/ubunye-ml:latest" .
#   aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR"
#   docker push "$ECR/ubunye-ml:latest"
#
#   aws emr-serverless update-application #     --application-id "$AWS_EMR_APPLICATION_ID" --region "$REGION" #     --image-configuration imageUri="$ECR/ubunye-ml:latest"
#
# then submit with MODEL_BACKEND=local:
#   --conf spark.emr-serverless.driverEnv.MODEL_BACKEND=local
#
# The weights are already inside the image, so the job does not touch the internet --
# which matters, because an EMR job in a private subnet cannot reach huggingface.co.


: "${AWS_S3_BUCKET:?}" "${AWS_EMR_APPLICATION_ID:?}" "${AWS_EMR_JOB_ROLE_ARN:?}"
REGION="${AWS_REGION:-eu-west-1}"
EXAMPLE="${1:-examples/11_run_anywhere}"
TASK="${2:-pipelines/portable/ingestion/document_index}"
S3="s3://${AWS_S3_BUCKET}"

echo "staging code and corpus to ${S3}"
aws s3 cp platforms/spark_entrypoint.py "${S3}/code/" --region "$REGION"
aws s3 cp platforms/fingerprint.py      "${S3}/code/" --region "$REGION"
aws s3 cp "${EXAMPLE}/" "${S3}/code/${EXAMPLE}/" --recursive --region "$REGION"
aws s3 cp "${EXAMPLE}/data/corpus/" "${S3}/data/corpus/" --recursive --region "$REGION" 2>/dev/null || true

# EMR Serverless sets the master itself. We do NOT pass spark.master — forcing it would
# run the whole job in the driver and ignore every executor, successfully and silently.
SUBMIT_PARAMS="--conf spark.jars.packages=io.delta:delta-spark_2.12:3.2.0 \
--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
--conf spark.submit.pyFiles=${S3}/code/fingerprint.py \
--conf spark.emr-serverless.driverEnv.UBUNYE_SINK=s3 \
--conf spark.emr-serverless.driverEnv.UBUNYE_DATA_ROOT=s3a://${AWS_S3_BUCKET}/data \
--conf spark.executorEnv.UBUNYE_SINK=s3 \
--conf spark.executorEnv.UBUNYE_DATA_ROOT=s3a://${AWS_S3_BUCKET}/data"

RUN_ID="$(aws emr-serverless start-job-run \
  --region "$REGION" \
  --application-id "$AWS_EMR_APPLICATION_ID" \
  --execution-role-arn "$AWS_EMR_JOB_ROLE_ARN" \
  --name ubunye-run-anywhere \
  --job-driver "{\"sparkSubmit\":{
      \"entryPoint\":\"${S3}/code/platforms/spark_entrypoint.py\",
      \"entryPointArguments\":[\"--task-dir\",\"${S3}/code/${EXAMPLE}/${TASK}\",
                              \"--mode\",\"PROD\",\"--dt\",\"2026-07-13\",
                              \"--fingerprint\",\"s3a://${AWS_S3_BUCKET}/data/documents\",
                                               \"s3a://${AWS_S3_BUCKET}/data/document_chunks\"],
      \"sparkSubmitParameters\":\"${SUBMIT_PARAMS}\"}}" \
  --configuration-overrides "{\"monitoringConfiguration\":{\"s3MonitoringConfiguration\":{\"logUri\":\"${S3}/logs/\"}}}" \
  --query jobRunId --output text)"

echo "job run: $RUN_ID"
aws emr-serverless get-job-run --region "$REGION" \
  --application-id "$AWS_EMR_APPLICATION_ID" --job-run-id "$RUN_ID" \
  --query 'jobRun.state' --output text

echo "logs: ${S3}/logs/applications/${AWS_EMR_APPLICATION_ID}/jobs/${RUN_ID}/"
