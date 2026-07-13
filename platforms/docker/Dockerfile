# The engine, Spark and Delta in one image — the same task, in a container.
#
# Nothing about the pipeline is Docker-aware. The image supplies a JVM, a Spark, a
# Delta, and the engine from PyPI; the task directory is copied in unchanged. That is
# the point: a container is just another place to stand.
# Pinned to bookworm on purpose. `python:3.11-slim` now tracks Debian trixie, which
# dropped openjdk-17 for openjdk-21 — so the unpinned base breaks the build with a
# bare `exit code: 100` from apt, which tells you nothing. Spark 3.5 wants Java 17.
FROM python:3.11-slim-bookworm

# Spark needs a JVM. Nothing else here does.
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

# Resolve JAVA_HOME rather than hard-code it — the path differs between arm64 and
# amd64, and a wrong one fails late and obscurely.
RUN JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" \
 && echo "JAVA_HOME=${JAVA_HOME}" > /etc/java.env

# Spark and Delta are a MATCHED PAIR. delta-spark 3.x needs Spark 3.x; 4.x needs 4.x.
# Mixing them installs cleanly and then dies at runtime with
#   ClassNotFoundException: delta.DefaultSource
# which looks like a config error and is not. Pin both, together, on purpose.
ARG PYSPARK_VERSION=3.5.3
ARG DELTA_VERSION=3.2.0
ARG UBUNYE_VERSION=0.3.0

RUN pip install --no-cache-dir \
      "pyspark==${PYSPARK_VERSION}" \
      "delta-spark==${DELTA_VERSION}" \
      "ubunye-engine==${UBUNYE_VERSION}"

# Delta on OSS Spark is a JAR, and pip does not install JARs. The engine builds its
# own SparkSession, so the only way to get the JAR onto that session's classpath is
# PYSPARK_SUBMIT_ARGS — the pip package alone gives you the Python API and a
# ClassNotFoundException.
ENV PYSPARK_SUBMIT_ARGS="--packages io.delta:delta-spark_2.12:${DELTA_VERSION} pyspark-shell"

# Pre-fetch the JAR into the image's ivy cache at BUILD time. Without this every
# container start pulls it from Maven — slow, and it silently makes the image need
# the internet at RUN time, which an air-gapped cluster does not have.
RUN python -c "from pyspark.sql import SparkSession; SparkSession.builder.appName('warm').getOrCreate().stop()"

WORKDIR /app
COPY examples/11_run_anywhere /app/examples/11_run_anywhere
COPY platforms/fingerprint.py /app/platforms/fingerprint.py
COPY platforms/docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV SPARK_MASTER="local[*]" \
    UBUNYE_SINK="s3" \
    UBUNYE_DATA_ROOT="file:///data" \
    DATA_DIR="/data"

ENTRYPOINT ["/app/entrypoint.sh"]
