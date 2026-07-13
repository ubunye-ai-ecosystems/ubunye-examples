# The engine, Spark, Delta and the examples in one image.
#
# Nothing in the pipelines is Docker-aware. The image supplies a JVM, a Spark, a Delta
# and the engine from PyPI; the task directories are copied in unchanged. A container
# is just another place to stand.
#
# Pinned to bookworm on purpose. `python:3.11-slim` now tracks Debian trixie, which
# dropped openjdk-17 for openjdk-21 — the unpinned base fails the build with a bare
# `exit code: 100` from apt, which tells you nothing. Spark 3.5 wants Java 17.
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

# Resolve JAVA_HOME rather than hard-code it — the path differs between arm64 and
# amd64, and a wrong one fails late and obscurely.
RUN JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" \
 && echo "export JAVA_HOME=${JAVA_HOME}" > /etc/java.env

# Spark and Delta are a MATCHED PAIR. delta-spark 3.x needs Spark 3.x; 4.x needs 4.x.
# Mixing them installs cleanly and then dies at runtime with
#   ClassNotFoundException: delta.DefaultSource
# which looks like a config error and is not. Pin both, together, on purpose.
ARG PYSPARK_VERSION=3.5.3
ARG DELTA_VERSION=3.2.0
ARG UBUNYE_VERSION=0.3.0

# `requests` is NOT a runtime dependency of the engine — it sits in the `dev` extra.
# So `pip install ubunye-engine[spark]` gives you a rest_api reader that cannot make a
# request. Nobody notices, because Databricks preinstalls requests; the moment you
# leave Databricks, example 02 dies on import. Install it explicitly.
RUN pip install --no-cache-dir \
      "pyspark==${PYSPARK_VERSION}" \
      "delta-spark==${DELTA_VERSION}" \
      "ubunye-engine[spark,ml]==${UBUNYE_VERSION}" \
      "requests>=2.28"

# Delta and the JDBC driver are JVM artifacts, and pip does not install JARs. The
# engine builds its own SparkSession, so PYSPARK_SUBMIT_ARGS is the only channel that
# reaches the classpath.
#
# Note what the postgresql coordinate means: adding a JDBC driver to open-source Spark
# is ONE line. It is Databricks *serverless* that cannot do it — which is why example
# 09 runs in this container and not there.
ENV DELTA_VERSION=${DELTA_VERSION}
ENV PYSPARK_SUBMIT_ARGS="--packages io.delta:delta-spark_2.12:${DELTA_VERSION},org.postgresql:postgresql:42.7.4 pyspark-shell"

# Warm the ivy cache at BUILD time. Without this, every container start pulls the JARs
# from Maven — slow, and it silently makes the image need the internet at RUN time,
# which an air-gapped cluster does not have.
RUN . /etc/java.env \
 && python -c "from pyspark.sql import SparkSession; SparkSession.builder.appName('warm').getOrCreate().stop()"

WORKDIR /app
COPY examples /app/examples
COPY platforms /app/platforms
RUN chmod +x /app/platforms/*.sh /app/platforms/docker/entrypoint.sh

ENV SPARK_MASTER="local[*]" \
    UBUNYE_SINK="s3" \
    UBUNYE_DATA_ROOT="file:///data" \
    DATA_DIR="/data"

ENTRYPOINT ["/app/platforms/docker/entrypoint.sh"]
# Default: the run-anywhere example. Override with any other task:
#   docker run ubunye-portable examples/02_ingest_rest_api weather ingestion hourly_forecast
CMD ["examples/11_run_anywhere", "portable", "ingestion", "document_index"]
