#!/usr/bin/env bash
# The Spark environment an open-source platform needs. Sourced, never executed.
#
#     . platforms/spark_env.sh
#
# This exists because I got it wrong: run_task.sh exported PYSPARK_SUBMIT_ARGS inside
# its own subshell, so the runner that *called* it — and then ran the fingerprint —
# had no Delta JAR on the classpath and died with ClassNotFoundException. Docker
# survived only because the image sets the variable as an ENV. Anything that starts a
# Spark session must source this; nobody may re-derive it.
#
# Everything here is a fact about the PLATFORM, not about any pipeline:
#
#   * Delta and the JDBC drivers are JVM artifacts, and pip does not install JARs.
#     `pip install delta-spark` ships the Python half and none of the JAR, and the
#     engine builds its own SparkSession — so PYSPARK_SUBMIT_ARGS is the only channel
#     that reaches the classpath. Get it wrong and you get ClassNotFoundException at
#     RUNTIME, never at install time.
#
#   * Delta has to be switched ON for open-source Spark. Databricks ships it already
#     active. That is why none of this is in a config.yaml: a pipeline must not know
#     which Spark distribution it landed on.
#
#   * A real Hive metastore, so `format: unity` — which is only ever spark.table() and
#     saveAsTable() — behaves off Databricks exactly as it does on it, and tables
#     survive between separate `ubunye run` invocations.

DATA="${DATA_DIR:-/tmp/ubunye}"
mkdir -p "${DATA}"

# Spark and Delta are a matched pair: delta-spark 3.x needs Spark 3.x, 4.x needs 4.x.
SPARK_MAJOR="$(python -c 'import pyspark;print(pyspark.__version__.split(".")[0])')"
case "$SPARK_MAJOR" in
  3) DELTA_PKG="io.delta:delta-spark_2.12:${DELTA_VERSION:-3.2.0}" ;;
  4) DELTA_PKG="io.delta:delta-spark_2.13:${DELTA_VERSION:-4.0.0}" ;;
  *) echo "unsupported pyspark major: $SPARK_MAJOR" >&2; return 1 2>/dev/null || exit 1 ;;
esac

# Adding a JDBC driver to open-source Spark is one Maven coordinate. It is Databricks
# SERVERLESS that cannot do it — which is why example 09 runs here and not there.
export PYSPARK_SUBMIT_ARGS="--packages ${DELTA_PKG},org.postgresql:postgresql:42.7.4 \
--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
--conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
--conf spark.sql.catalogImplementation=hive \
--conf spark.sql.warehouse.dir=${DATA}/warehouse \
--conf javax.jdo.option.ConnectionURL=jdbc:derby:;databaseName=${DATA}/metastore_db;create=true \
pyspark-shell"
