# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · JDBC — ingest from a real relational database
# MAGIC
# MAGIC > ### ⚠️ This is the one example that does **not** run on Databricks Free Edition
# MAGIC >
# MAGIC > It needs a **classic cluster**, and serverless cannot do this:
# MAGIC >
# MAGIC > - serverless ships **no JDBC drivers** (not for Postgres, MySQL, or anything else)
# MAGIC > - and it **cannot install a JAR**, so you cannot add one
# MAGIC >
# MAGIC > A free workspace has no classic compute at all — the API answers
# MAGIC > `does not have any associated worker environments`. So this example is deployed
# MAGIC > by a separate bundle with its own cluster, and needs a **paid workspace**.
# MAGIC >
# MAGIC > **What has and has not been verified:** the config, the `jdbc` reader, the
# MAGIC > pushed-down SQL, the partitioned read and the transformation were all run
# MAGIC > end-to-end against the live database on local Spark — 56 databases, 200,000
# MAGIC > sequences, 4 partitions. What has **not** been run is this notebook on a
# MAGIC > Databricks classic cluster, because no such cluster was available. Every other
# MAGIC > example in this repo has been run on real Databricks. This one has not.
# MAGIC
# MAGIC ## The lesson
# MAGIC
# MAGIC **A JDBC read is single-threaded unless you tell it not to be.** Spark issues one
# MAGIC query on one connection, and a single core drags the whole table through it —
# MAGIC however big your cluster is. There is no warning. The job is just slow forever,
# MAGIC and the Spark UI shows one task at 100% while everything else idles.
# MAGIC
# MAGIC `partitionColumn` + `lowerBound` + `upperBound` + `numPartitions` turns that one
# MAGIC query into N range queries that run in parallel.
# MAGIC
# MAGIC The data is [RNAcentral](https://rnacentral.org)'s public PostgreSQL mirror, run
# MAGIC by the EMBL-EBI: 54 million RNA sequences, published read-only credentials, free
# MAGIC for anyone. We only pull a bounded slice — it is somebody else's research
# MAGIC database and we are guests on it.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark]==0.3.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from pathlib import Path

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
task_dir = dbutils.widgets.get("task_dir")

if not task_dir:
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    example_root = Path("/Workspace") / Path(nb).parent.parent.relative_to("/")
    task_dir = str(example_root / "pipelines" / "rnacentral" / "ingestion" / "ingest_rna")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

print("task_dir:", task_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Is the driver actually here?
# MAGIC
# MAGIC The JAR is attached by the bundle (`libraries: - maven: org.postgresql:postgresql`).
# MAGIC If it is missing, the failure is a `ClassNotFoundException` thrown from deep inside
# MAGIC a Spark job — which reads like a Spark problem and is not one. Ask up front.

# COMMAND ----------

try:
    spark._jvm.java.lang.Class.forName("org.postgresql.Driver")
    print("org.postgresql.Driver is on the classpath")
except Exception:
    raise RuntimeError(
        "The Postgres JDBC driver is not on the classpath. This notebook needs a "
        "CLASSIC cluster with the driver attached as a library — serverless ships no "
        "JDBC drivers and cannot install a JAR. Deploy with the bundle in "
        "examples/09_jdbc_ingest/, which declares it."
    )

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What came back

# COMMAND ----------

FQ = catalog + "." + schema

databases = spark.table(FQ + ".rna_databases")
databases.orderBy("num_sequences", ascending=False).select(
    "id", "code", "display_name", "is_active", "num_sequences", "avg_length"
).show(10, truncate=False)

profile = spark.table(FQ + ".rna_sequence_profile")
profile.orderBy("min_length").show(truncate=False)

# COMMAND ----------

n_databases = databases.count()
row = profile.agg({"sequences": "sum"}).first()[0]
partitions = profile.select("read_partitions").first()["read_partitions"]

print("databases      :", n_databases)
print("sequences      :", row)
print("read partitions:", partitions)

assert n_databases > 0, "the dimension table came back empty"
assert row == 200_000, "expected the bounded 200k slice, got " + str(row)

# The whole point of the example. `numPartitions` is a HINT: Spark silently ignores
# it when partitionColumn is missing or the bounds are nonsense, and hands you back
# the single-threaded read you were trying to avoid — with no warning and no error.
#
# On the full 54M-row table that is the difference between minutes and hours, and it
# is invisible unless somebody looks. So look.
assert partitions == 4, (
    "the JDBC read came back in " + str(partitions) + " partition(s), not 4 — the "
    "partitioned read did not happen and this was a single-threaded pull"
)

print()
print("OK")
