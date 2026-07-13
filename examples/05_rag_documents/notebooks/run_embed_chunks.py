# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · RAG — embed the chunks
# MAGIC
# MAGIC Turns the `document_chunks` table from example 03 into vectors. `merge` on chunk_id, so re-running does not re-embed the world — embedding is the expensive step.
# MAGIC
# MAGIC Both models are **open source** and already served in the workspace:
# MAGIC `databricks-bge-large-en` (BAAI's BGE) for embeddings and
# MAGIC `databricks-gpt-oss-20b` for generation. Nothing is downloaded, no GPU is
# MAGIC needed, and no outbound internet is required — which matters, because
# MAGIC serverless blocks most of it.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark,ml]==0.3.0"

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
    task_dir = str(example_root / "pipelines" / "rag" / "knowledge" / "embed_chunks")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

# The tasks call the serving endpoints over plain HTTPS, because that is what
# serverless actually permits. The SDK client and mlflow.deployments both hung
# here until the task timed out, on a request the endpoint answers in 3 seconds.
os.environ["DATABRICKS_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

print("task_dir:", task_dir)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

emb = spark.table(catalog + "." + schema + ".chunk_embeddings")
n = emb.count()
dims = emb.select("embedding_dim").distinct().collect()

print("chunks embedded:", n)
print("dimensions     :", [d["embedding_dim"] for d in dims])
emb.select("title", "embedding_dim", "embedding_model").show(5, truncate=40)

assert n > 0, "nothing was embedded"
# One dimensionality, or the vectors are not in the same space and cosine
# similarity between them is a number with no meaning.
assert len(dims) == 1, "mixed embedding dimensions: " + str(dims)

# A vector of zeros is what you get when the endpoint quietly failed and something
# helpfully filled in a default. It would rank as maximally similar to nothing.
import pyspark.sql.functions as F

zero_vectors = emb.filter(F.aggregate("embedding", F.lit(0.0), lambda acc, x: acc + F.abs(x)) == 0).count()
assert zero_vectors == 0, str(zero_vectors) + " chunks embedded to all-zero vectors"

print("OK")
