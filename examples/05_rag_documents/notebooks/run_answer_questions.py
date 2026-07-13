# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · RAG — retrieve and generate
# MAGIC
# MAGIC Embeds each question into the SAME vector space as the chunks, retrieves the nearest by cosine similarity, and asks an open-source LLM to answer **from that context only** — storing every answer with the sources that produced it.
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
    task_dir = str(example_root / "pipelines" / "rag" / "knowledge" / "answer_questions")

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

answers = spark.table(catalog + "." + schema + ".rag_answers")
n = answers.count()
print("questions answered:", n)

for row in answers.collect():
    print()
    print("Q:", row["question"])
    print("A:", row["answer"][:300])
    print("   sources:", ", ".join(row["source_titles"][:3]), "| top score:", round(row["top_similarity"], 3))

assert n > 0, "no answers produced"
assert answers.filter("size(source_chunk_ids) = 0").count() == 0, "an answer cited no sources"

# The honesty check. One question in the set cannot be answered from the corpus.
# A RAG system that will not say "I do not know" is one that confidently makes
# things up — and that is worse than no RAG at all.
import pyspark.sql.functions as F

refusal = answers.filter(F.lower(F.col("question")).contains("swallow")).first()
print()
print("unanswerable question ->", refusal["answer"][:160])
assert "does not answer" in refusal["answer"].lower(), (
    "the model answered a question the corpus cannot support — it hallucinated"
)

print()
print("OK")
