# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Distillation, part 1 — the teacher labels the data
# MAGIC
# MAGIC 204 customer reviews, no labels. Hand-labelling them is a week of somebody's
# MAGIC life. Calling a large model for every review, forever, is a bill that never
# MAGIC stops.
# MAGIC
# MAGIC So do it **once**: an open-source LLM (`llama-3.1-8b`, served in the workspace)
# MAGIC labels the corpus, and part 2 trains a small model you own on those labels.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark]==0.3.0" requests

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
    task_dir = str(example_root / "pipelines" / "reviews" / "nlp" / "label_reviews")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
# Plain HTTPS to the serving endpoint. The SDK client and mlflow.deployments both
# hang on serverless until the task times out — see example 05.
os.environ["DATABRICKS_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

print("task_dir:", task_dir)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed

# COMMAND ----------

labelled = spark.table(catalog + "." + schema + ".labelled_reviews")
n = labelled.count()

print("reviews labelled:", n)
labelled.groupBy("label").count().orderBy("count", ascending=False).show()
labelled.groupBy("split").count().show()
labelled.select("label", "review_text").show(3, truncate=70)

assert n > 0, "nothing was labelled"

# Every label must be one of the three we asked for. A model told to answer in one
# word will sometimes answer in a sentence, and "Sure! The sentiment is" is not a
# class — it is a bug that would end up in the training set.
allowed = ["positive", "negative", "neutral"]
bad = labelled.filter(~labelled.label.isin(allowed)).count()
assert bad == 0, str(bad) + " reviews got an unusable label"

# If the teacher put everything in one class there is nothing for the student to
# learn, and a classifier that always says "positive" would score well on it.
assert labelled.select("label").distinct().count() > 1, "the teacher labelled everything the same"

print("OK")
