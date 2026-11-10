# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · ML — feature engineering
# MAGIC
# MAGIC Clean, derive, and split. Produces a **table**, not a model.
# MAGIC
# MAGIC The split is decided here and written down, so the training task and the
# MAGIC scoring task cannot disagree about which rows the model has seen. A split
# MAGIC recomputed at training time is a split you cannot trust.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")
dbutils.widgets.text("ubunye_run_records", "", "Run records table (blank = <catalog>.<schema>.uby_run_records)")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[ml]==0.9.2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from pathlib import Path

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
# Where the run record goes. A job passes ubunye_run_records (see databricks.yml);
# opened by hand, it is a table next to the outputs.
run_records = dbutils.widgets.get("ubunye_run_records") or catalog + "." + schema + ".uby_run_records"
task_dir = dbutils.widgets.get("task_dir")

if not task_dir:
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    example_root = Path("/Workspace") / Path(nb).parent.parent.relative_to("/")
    task_dir = str(example_root / "pipelines" / "taxi_fare" / "ml" / "feature_engineering")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

print("task_dir:", task_dir)

# COMMAND ----------

import ubunye

# One recorded run, three steps. Its run record (who ran it, from which commit,
# what went in and out, and the stage that failed if one did) goes to run_records.
with ubunye.notebook(task_dir, dt=dt, mode=mode, run_records=run_records) as run:
    sources = run.read()              # READ: every input in config.yaml, one DataFrame each, nothing changed yet
    outputs = run.transform(sources)  # TRANSFORM: transformations.py, one DataFrame per output; all business logic is there
    run.write(outputs)                # WRITE: data-quality rules first (nothing is written if one fails), then each output
outputs["taxi_features"].show(5, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed

# COMMAND ----------

features = spark.table(catalog + "." + schema + ".taxi_features")
features.groupBy("split").count().orderBy("split").show()

n = features.count()
splits = {r["split"] for r in features.select("split").distinct().collect()}
print("feature rows:", n)

assert n > 0, "no features produced"
# All three splits must exist, or a downstream task will read an empty table and
# fail in a way that looks like a modelling problem rather than a data one.
assert splits == {"train", "test", "score"}, "missing a split: " + str(splits)

print("OK")
