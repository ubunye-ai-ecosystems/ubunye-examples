# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Structured ingestion — Unity Catalog tables + SQL
# MAGIC
# MAGIC Reads `samples.bakehouse` (ships with every workspace), aggregates revenue by
# MAGIC franchise and by product, and writes two Delta tables: one with `merge`
# MAGIC (safe to re-run), one with `overwrite_partitions` (safe to backfill).
# MAGIC
# MAGIC Serverless. Press **Run all** — there is nothing to configure.

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

# Re-read the widgets: restartPython() wiped every variable set above.
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
task_dir = dbutils.widgets.get("task_dir")

# A job supplies task_dir. A human who opened this from a Git folder has no job,
# so derive it from where this notebook sits — that is what lets one file be both
# deployable and openable.
if not task_dir:
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    example_root = Path("/Workspace") / Path(nb).parent.parent.relative_to("/")
    task_dir = str(example_root / "pipelines" / "retail" / "ingestion" / "customer_revenue")

print("task_dir:", task_dir)
print("target  :", catalog + "." + schema)

# COMMAND ----------

# The engine writes tables. It does not provision catalogs.
spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

for name, df in outputs.items():
    print("===", name, "===")
    df.show(5, truncate=60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed
# MAGIC
# MAGIC An assertion, not a `display()`. When this runs as a job in CI, a silently
# MAGIC empty table is the failure mode that matters.

# COMMAND ----------

for table in ["franchise_revenue", "product_revenue"]:
    n = spark.table(catalog + "." + schema + "." + table).count()
    print(table, "->", n, "rows")
    assert n > 0, table + " is empty"

print("OK")
