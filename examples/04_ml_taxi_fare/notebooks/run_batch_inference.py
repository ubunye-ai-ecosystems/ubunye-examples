# Databricks notebook source
# MAGIC %md
# MAGIC # ML — batch inference
# MAGIC
# MAGIC Loads whatever model is in **production** and scores trips it has never seen. Shipping a better model is a promotion in the registry, not a change to this code.
# MAGIC
# MAGIC Runs on serverless. Press **Run all** — nothing to configure.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer from this notebook)")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Unity Catalog schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark,ml]==0.3.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from pathlib import Path

# Widgets have to be re-read: restartPython() wiped everything above.
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
task_dir = dbutils.widgets.get("task_dir")

# When a job launches this notebook it passes task_dir. When a human opens it
# from a Git folder there is no job, so work it out from where this file sits —
# that is what lets the same notebook be both deployable and openable.
if not task_dir:
    here = Path(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    ).parent
    task_dir = str(Path("/Workspace") / here.relative_to("/") / "../pipelines/taxi_fare/ml/batch_inference")

print(f"task_dir : {task_dir}")
print(f"target   : {catalog}.{schema}")

# COMMAND ----------

# The schema and volumes have to exist before the writers run — the engine
# writes tables, it does not provision catalogs.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")


os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["TAXI_MODEL_STORE"] = f"/Volumes/{catalog}/{schema}/model_store"

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

for name, df in outputs.items():
    print(f"
=== {name} ===")
    df.show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed
# MAGIC
# MAGIC An assertion, not a `display()`: when this notebook runs as a job in CI, a
# MAGIC silent empty table is the failure mode that matters.

# COMMAND ----------

p = spark.table(f"{catalog}.{schema}.taxi_fare_predictions")
print(f"predictions    {p.count():>8,}")
p.select("trip_distance", "fare_amount", "predicted_fare", "error", "model_version").show(5)
assert p.count() > 0, "nothing was scored"

# Sanity: the model should be roughly right, not merely present.
mae = p.selectExpr("avg(abs(error)) AS mae").first()["mae"]
print(f"mean absolute error on unseen trips: ${mae:.2f}")
assert mae < 10, f"predictions are not credible (MAE ${mae:.2f})"
print("
OK")
