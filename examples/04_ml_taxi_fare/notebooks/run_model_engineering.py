# Databricks notebook source
# MAGIC %md
# MAGIC # ML — model engineering
# MAGIC
# MAGIC Clean, engineer, split, train, **validate on data the model never saw**, and register to the model registry — but only if it clears the quality gate. One task, because none of those steps is useful on its own.
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
    task_dir = str(Path("/Workspace") / here.relative_to("/") / "../pipelines/taxi_fare/ml/model_engineering")

print(f"task_dir : {task_dir}")
print(f"target   : {catalog}.{schema}")

# COMMAND ----------

# The schema and volumes have to exist before the writers run — the engine
# writes tables, it does not provision catalogs.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.model_store")

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["TAXI_MODEL_STORE"] = f"/Volumes/{catalog}/{schema}/model_store"
os.environ["MLFLOW_EXPERIMENT_NAME"] = f"/Shared/ubunye_examples/taxi_fare"

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

m = spark.table(f"{catalog}.{schema}.taxi_fare_model_metrics")
m.filter("metric IN ('train_rmse','test_rmse','test_r2')").show(truncate=False)
assert m.count() > 0, "no metrics recorded"

# The model must be on the volume, or batch_inference has nothing to load.
import os
store = f"/Volumes/{catalog}/{schema}/model_store"
assert os.path.exists(store), "model store volume missing"
print("registry:", os.listdir(store))
print("
OK")
