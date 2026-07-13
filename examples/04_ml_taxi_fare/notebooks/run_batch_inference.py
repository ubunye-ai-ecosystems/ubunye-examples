# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · ML — batch inference
# MAGIC
# MAGIC Loads whatever model is in **production** and scores trips it has never seen.
# MAGIC
# MAGIC This notebook cannot choose which model is live — the registry decides that.
# MAGIC Shipping a better model is a promotion, not a change to this code.

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
    task_dir = str(example_root / "pipelines" / "taxi_fare" / "ml" / "batch_inference")

model_store = "/Volumes/" + catalog + "/" + schema + "/model_store"

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["TAXI_MODEL_STORE"] = model_store

print("task_dir:", task_dir)
print("registry:", model_store)

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
# MAGIC Not just "did rows appear" but "are the predictions credible". A model that
# MAGIC loaded and scored nonsense passes the first check and fails the second.

# COMMAND ----------

predictions = spark.table(catalog + "." + schema + ".taxi_fare_predictions")
n = predictions.count()
print("predictions:", n)

predictions.select(
    "trip_distance", "fare_amount", "predicted_fare", "error", "model_version"
).show(5)

assert n > 0, "nothing was scored"

mae = predictions.selectExpr("avg(abs(error)) AS mae").first()["mae"]
print("mean absolute error on unseen trips: $" + str(round(mae, 2)))
assert mae < 10, "predictions are not credible (MAE $" + str(round(mae, 2)) + ")"

print("OK")
