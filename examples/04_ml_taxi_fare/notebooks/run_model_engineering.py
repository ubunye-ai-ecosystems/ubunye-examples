# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · ML — model engineering
# MAGIC
# MAGIC Clean, engineer, split, train, **validate on data the model never saw**, and
# MAGIC register — but only if it clears the quality gate.
# MAGIC
# MAGIC One task, because none of those steps is useful on its own. A model that
# MAGIC fails the gate is not an artifact with a warning label; it is a failed
# MAGIC experiment, and it does not go in the registry, because somebody downstream
# MAGIC will eventually load "the production model" without reading the label.

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
    task_dir = str(example_root / "pipelines" / "taxi_fare" / "ml" / "model_engineering")

print("task_dir:", task_dir)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
spark.sql("CREATE VOLUME IF NOT EXISTS " + catalog + "." + schema + ".model_store")

model_store = "/Volumes/" + catalog + "/" + schema + "/model_store"

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
# The registry is a directory of files. /tmp is the driver's disk and the
# executors cannot see it; a volume is the only writable path on serverless.
os.environ["TAXI_MODEL_STORE"] = model_store
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Shared/ubunye_examples/taxi_fare"

print("registry:", model_store)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

for name, df in outputs.items():
    print("===", name, "===")
    df.show(20, truncate=60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed
# MAGIC
# MAGIC The metrics table is the audit trail. `test_*` are the numbers that count —
# MAGIC they were scored on trips the model never saw.

# COMMAND ----------

metrics = spark.table(catalog + "." + schema + ".taxi_fare_model_metrics")
metrics.filter("metric LIKE 'test_%' OR metric LIKE 'train_%'").show(truncate=False)

assert metrics.count() > 0, "no metrics recorded"

# The artifact has to be on the volume, or batch_inference has nothing to load.
import os

assert os.path.exists(model_store), "model store volume is missing"
print("registry contents:", os.listdir(model_store))

print("OK")
