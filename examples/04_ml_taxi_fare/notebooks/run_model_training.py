# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · ML — model training
# MAGIC
# MAGIC Fit on `train`. Score on `test`, which the model has never seen. Judge **those**
# MAGIC numbers against a bar agreed in advance. Only then register, and only then
# MAGIC promote to production. Log the lot to MLflow.
# MAGIC
# MAGIC A model that fails the gate is **not registered at all** — so nothing
# MAGIC downstream can load it by accident.

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
    task_dir = str(example_root / "pipelines" / "taxi_fare" / "ml" / "model_training")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
spark.sql("CREATE VOLUME IF NOT EXISTS " + catalog + "." + schema + ".model_store")

model_store = "/Volumes/" + catalog + "/" + schema + "/model_store"

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
# The registry is a directory of files. A UC volume is the only writable path on
# serverless — /tmp is the driver's disk and the executors cannot see it.
os.environ["TAXI_MODEL_STORE"] = model_store

# MLflow will create the EXPERIMENT but not the folder it lives in, and
# set_experiment() on a path whose parent is missing fails quietly. The training
# task treats MLflow as best-effort — an MLflow outage must not lose you a model
# that passed its gate — which means a missing folder would go unnoticed forever.
experiment = "/Shared/ubunye_examples/taxi_fare"
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs("/Shared/ubunye_examples")
os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment

print("task_dir:", task_dir)
print("registry:", model_store)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)
outputs["model_metrics"].show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed
# MAGIC
# MAGIC `test_*` are the numbers that count — scored on trips the model never saw.

# COMMAND ----------

metrics = spark.table(catalog + "." + schema + ".taxi_model_metrics")
metrics.filter("metric LIKE 'test_%'").orderBy("recorded_at", ascending=False).show(8, truncate=False)

assert metrics.count() > 0, "no metrics recorded"

# The model must be registered in production, or batch_inference has nothing to load.
from ubunye.models.registry import ModelRegistry, ModelStage

registry = ModelRegistry(model_store)
path, version = registry.get_model(
    use_case="taxi_fare", model_name="TaxiFareModel", stage=ModelStage.PRODUCTION
)
print("production model:", version.version)
print("artifact        :", path)
print("test_rmse       :", version.metrics.get("test_rmse"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow
# MAGIC
# MAGIC Asserted, not assumed. The training task logs to MLflow on a best-effort
# MAGIC basis, so a failure there is a warning and the model still registers. That is
# MAGIC the right behaviour — and it is exactly why the notebook has to check, or a
# MAGIC broken experiment path would go unnoticed forever.

# COMMAND ----------

import mlflow

mlflow.set_experiment(experiment)
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=3)
print(runs[["run_id", "tags.mlflow.runName", "metrics.test_rmse", "metrics.test_r2"]].to_string())

assert len(runs) > 0, "nothing was logged to MLflow"
assert runs.iloc[0]["metrics.test_rmse"] is not None, "the run has no test metrics"

print("OK")
