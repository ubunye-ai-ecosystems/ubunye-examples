# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · TensorFlow/Keras — the same net, the third framework
# MAGIC
# MAGIC One of three trainers. The `config.yaml` behind this notebook is **byte-for-byte
# MAGIC identical** to the other two — same inputs, same splits, same gate, same registry,
# MAGIC same output table. Only `model.py` differs.
# MAGIC
# MAGIC Needs the feature table from example **04** (`ml_taxi_fare`). Run that first.
# MAGIC
# MAGIC ## Read the pip line before you copy it
# MAGIC
# MAGIC Three things about TensorFlow here, all measured on this workspace, all silent
# MAGIC when you get them wrong:
# MAGIC
# MAGIC 1. Serverless is **aarch64**. `tensorflow-cpu` publishes **no ARM wheels** —
# MAGIC    `pip install tensorflow-cpu` fails with *No matching distribution found*.
# MAGIC    Plain `tensorflow` is the one that exists.
# MAGIC 2. Unpinned, TensorFlow pulls **numpy 2** and **protobuf 7**, which break scipy,
# MAGIC    scikit-learn and MLflow in the same environment — while pip prints
# MAGIC    *Successfully installed* and exits 0. `numpy<2` and `protobuf<5` keep the
# MAGIC    whole environment working.
# MAGIC 3. TensorFlow OOMs the driver unless its thread pools are capped — see
# MAGIC    `models/fare_keras.py`. It is not the data; it is one allocation arena per core.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark,ml]==0.3.0" tensorflow "numpy<2" "protobuf<5"

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
    task_dir = str(example_root / "pipelines" / "fare_bench" / "ml" / "train_keras")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["BENCH_MODEL_STORE"] = "/Volumes/" + catalog + "/" + schema + "/model_store"

# MLflow needs its parent folder to exist, or set_experiment fails and the logging
# silently does nothing at all.
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs("/Shared/ubunye_examples")
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Shared/ubunye_examples/fare_benchmark"

print("task_dir:", task_dir)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

FQ = catalog + "." + schema
metrics = spark.table(FQ + ".fare_bench_metrics")
latest = metrics.filter("framework = 'tensorflow'")
latest = latest.filter(latest.recorded_at == latest.agg({"recorded_at": "max"}).first()[0])
latest.select("framework", "model_name", "version", "metric", "value", "train_seconds").orderBy(
    "metric"
).show(truncate=False)

# COMMAND ----------

rows = {r["metric"]: r["value"] for r in latest.collect()}

# The model was judged on rows it never saw, and it cleared the bar — otherwise
# benchmark.run() would have raised and nothing would be registered.
assert rows["test_rmse"] < 6.0, "test RMSE " + str(rows["test_rmse"]) + " — worse than the gate allows"
assert rows["test_r2"] > 0.55, "test R2 " + str(rows["test_r2"]) + " — worse than the gate allows"

# And it scored the ranking split. Without this the leaderboard has nothing to rank.
assert "score_rmse" in rows, "the model never scored the ranking split"

print()
print("test_rmse :", round(rows["test_rmse"], 4))
print("score_rmse:", round(rows["score_rmse"], 4), "  <- what the leaderboard ranks on")
print("fit took  :", round(rows["train_seconds"] if "train_seconds" in rows else 0, 1), "s")
print()
print("OK")
