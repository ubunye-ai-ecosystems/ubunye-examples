# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Model monitoring — and the rollback that isn't always the answer
# MAGIC
# MAGIC **MLflow is a logbook.** It records what happened at *training* time and never
# MAGIC looks at the model again. It will keep reporting `test_r2 = 0.969` while the
# MAGIC model under-predicts every fare in production, and it cannot change what is
# MAGIC serving.
# MAGIC
# MAGIC A quality gate (example 04) stops a bad model being **born**. Monitoring catches
# MAGIC the one that got past, and the one that was fine until the world moved.
# MAGIC
# MAGIC This notebook runs the monitor against **two different failures**, because they
# MAGIC have two different remedies and telling them apart is the entire job:
# MAGIC
# MAGIC | | input drift | target drift | decay | better version exists | correct action |
# MAGIC |---|---|---|---|---|---|
# MAGIC | **A · the world moved** (fares inflate 35%) | no | **yes** | **yes** | no | **retrain** |
# MAGIC | **B · a bad model was promoted** | no | no | no | **yes** | **roll back** |
# MAGIC
# MAGIC Read the zeros. In **A**, the inputs do not move at all — inflation changes what
# MAGIC a trip *costs*, not how far it *goes* — so an input-drift dashboard, which is
# MAGIC what most "drift monitoring" actually is, reports **all green** while the model
# MAGIC is wrong on every row. In **B**, *nothing* drifts and *nothing* decays: the model
# MAGIC did not rot, it was born broken, so its live error matches its (bad) baseline and
# MAGIC the decay check stays silent.
# MAGIC
# MAGIC Each check catches a failure the others cannot. And a rollback only ever fixes
# MAGIC **B** — rolling back a changed world just picks a different model that also never
# MAGIC saw it.

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
import sys
from pathlib import Path

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
task_dir = dbutils.widgets.get("task_dir")

if not task_dir:
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    example_root = Path("/Workspace") / Path(nb).parent.parent.relative_to("/")
    task_dir = str(example_root / "pipelines" / "taxi_fare" / "mlops" / "monitor_model")

# This example monitors example 04's model, so it needs 04's model class to load
# the registered artifacts with. Imported, not copied: two copies of a model class
# is how a registry starts loading an artifact into a class that no longer matches
# it — the joblib still deserialises and the predictions quietly mean something else.
examples_root = Path(task_dir).parents[4]
models_dir = str(examples_root / "04_ml_taxi_fare" / "models")
model_store = "/Volumes/" + catalog + "/" + schema + "/model_store"

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["TAXI_MODEL_STORE"] = model_store
os.environ["TAXI_MODELS_DIR"] = models_dir
sys.path.insert(0, models_dir)

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)

print("task_dir  :", task_dir)
print("models_dir:", models_dir)
print("registry  :", model_store)

# COMMAND ----------

import ubunye
from ubunye.models.registry import ModelRegistry, ModelStage

registry = ModelRegistry(model_store)
FQ = catalog + "." + schema


def production():
    _, mv = registry.get_model(
        use_case="taxi_fare", model_name="TaxiFareModel", stage=ModelStage.PRODUCTION
    )
    return mv


def last_incident():
    df = spark.table(FQ + ".monitoring_incidents")
    return df.orderBy("detected_at", ascending=False).first()


def show_run():
    m = spark.table(FQ + ".model_monitoring")
    latest = m.agg({"monitored_at": "max"}).first()[0]
    m.filter(m.monitored_at == latest).select(
        "feature", "kind", "psi", "drift_status", "baseline_mae", "live_mae"
    ).orderBy("psi", ascending=False).show(truncate=False)

    c = spark.table(FQ + ".model_candidates")
    latest_c = c.agg({"evaluated_at": "max"}).first()[0]
    c.filter(c.evaluated_at == latest_c).select(
        "model_version", "role", "live_mae", "best_challenger"
    ).orderBy("live_mae").show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Start from a known-good champion
# MAGIC
# MAGIC Previous runs of this notebook leave versions behind — including the broken one
# MAGIC it registers on purpose below. Promote whichever version has the best *recorded*
# MAGIC test error, so the run starts from a healthy production model no matter what
# MAGIC happened last time.

# COMMAND ----------

versions = registry.list_versions("taxi_fare", "TaxiFareModel")
healthiest = min(versions, key=lambda v: v.metrics.get("test_mae", float("inf")))
registry.rollback(
    use_case="taxi_fare", model_name="TaxiFareModel", to_version=healthiest.version
)
print("champion:", healthiest.version, "| its test_mae:", round(healthiest.metrics["test_mae"], 4))

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario A · The world moved
# MAGIC
# MAGIC Fares inflate 35%. Distances, durations and times of day are untouched.

# COMMAND ----------

os.environ["DRIFT_FACTOR"] = "1.35"

before_a = production().version
ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)
show_run()

# COMMAND ----------

a = last_incident()
print("diagnosis:", a["diagnosis"], "->", a["action"])
print(a["detail"])

# The alarm must actually fire. A monitor whose alarm never fires is a smoke
# detector with no battery: it passes every test and is wired to nothing.
assert a["target_drift"], "the injected +35% fare drift was NOT detected"
assert a["decay_detected"], "the model's error went up 9x and the monitor did not notice"

# ...and the inputs must be CLEAN. This is the point of the scenario: an
# input-drift dashboard would be showing all green right now.
assert not a["input_drift"], "inputs should not have moved: " + str(a["drifted_inputs"])

# Rolling back cannot fix a changed world — every registered version learned the
# old one. The monitor must MEASURE that and refuse, not roll the dice.
assert a["action"] == "retrain", "expected retrain, got " + a["action"]
assert production().version == before_a, "production was churned for no gain"

print()
print("inputs green, target drifted, error 9x, no better version -> retrain. Production untouched.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario B · A bad model was promoted
# MAGIC
# MAGIC The canonical rollback case, built the way it really happens: a training run
# MAGIC picks up a broken upstream partition, fits on a handful of rows, and somebody
# MAGIC promotes it by hand — around the gate that exists to stop exactly this.
# MAGIC
# MAGIC Its recorded metrics are **honest and bad**. Nothing is faked.

# COMMAND ----------

from sklearn.ensemble import HistGradientBoostingRegressor
from taxi_fare_model import TaxiFareModel

features = spark.table(FQ + ".taxi_features")
train_pdf = features.filter("split = 'train'").limit(200).toPandas()  # the broken partition
test_pdf = features.filter("split = 'test'").toPandas()

broken = TaxiFareModel()
broken._model = HistGradientBoostingRegressor(max_iter=1, max_depth=1, random_state=0)
train_metrics = broken.train(train_pdf)
test_metrics = broken.validate(test_pdf)  # honest metrics, on the real test split

bad_version = registry.register(
    use_case="taxi_fare",
    model_name="TaxiFareModel",
    version=None,
    model=broken,
    metrics={**train_metrics, **test_metrics},
)
# Promoted with no gates — which is what "somebody promoted it by hand" means.
registry.promote(
    use_case="taxi_fare",
    model_name="TaxiFareModel",
    version=bad_version.version,
    to_stage=ModelStage.PRODUCTION,
)
print("promoted", bad_version.version, "with an honest test_mae of", round(test_metrics["test_mae"], 3))

# COMMAND ----------

os.environ["DRIFT_FACTOR"] = "1.0"  # the world is fine. The model is not.

ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)
show_run()

# COMMAND ----------

b = last_incident()
print("diagnosis:", b["diagnosis"], "->", b["action"])
print(b["detail"])

# Nothing drifted — the world is exactly as it was.
assert not b["input_drift"] and not b["target_drift"], "nothing should have drifted here"

# And nothing DECAYED. This is the lesson: the model did not rot, it was born
# broken, so its live error matches its own bad baseline and the decay check is
# silent. A decay-only monitor sees nothing wrong and this model serves forever.
assert not b["decay_detected"], (
    "a model that was always bad has not decayed — if this fires, the decay check "
    "is measuring against the wrong baseline"
)

# What catches it is the question almost nobody asks: is the live model still the
# best model I have?
assert b["action"] == "rollback", "expected rollback, got " + b["action"]
assert b["best_challenger_mae"] < b["live_mae"], "rolled back to something no better"

after_b = production()
assert after_b.version != bad_version.version, "the broken model is STILL in production"
assert after_b.version == b["best_challenger"], "rolled back to a version we did not measure"

print()
print("broken model  :", bad_version.version, "live MAE", round(b["live_mae"], 3))
print("rolled back to:", after_b.version, "live MAE", round(b["best_challenger_mae"], 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production is healthy again
# MAGIC
# MAGIC The registry moved — the step MLflow cannot take, because it does not know what
# MAGIC is serving. And a production model still exists: rolling back to *nothing* is not
# MAGIC a fix, it is an outage, and example 04's inference task has to be able to load
# MAGIC something on its next run.

# COMMAND ----------

final = production()
assert final.stage == ModelStage.PRODUCTION
assert final.metrics["test_mae"] < 1.0, "production is not a healthy model"

print("production:", final.version, "| test_mae", round(final.metrics["test_mae"], 4))
print()
print("OK")
