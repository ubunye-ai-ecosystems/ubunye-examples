# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Distillation, part 2 — fine-tune the student
# MAGIC
# MAGIC Fine-tunes **DistilBERT** (open source, 66M parameters) on the teacher's labels.
# MAGIC CPU only — serverless has no GPU, and a model this size does not need one.
# MAGIC
# MAGIC The result is a model you **own**: the weights sit on a volume, inference costs
# MAGIC nothing per call, and no endpoint can rate-limit you or be deprecated out from
# MAGIC under you.
# MAGIC
# MAGIC It is governed exactly like the sklearn model in example 04 — split, fit, judged
# MAGIC on data it never saw, gated, registered. That a transformer is doing the fitting
# MAGIC changes nothing about that.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark,ml]==0.3.0" transformers torch --quiet

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
    task_dir = str(example_root / "pipelines" / "reviews" / "nlp" / "finetune_classifier")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
spark.sql("CREATE VOLUME IF NOT EXISTS " + catalog + "." + schema + ".student_store")

model_store = "/Volumes/" + catalog + "/" + schema + "/student_store"

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["STUDENT_MODEL_STORE"] = model_store
os.environ["TEACHER_ENDPOINT"] = "databricks-meta-llama-3-1-8b-instruct"

experiment = "/Shared/ubunye_examples/review_sentiment"
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs("/Shared/ubunye_examples")
os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment

import torch

print("task_dir:", task_dir)
print("registry:", model_store)
print("device  :", "cuda" if torch.cuda.is_available() else "cpu (66M params — fine)")

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)
outputs["student_metrics"].show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed
# MAGIC
# MAGIC `test_accuracy` is the student's agreement with the teacher **on reviews it never
# MAGIC saw**. That is the number that says whether the distillation worked. A student
# MAGIC that cannot reproduce its teacher out of sample has not compressed anything —
# MAGIC it is merely cheaper, and wrong.

# COMMAND ----------

metrics = spark.table(catalog + "." + schema + ".student_model_metrics")
metrics.filter("metric LIKE 'test_%'").orderBy("recorded_at", ascending=False).show(
    10, truncate=False
)
assert metrics.count() > 0, "no metrics recorded"

from ubunye.models.registry import ModelRegistry, ModelStage

registry = ModelRegistry(model_store)
path, version = registry.get_model(
    use_case="review_sentiment", model_name="ReviewSentimentModel", stage=ModelStage.PRODUCTION
)
print("student in production:", version.version)
print("agreement with the teacher on unseen reviews:", version.metrics.get("test_accuracy"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use the model you now own
# MAGIC
# MAGIC No endpoint. No per-call cost. No rate limit. Just weights on a volume.
# MAGIC
# MAGIC Scored on **real held-out reviews** — the kind of text the student was trained
# MAGIC on. It was first checked with three invented one-line sentences and got them
# MAGIC wrong, which was not a bug in the model: it learned from 100-to-200-word formal
# MAGIC reviews, and a terse one-liner is out of distribution. A model is only as good
# MAGIC as its agreement on data that looks like its training data, and pretending
# MAGIC otherwise is how models get deployed into inputs they have never seen.

# COMMAND ----------

import sys

sys.path.insert(0, str(Path(task_dir).parents[3] / "models"))
from review_sentiment_model import ReviewSentimentModel

student = ReviewSentimentModel.load(path)

held_out = (
    spark.table(catalog + "." + schema + ".labelled_reviews")
    .filter("split = 'test'")
    .select("review_text", "label")
    .toPandas()
)

scored = student.predict(held_out)
scored["agrees_with_teacher"] = scored["predicted_label"] == scored["label"]

print(
    scored[["label", "predicted_label", "agrees_with_teacher", "review_text"]]
    .head(6)
    .to_string(index=False, max_colwidth=60)
)
print()
print("agreement on held-out reviews:", round(scored["agrees_with_teacher"].mean(), 3))

# The student must actually use more than one class. A model that answers
# "positive" to everything can still score well when the corpus leans positive —
# that is the collapse the recall gate exists to catch.
predicted_classes = set(scored["predicted_label"])
assert len(predicted_classes) > 1, (
    "the student predicted a single class for every review — it collapsed to the "
    "majority and learned nothing"
)
assert scored["agrees_with_teacher"].mean() > 0.7, "the student does not reproduce its teacher"

print()
print("OK")
