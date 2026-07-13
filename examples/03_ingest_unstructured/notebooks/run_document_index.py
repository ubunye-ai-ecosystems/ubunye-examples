# Databricks notebook source
# MAGIC %md
# MAGIC # Unstructured ingestion — free text + raw files
# MAGIC
# MAGIC Reads 204 real customer reviews from a table **and** raw `.txt` files off a Unity Catalog volume (Spark's `binaryFile` source), then chunks both into overlapping windows ready for embedding.
# MAGIC
# MAGIC Runs on serverless. Press **Run all** — nothing to configure.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer from this notebook)")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Unity Catalog schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark]==0.3.0"

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
    task_dir = str(Path("/Workspace") / here.relative_to("/") / "../pipelines/docs/ingestion/document_index")

print(f"task_dir : {task_dir}")
print(f"target   : {catalog}.{schema}")

# COMMAND ----------

# The schema and volumes have to exist before the writers run — the engine
# writes tables, it does not provision catalogs.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.corpus")

# Copy the committed corpus onto the volume. Nothing is downloaded: serverless
# compute cannot be relied on to reach the internet, and an example that needs
# egress is an example most people cannot run.
import shutil
corpus_src = Path(task_dir).parents[3] / "data" / "corpus"
corpus_dst = Path(f"/Volumes/{catalog}/{schema}/corpus")
copied = 0
for f in corpus_src.glob("*.txt"):
    shutil.copy(str(f), str(corpus_dst / f.name))
    copied += 1
print(f"corpus   : {copied} files -> {corpus_dst}")

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema


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

docs = spark.table(f"{catalog}.{schema}.documents")
chunks = spark.table(f"{catalog}.{schema}.document_chunks")
print(f"documents      {docs.count():>8,}")
print(f"chunks         {chunks.count():>8,}")
print(f"  from files   {docs.filter(\"source = 'file'\").count():>8,}")
print(f"  from reviews {docs.filter(\"source = 'review'\").count():>8,}")
assert docs.count() > 0, "no documents"
assert chunks.count() > docs.count(), "chunking did not split anything"
assert docs.filter("source = 'file'").count() > 0, "the binary reader picked up nothing"
print("
OK")
