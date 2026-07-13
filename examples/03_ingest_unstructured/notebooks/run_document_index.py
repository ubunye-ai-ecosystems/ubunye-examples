# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Unstructured ingestion — free text + raw files
# MAGIC
# MAGIC Reads 204 real customer reviews from a table **and** raw `.txt` files off a
# MAGIC Unity Catalog volume via Spark's `binaryFile` source, then cuts both into
# MAGIC overlapping word windows — the shape a vector index wants.
# MAGIC
# MAGIC Nothing is downloaded. Serverless compute cannot be relied on to reach the
# MAGIC internet, and an example that needs egress is one most people cannot run.

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
import shutil
from pathlib import Path

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
task_dir = dbutils.widgets.get("task_dir")

if not task_dir:
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    example_root = Path("/Workspace") / Path(nb).parent.parent.relative_to("/")
    task_dir = str(example_root / "pipelines" / "docs" / "ingestion" / "document_index")

print("task_dir:", task_dir)
print("target  :", catalog + "." + schema)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
spark.sql("CREATE VOLUME IF NOT EXISTS " + catalog + "." + schema + ".corpus")

# Copy the committed corpus onto the volume. Spark cannot read /Workspace files
# with the binaryFile source reliably; a volume is a real path every executor sees.
example_root = Path(task_dir).parents[3]
corpus_src = example_root / "data" / "corpus"
corpus_dst = Path("/Volumes") / catalog / schema / "corpus"

copied = 0
for f in sorted(corpus_src.glob("*.txt")):
    shutil.copy(str(f), str(corpus_dst / f.name))
    copied += 1

print("copied", copied, "documents ->", str(corpus_dst))

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

# COMMAND ----------

docs = spark.table(catalog + "." + schema + ".documents")
chunks = spark.table(catalog + "." + schema + ".document_chunks")

n_docs = docs.count()
n_chunks = chunks.count()
n_files = docs.filter("source = 'file'").count()
n_reviews = docs.filter("source = 'review'").count()

print("documents     ", n_docs)
print("  from files  ", n_files)
print("  from reviews", n_reviews)
print("chunks        ", n_chunks)

assert n_docs > 0, "no documents"
assert n_files > 0, "the binary reader picked up nothing — is the volume populated?"
assert n_reviews > 0, "no reviews were read"
# If chunks == docs, the window is bigger than every document and the chunking
# never actually ran.
assert n_chunks > n_docs, "chunking did not split anything"

print("OK")
