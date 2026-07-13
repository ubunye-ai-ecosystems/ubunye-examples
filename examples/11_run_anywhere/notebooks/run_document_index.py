# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · The same task, on Databricks
# MAGIC
# MAGIC This notebook runs the **identical task directory** that
# MAGIC `platforms/local/run.sh`, the Docker image and the Kubernetes Job run. Not a
# MAGIC copy, not a port — the same `config.yaml` and the same `transformations.py`,
# MAGIC byte for byte.
# MAGIC
# MAGIC A notebook is a **runner**, not a pipeline. `dbutils` appears here and nowhere
# MAGIC near the task, which is exactly the boundary that makes the task portable:
# MAGIC every platform brings its own way of launching things, and none of them belongs
# MAGIC inside the pipeline.
# MAGIC
# MAGIC What changes between platforms is three environment variables. Nothing else.
# MAGIC
# MAGIC | | local / Docker / k8s | here |
# MAGIC |---|---|---|
# MAGIC | `SPARK_MASTER` | `local[*]` | *ignored — Databricks owns the session* |
# MAGIC | `UBUNYE_SINK` | `s3` (path) | `unity` (table) |
# MAGIC | `UBUNYE_DATA_ROOT` | `file:///tmp/ubunye` | `/Volumes/<catalog>/<schema>` |

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
    task_dir = str(example_root / "pipelines" / "portable" / "ingestion" / "document_index")
else:
    example_root = Path(task_dir).parents[3]

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
spark.sql("CREATE VOLUME IF NOT EXISTS " + catalog + "." + schema + ".portable")

data_root = "/Volumes/" + catalog + "/" + schema + "/portable"

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema
os.environ["UBUNYE_SINK"] = "unity"  # <- the only real difference
os.environ["UBUNYE_DATA_ROOT"] = data_root

# Stage the corpus. Every platform does this its own way — that is the bootstrap,
# and the bootstrap is allowed to differ. The pipeline is not.
corpus = data_root + "/corpus"
os.makedirs(corpus, exist_ok=True)
for txt in (example_root / "data" / "corpus").glob("*.txt"):
    shutil.copyfile(str(txt), corpus + "/" + txt.name)

print("task_dir :", task_dir)
print("sink     :", os.environ["UBUNYE_SINK"])
print("data root:", data_root)
print("corpus   :", len(os.listdir(corpus)), "files")

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The fingerprint
# MAGIC
# MAGIC The same script the other platforms run. If this hash matches theirs, the claim
# MAGIC on the tin is true; if it does not, it is false, and we would rather know.

# COMMAND ----------

import sys

sys.path.insert(0, str(Path(task_dir).parents[3].parent.parent / "platforms"))
from fingerprint import fingerprint  # noqa: E402

FQ = catalog + "." + schema
report = [
    fingerprint(spark, FQ + ".portable_documents"),
    fingerprint(spark, FQ + ".portable_document_chunks"),
]

import hashlib

for entry in report:
    print(entry["target"], "\n  rows  :", entry["rows"], "\n  sha256:", entry["sha256"])

combined = hashlib.sha256("".join(e["sha256"] for e in report).encode()).hexdigest()
print()
print("FINGERPRINT=" + combined)

# COMMAND ----------

assert report[0]["rows"] == 6, "expected 6 documents, got " + str(report[0]["rows"])
assert report[1]["rows"] > 6, "chunking did not run — fewer chunks than documents"

print()
print("OK")
