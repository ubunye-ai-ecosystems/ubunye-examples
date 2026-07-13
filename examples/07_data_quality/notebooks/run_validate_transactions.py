# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Data quality — a contract, enforced
# MAGIC
# MAGIC Declares what a valid transaction **is**, splits the data into rows that meet
# MAGIC the contract and rows that do not, writes **both**, and fails the run when the
# MAGIC breach is structural rather than incidental.
# MAGIC
# MAGIC **Bad rows are quarantined, not dropped.** A dropped row is a silent data loss
# MAGIC — the counts move, nobody knows why, and the evidence is gone. A quarantined row
# MAGIC is a bug report with the payload attached.
# MAGIC
# MAGIC Five deliberately broken rows are injected, because an example whose failure
# MAGIC path never runs proves nothing: it would pass, look healthy, and keep passing
# MAGIC even if the quarantine were broken.

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
from pathlib import Path

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
dt = dbutils.widgets.get("dt")
mode = dbutils.widgets.get("mode")
task_dir = dbutils.widgets.get("task_dir")

if not task_dir:
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    example_root = Path("/Workspace") / Path(nb).parent.parent.relative_to("/")
    task_dir = str(example_root / "pipelines" / "quality" / "contracts" / "validate_transactions")

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)
os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

print("task_dir:", task_dir)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The contract report
# MAGIC
# MAGIC One row per rule, **including the rules that passed**. The zeros are the point:
# MAGIC a rule that fails nothing today and 4,000 rows tomorrow is the earliest warning
# MAGIC you will get that a source system changed — and you cannot see that change if
# MAGIC you only ever record the failures.

# COMMAND ----------

results = spark.table(catalog + "." + schema + ".contract_results")
results.select("rule", "severity", "rows_checked", "rows_failed", "passed").orderBy(
    "rows_failed", ascending=False
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What was quarantined, and why

# COMMAND ----------

quarantined = spark.table(catalog + "." + schema + ".quarantined_transactions")
quarantined.select(
    "transactionID", "quantity", "unitPrice", "totalPrice", "paymentMethod", "broken_rules"
).show(truncate=False)

# COMMAND ----------

clean = spark.table(catalog + "." + schema + ".clean_transactions")

n_clean = clean.count()
n_quarantined = quarantined.count()
print("clean      :", n_clean)
print("quarantined:", n_quarantined)

assert n_clean > 0, "everything was quarantined — the contract is wrong, not the data"

# The five injected faults must ALL be caught. If they are not, the quarantine is
# broken and every real bad row would be sailing straight into the clean table.
injected = [900001, 900002, 900003, 900004, 900005]
caught = {r["transactionID"] for r in quarantined.select("transactionID").collect()}
missed = [i for i in injected if i not in caught]
assert not missed, "the quarantine MISSED deliberately broken rows: " + str(missed)

# And none of them may reach the clean table.
leaked = clean.filter(clean.transactionID.isin(injected)).count()
assert leaked == 0, str(leaked) + " broken rows leaked into clean_transactions"

# Every rule must be reported, passing or not.
#
# NOT `results.count() == 7`: contract_results is an APPEND table, so the second
# run makes that 14 and the assertion only ever passes on a virgin table. It
# passed the first time and failed the second — which is precisely the class of
# bug this whole example exists to catch.
latest = results.agg({"checked_at": "max"}).first()[0]
this_run = results.filter(results.checked_at == latest)

assert this_run.count() == 7, "the contract report is missing rules from this run"
assert results.select("rule").distinct().count() == 7, "a rule vanished from the report"

print()
print("all 5 injected faults quarantined, none leaked")
print("OK")
