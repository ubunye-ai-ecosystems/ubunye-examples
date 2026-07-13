# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Structured ingestion — a public REST API
# MAGIC
# MAGIC Pulls a 7-day hourly weather forecast from the **Open-Meteo** public API. No
# MAGIC key, no signup, no quota.
# MAGIC
# MAGIC The connector does the HTTP: query params, rate limiting, retry on 429/503.
# MAGIC `transformations.py` does what no connector can — the API returns ONE document
# MAGIC holding parallel arrays of 168 timestamps and 168 temperatures, and turning
# MAGIC that back into rows is a business rule, not a setting.
# MAGIC
# MAGIC **Egress:** serverless restricts outbound internet to trusted domains.
# MAGIC `api.open-meteo.com` is reachable; `raw.githubusercontent.com` is **not** —
# MAGIC verified by running `scripts/verify_egress.py` on the cluster. That is why no
# MAGIC example here downloads its own data.

# COMMAND ----------

dbutils.widgets.text("task_dir", "", "Task directory (blank = infer)")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "ubunye_examples", "Schema")
dbutils.widgets.text("dt", "2026-07-13", "Data timestamp")
dbutils.widgets.dropdown("mode", "PROD", ["DEV", "PROD"], "Run mode")

# COMMAND ----------

# MAGIC %pip install "ubunye-engine[spark]==0.3.0" requests

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
    task_dir = str(example_root / "pipelines" / "weather" / "ingestion" / "hourly_forecast")

print("task_dir:", task_dir)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS " + catalog + "." + schema)

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

for name, df in outputs.items():
    print("===", name, "===")
    df.show(5, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check it landed

# COMMAND ----------

fc = spark.table(catalog + "." + schema + ".hourly_forecast")
n = fc.count()
print("forecast hours:", n)

fc.groupBy("conditions").count().orderBy("count", ascending=False).show()
fc.select("observed_at", "temperature_2m", "precipitation", "wind_speed_10m", "conditions").orderBy(
    "observed_at"
).show(5, truncate=False)

assert n > 0, "the API returned nothing — can this workspace reach api.open-meteo.com?"

# 7 days x 24 hours. If the arrays were exploded separately instead of zipped, this
# would be 168 x 168 = 28,224 rows of nonsense, each hour paired with somebody
# else's temperature.
assert 150 < n < 200, "expected ~168 hourly rows, got " + str(n)
assert fc.filter("temperature_2m IS NULL").count() == 0, "measures did not line up with hours"

print("OK")
