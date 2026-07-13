# Databricks notebook source
# MAGIC %md
# MAGIC # Egress probe
# MAGIC
# MAGIC Serverless compute restricts outbound internet to a set of trusted domains.
# MAGIC Which domains, exactly, is not documented — so an example that calls a public
# MAGIC API is a gamble until this notebook has run.
# MAGIC
# MAGIC This answers one question: **what can a notebook actually reach?** The answer
# MAGIC decides whether example 02 ingests from a public REST API, or from the
# MAGIC workspace's own Unity Catalog REST API (which is guaranteed reachable).

# COMMAND ----------

import json
import urllib.request

TIMEOUT = 15

TARGETS = [
    # The candidate public APIs for a REST ingestion example.
    ("open-meteo (public API, no auth)", "https://api.open-meteo.com/v1/forecast?latitude=-26.2&longitude=28.05&hourly=temperature_2m"),
    ("USGS earthquakes (public API)", "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson"),
    # Whether ANY example may download its own data.
    ("raw.githubusercontent.com", "https://raw.githubusercontent.com/databricks/databricks-cli/main/README.md"),
    # The foundation of everything: the notebooks all %pip install from here.
    ("pypi.org (the engine itself)", "https://pypi.org/pypi/ubunye-engine/json"),
]


def probe(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ubunye-egress-probe"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return True, f"HTTP {r.status}, {len(r.read(2048))} bytes read"
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:90]


results = {}
print("=" * 78)
for label, url in TARGETS:
    ok, detail = probe(url)
    results[label] = ok
    print(f"{'REACHABLE' if ok else 'BLOCKED  '}  {label}")
    print(f"            {detail}")
print("=" * 78)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The workspace's own REST API
# MAGIC
# MAGIC The fallback for example 02. A notebook can always reach its own control
# MAGIC plane — if this works, a REST ingestion example is possible even when the
# MAGIC public internet is not.

# COMMAND ----------

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

url = f"https://{host}/api/2.1/unity-catalog/tables?catalog_name=samples&schema_name=bakehouse&max_results=5"
try:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        payload = json.loads(r.read())
    tables = payload.get("tables", [])
    print(f"REACHABLE  workspace UC REST API — {len(tables)} tables returned")
    print(f"           next_page_token present: {'next_page_token' in payload}")
    for t in tables[:3]:
        print(f"           - {t.get('full_name')}")
    results["workspace UC REST API"] = True
except Exception as exc:
    print(f"BLOCKED    workspace UC REST API: {type(exc).__name__}: {str(exc)[:90]}")
    results["workspace UC REST API"] = False

# COMMAND ----------

print("SUMMARY")
for label, ok in results.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")

dbutils.notebook.exit(json.dumps(results))
