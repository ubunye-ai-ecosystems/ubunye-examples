# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · The leaderboard
# MAGIC
# MAGIC Three frameworks, one dataset, one gate, one contract. Ranked on the **score**
# MAGIC split — rows no model was fitted on and no gate was judged against. Ranking on
# MAGIC `test_rmse` would be picking the winner on the exam it was allowed to resit.
# MAGIC
# MAGIC This notebook installs **neither torch nor tensorflow nor sklearn**. It compares
# MAGIC numbers three other tasks already wrote. That is not a shortcut — TensorFlow needs
# MAGIC `numpy<2` on this runtime, and an honest attempt to load all three models into one
# MAGIC interpreter runs straight into a dependency wall.

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
    task_dir = str(example_root / "pipelines" / "fare_bench" / "ml" / "select_champion")

os.environ["UBUNYE_CATALOG"] = catalog
os.environ["UBUNYE_SCHEMA"] = schema

print("task_dir:", task_dir)

# COMMAND ----------

import ubunye

outputs = ubunye.run_task(task_dir=task_dir, dt=dt, mode=mode, lineage=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The result

# COMMAND ----------

FQ = catalog + "." + schema
board = spark.table(FQ + ".fare_bench_leaderboard")

board.select(
    "rank",
    "framework",
    "version",
    "score_rmse",
    "test_rmse",
    "test_r2",
    "train_seconds",
    "is_champion",
    "margin_pct",
    "verdict",
).orderBy("rank").show(truncate=False)

# COMMAND ----------

rows = board.orderBy("rank").collect()
champion = [r for r in rows if r["is_champion"]]

# All three frameworks must be on the board. If one is missing it failed its gate or
# never ran — and a two-horse race quietly presented as a three-way one is worse than
# a failure, because it looks like a result.
frameworks = sorted(r["framework"] for r in rows)
assert frameworks == ["pytorch", "scikit-learn", "tensorflow"], (
    "expected all three frameworks on the leaderboard, got " + str(frameworks)
)

assert len(champion) == 1, "expected exactly one champion, got " + str(len(champion))

# The champion must genuinely be the best on the ranking split — not merely flagged.
best = min(r["score_rmse"] for r in rows)
assert champion[0]["score_rmse"] == best, "the champion is not the best model on the board"

# Everything on the board cleared the same gate.
for r in rows:
    assert r["test_rmse"] < 6.0 and r["test_r2"] > 0.55, r["framework"] + " is on the board but failed the gate"

print()
for r in rows:
    print(
        "  %d. %-13s score_rmse %7.4f   test_rmse %7.4f   test_r2 %6.4f   fit %6.1fs"
        % (
            r["rank"],
            r["framework"],
            r["score_rmse"],
            r["test_rmse"],
            r["test_r2"],
            r["train_seconds"],
        )
    )

margin = champion[0]["margin_pct"]
verdict = champion[0]["verdict"]

print()
print("champion :", champion[0]["framework"], "— score_rmse", round(champion[0]["score_rmse"], 4))
print("margin   :", str(round(margin, 1)) + "% ahead of the runner-up")
print("verdict  :", verdict)

if verdict != "clear win":
    print()
    print("  These three are TIED. Sorting any three numbers produces a first place —")
    print("  that is arithmetic, not evidence. Note that the winner even FLIPS depending")
    print("  on which held-out split you rank on. Choose on what actually differs:")
    print("  training cost, dependency weight, and who has to maintain it.")

print()
print("OK")
