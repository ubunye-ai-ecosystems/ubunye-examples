# 10 · scikit-learn vs PyTorch vs TensorFlow — the same model, three times

One dataset. One feature set. One split. One quality gate. One registry. One output
table.

**Three frameworks — and three `config.yaml` files that are byte-for-byte identical.**

```
pipelines/fare_bench/ml/
    train_sklearn/   config.yaml   transformations.py   ← 4 lines of logic
    train_torch/     config.yaml   transformations.py   ← 4 lines of logic
    train_keras/     config.yaml   transformations.py   ← 4 lines of logic
    select_champion/ config.yaml   transformations.py
```

Each `transformations.py` names a model class and hands it to the same training run:

```python
class TrainKeras(Task):
    def transform(self, sources):
        return benchmark.run(KerasFareModel(), sources)
```

The engine has no idea TensorFlow is involved. **ETL or ML, tree or neural net, is not
a distinction the engine makes** — it is a distinction inside `model.py`. That is the
entire argument of this example, and the three identical configs are the proof.

---

## What actually happened

Ranked on the **`score`** split — rows no model was fitted on and **no gate was judged
against**. Ranking on `test` would be picking the winner on the exam it was allowed to
resit.

| rank | framework | score_rmse | test_rmse | test_r2 | fit |
|---|---|---|---|---|---|
| 1 | **pytorch** | **1.3011** | 1.8512 | 0.9663 | 11.4s |
| 2 | tensorflow | 1.3072 | 1.8617 | 0.9659 | 18.0s |
| 3 | scikit-learn | 1.3791 | **1.7676** | **0.9693** | 18.3s |

**Verdict: `too close to call` — the winner is 0.5% ahead.**

Read that table properly, because it says three things a leaderboard usually hides:

**1. The winner flips depending on which split you rank on.** scikit-learn has the
*best* `test_rmse` and the *best* `test_r2`. PyTorch has the best `score_rmse`. Pick a
different held-out split and you crown a different framework. If a benchmark's result
depends on which of two equally valid splits you chose, the benchmark has not found a
difference — it has found noise, and dressed it up.

**2. The gap between the frameworks is smaller than the gap between the splits.** Every
model is ~0.5 RMSE worse on `test` than on `score` — all three, by almost the same
margin. That is a property of the **data** (the hashed split happened to put harder
rows in `test`), not of the models. When the split matters more than the framework, the
framework is not what you should be arguing about.

**3. So the leaderboard says so, out loud.** `select_champion` computes the margin to
the runner-up and refuses to call a 0.5% lead a win. Sorting three numbers always
produces a first place; that is arithmetic, not evidence. The task emits
`verdict: too close to call`, and the honest recommendation is to choose on what
genuinely differs: **training cost, dependency weight, and who has to maintain it** —
by which measure the boring one wins, because it has no scaler to save, no threads to
cap, and no numpy pin.

---

## What the environment cost us

Every one of these was **measured on the workspace**, and every one is silent when you
get it wrong. This is why the example exists in a repo rather than a blog post.

### TensorFlow will not install the way you think

| | |
|---|---|
| `pip install tensorflow-cpu` | ❌ **No matching distribution found** |
| `pip install tensorflow` | ✅ …and it breaks your environment |
| `pip install tensorflow "numpy<2" "protobuf<5"` | ✅ correct |

Serverless is **`aarch64`**. `tensorflow-cpu` publishes **no ARM wheels at all**.

Plain `tensorflow` installs, pulls **numpy 2** and **protobuf 7**, and quietly breaks
`scipy`, `scikit-learn` **and MLflow** in the same interpreter — while pip prints
*Successfully installed* and exits `0`:

```
databricks-connect requires numpy<2      → you have numpy 2.4.6
scipy 1.11.1      requires numpy<1.28    → you have numpy 2.4.6   ← breaks sklearn
mlflow-skinny     requires protobuf<5    → you have protobuf 7.35 ← breaks MLflow
```

### TensorFlow OOMs the driver unless you cap its threads

TF's default `intra_op` parallelism is *one thread per core*, and each thread gets its
own allocation arena. On a many-core serverless box with a 16 GB driver, a **6-feature
MLP on 15,000 rows** killed the job with *Execution ran out of memory*. It reads like
the data was too big. It was not.

```python
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
```

Capped: **6.8 seconds, flat 2.3 GB.**

### You cannot save a Keras or PyTorch model onto a UC volume

```
OSError: [Errno 95] Operation not supported
```

A `.keras` file — and `torch.save`'s default container — is a **ZIP archive**, and
writing a zip requires **seeking**. A Unity Catalog volume is a FUSE mount that does
not support it. This matters because the engine's `ModelRegistry` hands `save()` a path
*on the volume*.

Write to local disk, then copy the finished file across. Reading it back off the volume
is fine — it is only the write.

```python
with tempfile.TemporaryDirectory() as tmp:
    local = os.path.join(tmp, "model.keras")
    self._model.save(local)
    shutil.copyfile(local, str(target / "model.keras"))
```

scikit-learn is unaffected: `joblib` writes a plain sequential stream.

### The three frameworks cannot share one interpreter

TensorFlow needs `numpy<2`. That is not a preference you can negotiate with. So
`select_champion` **loads no models at all** — it ranks numbers the three trainers
already wrote, and installs neither torch nor tensorflow nor sklearn.

That is not a dodge; it is the architecture. **The task is the unit of isolation.**
Three trainers, three environments, three `%pip install` lines — and a comparison step
that needs none of them.

---

## The ML details that are not framework trivia

**The neural nets standardise their inputs; the tree does not.** `pickup_zip` runs to
tens of thousands, `pickup_hour` runs 0–23. A tree splits on thresholds and does not
care. A net multiplies every input by a weight, so unscaled, the zip code drowns
everything, the gradients explode, and the loss becomes `nan`.

**The fitted scaler is saved *with* the weights.** It is part of the model, not a note
about it. Restore the weights without it and inference runs on unscaled inputs — no
error, just confidently wrong numbers. `tests/test_fare_common.py` asserts that
refitting a scaler on live data hides a real shift, which is the classic silent version
of this bug.

**All the metrics are computed in pure numpy, once, identically.** If the tree measured
its RMSE with sklearn and the nets used their framework's internal metric, the
leaderboard would be comparing two different numbers and calling one of them a winner.

---

## Run it

Needs the feature table from example **04** — run `ml_taxi_fare` first.

```bash
databricks bundle deploy --target dev
databricks bundle run ml_frameworks --target dev
```

Four tasks: the three trainers run **in parallel** (they append to one Delta table, and
concurrent appends do not conflict — three *overwrites* would), then `select_champion`
waits for all three.

| Table | |
|---|---|
| `fare_bench_metrics` | every metric of every framework, appended forever |
| `fare_bench_leaderboard` | who is winning right now, and whether the lead means anything |
