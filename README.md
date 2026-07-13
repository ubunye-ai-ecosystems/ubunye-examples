# Ubunye Engine — Examples

Worked, deployable pipelines for [Ubunye Engine](https://github.com/ubunye-ai-ecosystems/ubunye_engine).
Every one of them runs on Databricks, deploys as a Databricks Asset Bundle, and
reads data that already exists in your workspace — nothing to upload, no API keys,
no cloud storage to configure.

They install the engine from PyPI (`ubunye-engine==0.3.0`), so what you run here is
what you get from `pip install` — not some unreleased branch.

**Every example in this repo has been run on real Databricks.** Not validated,
not reviewed — run, with the output inspected. Three of the bugs that fixing
found are described in the commit history, because none of them could have been
caught by reading the code.

---

## The abstraction

Every task is the same two files, whatever it does:

```
<usecase>/<package>/<task>/
    config.yaml          where the data comes from, where it goes, Spark settings
    transformations.py   the business rule
```

That is the whole model. **ETL or ML is not a distinction the engine makes.**
Training a model is a task. Scoring with one is a task. Aggregating a table is a
task. They differ in what `transformations.py` says, not in shape.

---

## The examples

**Ingestion**

| | Example | Reads | Shows |
|---|---|---|---|
| **01** | [Structured — tables + SQL](examples/01_ingest_tables_sql) | `samples.bakehouse` | Reading a table *and* pushing a join down as SQL; writing with `merge` (safe to re-run) and `overwrite_partitions` (safe to backfill) |
| **02** | [Structured — REST API](examples/02_ingest_rest_api) | Open-Meteo public API | The `rest_api` connector: query params, rate limiting, retries — and fanning one JSON document out into 168 rows |
| **03** | [Unstructured — text + files](examples/03_ingest_unstructured) | 204 real customer reviews, plus `.txt` files on a volume | Spark's `binaryFile` source; chunking text into overlapping windows for embedding |

**ML and MLOps**

| | Example | Reads | Shows |
|---|---|---|---|
| **04** | [ML — the full lifecycle](examples/04_ml_taxi_fare) | `samples.nyctaxi.trips` | Features → train → validate → gate → register → score. **Produce a model**, then **use it** |
| **05** | [RAG](examples/05_rag_documents) | the document chunks from 03 | Embeddings and a chat model on the workspace's own serving endpoints; retrieval, then a grounded answer |
| **06** | [Fine-tuning an open LLM](examples/06_finetune_llm) | `samples.bakehouse` reviews | An LLM labels the data, a small DistilBERT **learns from it** and is gated on `recall_negative` — distillation, end to end |
| **07** | [Data quality — a contract, enforced](examples/07_data_quality) | `samples.bakehouse` | Rules with severities; bad rows **quarantined, not dropped**; the run fails when the breach is structural |
| **08** | [Model monitoring & rollback](examples/08_model_monitoring) | the model and features from 04 | Drift, decay, and champion-vs-challengers — and **why a rollback is the right answer to only one of them** |

### The ML example is two tasks, not five

| Task | Does |
|---|---|
| `feature_engineering` | clean → features → deterministic train/test/score split |
| `model_training` | train → **validate on data it never saw** → gate → register → promote |
| `batch_inference` | load whatever model is in **production** → score new instances |

The split is *"produce a model"* vs *"use a model."* Validation lives with training
because it is how you decide whether the artifact is fit to register at all — and a
model that fails the gate **is never registered**, so inference cannot pick up
something nobody vetted.

Shipping a better model is a **promotion in the registry**, not a code change to the
inference task.

### And a gate is not monitoring

A gate stops a bad model being **born**. It cannot help you afterwards, and neither can
MLflow — a logbook of what happened at training time, which never looks at the model
again and cannot change what is serving. Example **08** is the other half: it catches
the model that got past the gate, *and* the one that was fine until the world moved —
and it tells the two apart, because **a rollback only fixes one of them**.

---

## Run it

### On a free Databricks workspace, in about five minutes

1. Sign up for [Databricks Free Edition](https://www.databricks.com/learn/free-edition) — no credit card.
2. In the workspace: **Workspace → Create → Git folder**, and paste this repo's URL.
   (It is public; you need no token.)
3. Open any notebook under `examples/*/notebooks/` and press **Run all**.

That is it. The notebook installs the engine, creates the schema and volumes it
needs, and runs the task. Nothing is downloaded from the internet — serverless
compute cannot be relied on to reach it, and an example that needs egress is an
example most people cannot run.

### As a bundle

```bash
databricks bundle validate --target dev
databricks bundle deploy   --target dev
databricks bundle run ml_taxi_fare --target dev
```

Everything here is serverless: there is no `new_cluster` block in `databricks.yml`,
because a free workspace has no other kind of compute.

---

## Where things get written

Defaults to `workspace.ubunye_examples`, overridable per-run:

```bash
databricks bundle deploy --target dev --var catalog=main --var schema=my_schema
```

| | |
|---|---|
| Tables | `<catalog>.<schema>.*` |
| Model registry | `/Volumes/<catalog>/<schema>/model_store` |
| Document corpus | `/Volumes/<catalog>/<schema>/corpus` |

The model registry lives on a **Unity Catalog volume** because `/tmp` is not
writable from serverless compute — the executors cannot see the driver's disk.

---

## Tests

```bash
pip install "ubunye-engine[spark,ml]==0.3.0" pytest pandas scikit-learn
pytest examples/*/tests
```

They run in seconds with no cluster and no workspace, because the parts worth
testing — the chunker, the model, the quality gate — are ordinary Python. The one
test that matters most asserts that a model trained on **noise fails the gate**: if
it passed, the gate would be decoration and the registry would fill up with junk.

---

## Licence

Apache 2.0.
