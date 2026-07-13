# 11 · Define once, run anywhere — and prove it

One `config.yaml`. One `transformations.py`. Run **unchanged** on local Spark, Docker,
Kubernetes and Databricks — and a CI job asserts all of them produced the **same
output hash**.

That last part is the only part that matters. A badge saying *runs on 4 platforms* is
decoration. Without an assertion that the outputs **agree**, "portable" quietly decays
into "it didn't crash on any of them", which is a far weaker claim — and it is the
version of this test that almost everyone ships.

These are the numbers from the actual runs, not an illustration:

| platform | fingerprint |
|---|---|
| local Spark | `067da98ea04a7d2f08969aa8ee614ac69fe6f659176145767d4e8fa5e296292d` |
| Docker | `067da98ea04a7d2f08969aa8ee614ac69fe6f659176145767d4e8fa5e296292d` |
| Kubernetes | `067da98ea04a7d2f08969aa8ee614ac69fe6f659176145767d4e8fa5e296292d` |
| Databricks | `067da98ea04a7d2f08969aa8ee614ac69fe6f659176145767d4e8fa5e296292d` |

Open-source Spark and the Databricks Runtime, a laptop and a Kubernetes pod, a Delta
path and a Unity Catalog table — **byte-for-byte the same output**.

## The bug this caught on its first run

Three platforms agreed. Databricks came back with a completely different hash:

```
local / docker / k8s :  067da98e…
databricks           :  46d06fcb…
```

The pipeline was fine. **The data was not.**

CI checks the repo out on Linux and gets **LF**. A Databricks bundle uploads files from
the developer's **working tree** — and on Windows, git's `autocrlf` had silently
rewritten the corpus to **CRLF**. Same six files, seven extra bytes each, a different
`sha2(text)`, a different `doc_id`, a different hash. Nothing errored. One platform
simply ingested different bytes, quietly, and would have gone on doing so forever.

`.gitattributes` now pins LF everywhere, and the hashes agree.

That is the whole argument for asserting on *output* rather than on exit codes. **"It
didn't crash on four platforms" would have passed this.** A green tick would have been
sitting on top of a pipeline that was demonstrably reading different data on one of
them.

## The entire portability surface is three environment variables

| | what it decides | values |
|---|---|---|
| `SPARK_MASTER` | where the compute is | `local[*]` · `k8s://…` · `yarn` |
| `UBUNYE_SINK` | what kind of table to write | `s3` (a path) · `unity` (a table) |
| `UBUNYE_DATA_ROOT` | where the data lives | `file://` · `s3a://` · `gs://` · `/Volumes/…` |

Nothing else changes. Not the config, not the code, not the SQL.

```yaml
outputs:
  documents:
    format: "{{ env.UBUNYE_SINK | default('s3') }}"
    path:  "{{ env.UBUNYE_DATA_ROOT }}/documents"          # used when sink = s3
    table: "{{ env.UBUNYE_CATALOG }}.…​.portable_documents" # used when sink = unity
    mode: merge
    merge_keys: ["doc_id"]
    file_format: delta
```

Three facts make that work, and none of them required an engine change:

1. **`format:` is Jinja-rendered before it is validated.** It is an ordinary string
   scalar, so `{{ env.UBUNYE_SINK }}` resolves to a real connector name.
2. **`path` and `table` can both be set at once.** Config validation only enforces the
   keys the *resolved* format needs, and each writer reads only its own — the `s3`
   writer never looks at `table`, the `unity` writer never looks at `path`. So one
   output block serves both worlds, and the env var picks which is real.
3. **The `s3` connector is misnamed.** It has nothing to do with AWS: it is a generic
   *path* connector and takes `file://`, `s3a://`, `gs://`, `abfss://`. It is the
   portable workhorse of this whole repo.

## What you cannot do — and why that turned out to be good

There is no `{% if platform == "databricks" %}` in the config, and there **cannot
be**: the engine parses the YAML *first* and only then renders string values, so
structural conditionals are impossible. Only values can be templated, never shape.

That constraint is a gift. It makes the honest design the only available one — a
config with platform branches in it is three configs wearing a trenchcoat, and it will
drift apart the first time someone edits one branch and not the others.

## The runner is not the pipeline

`dbutils` appears in this example exactly once: in the Databricks **notebook**. It is
nowhere near the task, and that boundary is the whole trick.

> **The task is the portable unit. The runner is not, and was never meant to be.**

Every platform launches things its own way — a notebook, a shell script, an entrypoint,
a Kubernetes Job — and each stages the corpus its own way. **The bootstrap is allowed
to differ. The pipeline is not.** Pretending otherwise is what makes people believe
portability is impossible: they try to make the *launcher* portable, fail, and
conclude the pipeline can't be either.

## Why the output is deliberately boring

No `current_timestamp()`. No randomness. Document ids hashed from **content**, not from
the file path.

Every other example in this repo stamps its rows with an ingest time — the right thing
in a real pipeline, and exactly the wrong thing here. A wall-clock column would make
local, Docker, Kubernetes and Databricks differ on every single run, and the identity
assertion would become untestable: it would still pass, because each hash would only
ever be compared with itself.

Hashing the id from content also means `mode: merge` is idempotent **across platforms**,
not just across runs. The same document is the same document whether it arrived from
`file:///tmp`, `s3a://`, or a Unity Catalog volume.

## Run it

```bash
# local — nothing but Python, Java and Spark
platforms/local/run.sh

# docker
docker build -f platforms/docker/Dockerfile -t ubunye-portable .
docker run --rm ubunye-portable

# kubernetes
kind create cluster && kind load docker-image ubunye-portable
kubectl apply -f platforms/k8s/job.yaml && kubectl logs -f job/ubunye-portable

# databricks
databricks bundle deploy --target dev && databricks bundle run run_anywhere --target dev
```

Each prints a `FINGERPRINT=`. They must match. `.github/workflows/portability.yml`
checks that they do, on every pull request.

## What does *not* port, and why I am not pretending otherwise

| | |
|---|---|
| **05 · RAG**, **06 · fine-tuning** | Bolted to Databricks-hosted serving endpoints (`databricks-bge-large-en`, `llama-3.1-8b`). Without another model host these genuinely do not port. |
| **01, 04, 07, 08, 10** | Read `samples.nyctaxi` / `samples.bakehouse`, which exist only on Databricks. The *pipelines* are portable; the **data bootstrap** is not, and needs a per-platform seed step. |
| **Model registry** | `ModelRegistry` is `pathlib`-based, so it needs a POSIX-visible filesystem. It cannot write to an `s3://` URI — it needs a mount (a PVC, `s3fs`, or a UC volume). That limits where the ML examples can put their artifacts. |

Ingestion and transformation port cleanly today. ML ports as far as the registry's
filesystem assumption allows. Anything bound to a hosted model does not port at all,
and saying so is more useful than a matrix of green ticks that quietly omits it.
