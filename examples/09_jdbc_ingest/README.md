# 09 · JDBC — ingest from a real relational database

> ## ⚠️ The one example that does not run on Databricks Free Edition
>
> It needs a **classic cluster**. Serverless ships **no JDBC drivers** — not for
> Postgres, not for MySQL, not for anything — and it **cannot install a JAR**, so you
> cannot add one. A free workspace has no classic compute at all; asking the API for a
> cluster answers `does not have any associated worker environments`.
>
> So this example has **its own bundle**, with its own cluster, and needs a **paid
> workspace**. That separation is not tidiness — it is load-bearing. On a
> serverless-only workspace this bundle fails at *job-creation* time
> (`Only serverless compute is supported in the workspace`), so folding it into the
> root bundle would break `databricks bundle deploy` and take the other **eight**
> working examples down with it.
>
> ### What is verified, and what is not
>
> Every other example in this repo has been **run on real Databricks** with its output
> inspected. This one has not, because no classic cluster was available to run it on.
> Saying otherwise would make the claim on the front page worthless.
>
> What *has* been run, end-to-end against the live database on local Spark:
>
> - the `config.yaml`, parsed by the engine, with every value coming from its
>   `default(...)` and no environment variables set
> - the **`jdbc` reader** — both the plain table read and the pushed-down SQL
> - the **partitioned read**: 200,000 rows came back in **4 partitions**, not 1
> - `transformations.py`, producing both outputs with keys matching the config
>
> **56** contributing databases, **200,000** sequences. What has *not* been executed is
> this notebook on a Databricks classic cluster: the cluster spec and the Maven library
> attachment are the unverified parts.

## The lesson

**A JDBC read is single-threaded unless you tell it not to be.**

Spark issues **one** query on **one** connection, and a single core drags the entire
table through it — however big your cluster is. There is no warning and no error. The
job is simply slow forever, and the Spark UI shows one task at 100% while the rest of
the cluster sits idle. This is why people conclude "JDBC is slow". JDBC is not slow.
Asking for 54 million rows down one socket is slow.

```yaml
partitionColumn: "id"
lowerBound: 1
upperBound: 200000
numPartitions: 4
```

Spark rewrites that single query into four, and runs them in parallel:

```sql
... WHERE id <  50001                    -- unbounded below
... WHERE id >= 50001  AND id < 100001
... WHERE id >= 100001 AND id < 150001
... WHERE id >= 150001                   -- unbounded above
```

Two things people get wrong about those bounds:

- **They are not a filter.** Rows outside `[lowerBound, upperBound]` are still read —
  the first and last partitions are open-ended. The bounds say *how to slice*, not
  *what to keep*. To filter, filter — in the `sql:`, so Postgres does it.
- **`numPartitions` is a hint.** If `partitionColumn` is missing, or is not numeric, or
  the bounds are nonsense, Spark **silently ignores all of it** and gives you back the
  single-threaded read you were trying to avoid. So the task checks
  `rdd.getNumPartitions()` and the notebook asserts it is 4. On a 54M-row table that
  check is the difference between minutes and hours, and it is invisible unless
  somebody looks.

And `fetchsize`: the Postgres JDBC driver's default is to pull the **entire result set
into the driver's memory** before returning a single row. That is how a slow query
becomes an OOM.

## The data

[RNAcentral](https://rnacentral.org)'s public PostgreSQL mirror, run by the EMBL-EBI:
**54 million RNA sequences**, with read-only credentials published in their own docs
for anyone to use. That is why the password is in `config.yaml` rather than in a
secret — it is public, and pretending otherwise would teach a worse habit than being
plain about it.

We pull a **bounded slice**. It is somebody else's research database and we are guests
on it.

| Input | Read as | Why |
|---|---|---|
| `rnacen.rnc_database` | plain `table:` read | ~56 rows. Slicing a small dimension into 4 parallel queries costs more than it saves |
| `rnacen.rna` | `sql:` + partitioned | 54M rows. Projection and `WHERE` pushed down to Postgres; the read split across 4 connections |

## Run it

You need a workspace with **classic compute**.

```bash
cd examples/09_jdbc_ingest
databricks bundle deploy --target dev
databricks bundle run jdbc_ingest --target dev
```

The node type is cloud-specific and defaults to AWS:

```bash
databricks bundle deploy --target dev --var node_type_id=Standard_DS3_v2   # Azure
```

The driver comes from Maven at cluster start:

```yaml
libraries:
  - maven:
      coordinates: org.postgresql:postgresql:42.7.4
```

That single line is the whole reason this example cannot be serverless. The notebook
checks the class is on the classpath **before** it runs anything, because otherwise the
failure is a `ClassNotFoundException` thrown from inside a Spark task — which reads
like a Spark bug and is not one.

## What it writes

| Table | Mode | |
|---|---|---|
| `rna_databases` | `merge` on `id` | the contributing databases — safe to re-run |
| `rna_sequence_profile` | `overwrite` | the length distribution of the slice, and the partition count the read actually achieved |
