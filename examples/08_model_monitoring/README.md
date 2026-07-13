# 08 · Model monitoring — and the rollback that isn't always the answer

> **Why not just use MLflow?**
>
> MLflow is a **logbook**. It records what happened at *training* time — params,
> metrics on a test set, which version was promoted — and then it never looks at the
> model again. It will keep reporting `test_r2 = 0.969` while the model under-predicts
> every fare in production, and it cannot change what is serving.
>
> A quality gate (example 04) stops a bad model being **born**. Monitoring catches the
> one that got past, and the one that was fine until the world moved.

This example runs **one** monitoring task against **two different failures**, because
they have two different remedies and telling them apart is the entire job.

## What actually happened when this ran

These are the numbers from the deployed run, not an illustration:

| | input drift | target drift | decay | is production still the best model I have? | action |
|---|---|---|---|---|---|
| **A · the world moved** — fares inflate 35% | no · PSI ≤ 0.005 | **yes · PSI 0.349** | **yes · MAE 0.446 → 4.33** | yes, nothing beats it (best challenger also 4.33) | **retrain** |
| **B · a bad model was promoted** | no · PSI ≤ 0.005 | no · PSI 0.001 | **no** · MAE 5.99 vs its own 6.15 baseline | **no** — five versions score 0.43 on the same rows | **roll back** |

**Read the zeros. They are the point.**

In **A**, the inputs do not move *at all*. Inflation changes what a trip **costs**, not
how far it **goes**. So an input-drift dashboard — which is what most "drift monitoring"
in practice actually is — shows **all green** while the model is wrong on every row.

In **B**, *nothing* drifts and *nothing* decays. The broken model's live error (5.99)
matches its own honest baseline (6.15), because it did not rot — **it was born broken**.
A decay-only monitor sees a model performing exactly as advertised and serves it forever.

Each check catches a failure the other two cannot. That is why there are three.

## The part almost nobody builds

> **Is the live model still the best model I have?**

The monitor loads **every registered version** and scores them all on **the same live
window** — not on their own test sets, which describe the world they were trained in.
The only way to know whether an old version would do better *now* is to run it *now*.

```
model_version  role         live_mae   best_challenger
1.0.1          challenger      0.432    true
1.0.0          challenger      0.432    false
1.0.4          challenger      0.440    false
1.0.5          production      5.991    false     <- what is serving
```

That table is why the rollback in scenario B is a **decision** and not a reflex.

**A rollback only ever fixes one thing: a model that is worse than one you already
had.** It cannot fix a changed world — every version in the registry learned the *old*
world, so they are all equally wrong, and rolling between them is theatre with a
changelog. In scenario A the monitor measures exactly that (best challenger: 4.33,
production: 4.33), **refuses to roll back**, and says `retrain` — with the number that
proves a rollback would have been pointless.

Then it acts, which is the step MLflow cannot take for you: it does not know what is
serving, and it cannot change it. The registry does, and did.

## How the failures are built

Nothing here is faked, and both failures are injected **on purpose** — a monitor whose
alarm never fires is a smoke detector with no battery: it passes every test, reassures
everybody, and is wired to nothing.

- **A** — `DRIFT_FACTOR` (default `1.35`) multiplies the true fares in the live window.
  Set it to `1.0` for the undrifted world. Distances and durations are untouched.
- **B** — the notebook trains a model on 200 rows with a depth-1 stump, registers it
  with its **real, honest, bad** metrics, and promotes it with **no gate** — which is
  what "somebody promoted it by hand" means in practice. This is the canonical rollback
  case, built the way it really happens.

## Run it

```bash
databricks bundle deploy --target dev
databricks bundle run model_monitoring --target dev
```

It monitors example **04**'s model, so run `ml_taxi_fare` first — it needs the
`taxi_features` table and a registered model. It imports 04's model class rather than
copying it: two copies of a model class is how a registry starts loading an artifact
into a class that no longer matches it — the joblib still deserialises, the columns
still line up, and the predictions quietly mean something else.

Each run leaves one deliberately-broken version in the registry (archived), and starts
by promoting the healthiest recorded version, so it is safe to run repeatedly and always
leaves a **healthy production model** behind. It has to: example 04's inference task
loads whatever is in production on its next run. Rolling back to *nothing* is not a fix,
it is an outage.

## What it writes

| Table | |
|---|---|
| `model_monitoring` | PSI per feature per run, tagged `input` or `target`, plus the live error |
| `model_candidates` | every registry version scored on today's data — the evidence behind the decision |
| `monitoring_incidents` | the diagnosis and what was done about it |

## Thresholds

| | | |
|---|---|---|
| `PSI_WARN` | `0.10` | the world has moved; watch it |
| `PSI_BREACH` | `0.25` | it has moved enough that the model is answering a different question |
| `MAX_ERROR_MULTIPLE` | `3.0` | twice its test error is noise; three times is a different model |
| `MIN_IMPROVEMENT` | `0.10` | how much better a challenger must be to justify churning production — a rollback is a deployment, and 2% better is noise wearing a rosette |

PSI is an industry rule of thumb, and it is *only* a rule of thumb. The numbers that
matter are in `tests/test_psi.py`, which proves the calculation cannot go **silent** —
including the bucketing bug where live values outside the training range are dropped,
the survivors renormalise back to the reference shape, and PSI reports a reassuring
`0.00` about the most alarming rows in the batch.
