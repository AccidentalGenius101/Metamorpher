# Competence-boundary experiment

This benchmark asks whether an online learner can recognize a specific limit:
the same observed inputs have accumulated incompatible labels, and no available
coordinate can identify which rule applies. The appropriate response is to
preserve the contradiction and abstain, not to claim that replay resolved it.

## Deliberately unidentifiable regime

The stream first trains the ring, XOR, and wave regions from the continual
replay benchmark. It then presents a collision regime whose feature vectors
come from the same distribution as XOR but whose labels are exactly inverted:

```text
ring → XOR → wave → inverted XOR → ring → XOR → wave
```

The tested learner receives only the two raw coordinates. It receives no task
ID, phase marker, regime coordinate, or task-specific prediction head. Thus a
single-valued classifier cannot be correct on both XOR regimes for identical
inputs. This is an intentional identifiability test, not conventional concept
drift with a recoverable latest rule.

## Compared policies

- **Always replay** applies loss-prioritized replay throughout and always emits
  a prediction.
- **Boundary-aware replay** audits nearby memory before updating. A boundary
  requires nearby contradictory labels in two consecutive batches. Gradient
  disagreement plus measured retained-loss damage supports the declaration;
  overwhelming local label contradiction can also qualify it directly. The
  candidate probe is rolled back. Once declared, the learner abstains for the
  learned input neighborhood and performs no further updates there.
- **Privileged oracle** has a separate model per generating regime and is
  routed with the hidden regime label. It is an upper bound with extra context
  and capacity, not a tested no-task-ID method.

Run:

```bash
python -m pip install -e ".[numpy]"
python experiments/competence_boundary.py --seeds 100
```

Default 100-seed result:

| Method | Detection rate | False declarations | Mean detection delay | Coverage | Selective accuracy | Resolved-region accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Always replay | 0% | 0 | — | 1.00 | 0.6142 | 0.6818 |
| Boundary-aware replay | 88% | 0 | 3.17 batches | 0.78 | 0.6840 | 0.6927 |
| Privileged oracle | — | — | — | 1.00 | 0.6928 | 0.7261 |

The boundary method performs zero updates inside a region after declaring it,
but it spends about 8,087 forward-audit examples per run. It misses the
collision in 12% of seeds; those misses remain ordinary forced predictions and
must not be counted as successful abstentions.

The report exposes detection rate and delay, false declarations, post-detection
harmful updates, replay and audit volume, coverage, and selective accuracy.
`collision_accuracy` is `null` for the boundary method when every run detects;
per-seed detected runs use `NaN` because they intentionally emit no collision
prediction. If only some seeds detect, the aggregate collision accuracy covers
the non-detecting runs only.

## What the result can establish

A successful run establishes that this particular persistent local-
contradiction rule can sometimes identify a constructed observational
collision, roll back the triggering probe, and preserve abstention afterward.
It can compare the retained accuracy and computation cost of that behavior with
always replaying.

It does **not** establish autonomous task discovery, universal out-of-
distribution detection, calibrated uncertainty, or recovery of the missing
context. The learned abstention neighborhood is a coarse batch-centroid radius,
and the reported coverage treats the four evaluation regimes equally rather
than estimating deployment prevalence. Detector thresholds were chosen for
this synthetic generator and need held-out calibration before any broader
claim.

Most importantly, the oracle's result must never be attributed to the tested
learner: the oracle is explicitly supplied the hidden regime identity. A future
expansion experiment may propose and validate a candidate context coordinate,
but this benchmark stops at a justified boundary.
