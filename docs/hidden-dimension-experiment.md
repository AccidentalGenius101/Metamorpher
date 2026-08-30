# Hidden-dimension expansion experiment

This synthetic mechanism test asks whether evidence gates can control adoption
of a supplied cycle coordinate while retaining a valid circular parent model
when future observations come from a helix.

## Protocol

- Parent observations contain phase coordinates and a phase-dependent target.
- Helix worlds add target displacement per completed traversal.
- The fixed baseline retains only circular phase coordinates.
- The always-expand baseline receives traversal count immediately.
- The gated learner proposes traversal count only when parent residuals covary
  with it, then promotes the expansion only if it improves held-out cycles.
- A final segment returns to the parent regime. An oracle regime label manually
  selects the retained parent; destructive replacement continues using the
  helix model.

Traversal count is the exact generating coordinate and is supplied to residual
analysis, expanded-model fitting, validation, and prediction. The experiment
therefore tests gated adoption of a supplied feature and registry bookkeeping
around an external NumPy learner. It does not test latent-coordinate discovery,
candidate selection, autonomous return routing, or end-to-end representational
expansion.

The follow-on
[`decision-gated-expansion-experiment.md`](decision-gated-expansion-experiment.md)
adds supplied action consequences and tests whether a promoted coordinate must
also be resolved for control.

## Default result

Command:

```bash
python experiments/hidden_dimension_expansion.py --seeds 100
```

Configuration: 100 circle worlds, 100 helix worlds, drift 0.35, Gaussian noise
standard deviation 0.05, 16 phase observations per traversal.

| Measure | Result |
|---|---:|
| Helix expansion detection | 100% |
| Circle false expansion | 0% |
| Helix future MSE, fixed parent | 11.2131 |
| Helix future MSE, gated expansion | 0.00291 |
| Helix future MSE, always expanded | 0.00291 |
| Circle future MSE, fixed/gated | 0.00261 |
| Parent-return MSE, oracle-selected retained parent | 0.00262 |
| Parent-return MSE, destructive replacement | 22.4167 |

Under this configuration, gating adopts the supplied coordinate, matches the
always-expanded predictor in helix worlds, does not expand circle worlds, and
retains a parent predictor that performs well when an oracle selects it for the
return segment.

## Sensitivity boundary

The result is not threshold-free. With the default residual-correlation and
held-out-improvement gates, 100-seed sweeps produced:

| Drift | Noise | Detection | False expansion |
|---:|---:|---:|---:|
| 0.03 | 0.05 | 95% | 0% |
| 0.03 | 0.10 | 0% | 0% |
| 0.05 | 0.10 | 47% | 0% |
| 0.10 | 0.10 | 100% | 0% |
| 0.10 | 0.20 | 47% | 0% |
| 0.20 | 0.20 | 100% | 0% |
| 0.35 | 0.40 | 100% | 0% |

The current detector is deliberately conservative: weak structure disappears
below its signal-to-noise boundary rather than causing false expansion. These
numbers are deterministic for the recorded seeds and implementation, but they
are synthetic benchmark results rather than a general learning guarantee.
