# Hidden-dimension expansion experiment

This synthetic mechanism test asks whether evidence-gated expansion can retain
a valid circular parent representation while adding axial displacement when
future observations come from a helix.

## Protocol

- Parent observations contain phase coordinates and a phase-dependent target.
- Helix worlds add target displacement per completed traversal.
- The fixed baseline retains only circular phase coordinates.
- The always-expand baseline receives traversal count immediately.
- The gated learner proposes traversal count only when parent residuals covary
  with it, then promotes the expansion only if it improves held-out cycles.
- A final segment returns to the parent regime. Scoped routing uses the retained
  parent; destructive replacement continues using the helix model.

The experiment supplies traversal count as a candidate coordinate. It tests the
control and preservation semantics of expansion, not autonomous discovery of a
never-represented variable.

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
| Parent-return MSE, scoped projection | 0.00262 |
| Parent-return MSE, destructive replacement | 22.4167 |

Under this configuration, gating recovers the supplied missing coordinate,
matches the always-expanded predictor in helix worlds, does not expand circle
worlds, and preserves the parent predictor for the return segment.

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
