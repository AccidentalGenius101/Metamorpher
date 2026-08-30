# Decision-gated representational expansion

This synthetic mechanism test connects two existing Metamorpher ideas:
evidence-gated representational expansion and control equivalence across an
unresolved version space. It asks whether an evidence-supported distinction
must also be resolved at decision time.

These are separate judgments:

- **Representation warranted:** held-out evidence supports the richer model.
- **Control refinement warranted:** collapsing the surviving hidden states
  changes their common-safe actions.

The second does not follow automatically from the first.

## Protocol

Both worlds reuse the helix generator and supplied cycle coordinate from the
hidden-dimension experiment. In both, traversal count explains structured
parent residuals and must pass the same held-out predictive-improvement gate.
The future segment contains 64 decisions across cycles 8 through 11.

- **Descriptive-only world:** low- and high-cycle hypotheses both permit
  `coast`. The richer predictor is retained, but
  `UnresolvedCell.common_safe_actions()` permits control at the coarse level.
- **Control-relevant world:** `coast` is safe below cycle 10 and `stabilize` is
  safe at or above cycle 10. Their common-safe intersection is empty, so cycle
  must be resolved before acting.

Three policies are compared:

- **Never refine:** never measures cycle; where no common-safe action exists,
  it continues the represented pre-boundary action.
- **Always refine:** measures cycle before every decision.
- **Decision gated:** measures cycle only when the coordinate passed epistemic
  promotion and the promoted distinction changes the common-safe action set.
  A rejected coordinate cannot authorize its own use even when the supplied
  action table says it would have been useful.

Run:

```bash
python -m pip install -e ".[numpy]"
python experiments/decision_gated_expansion.py --seeds 100
```

## Default result

The supplied coordinate was epistemically promoted in 100% of both world
types. Across 100 seeds, fixed-parent future MSE was 11.2131 and expanded future
MSE was 0.00291 in both worlds.

| World and policy | Measurements / 64 | Action error | Unsafe disagreement | Regret |
|---|---:|---:|---:|---:|
| Descriptive, never refine | 0 | 0% | 0% | 0 |
| Descriptive, always refine | 64 | 0% | 0% | 0 |
| Descriptive, decision gated | 0 | 0% | 0% | 0 |
| Control-relevant, never refine | 0 | 50% | 50% | 32 |
| Control-relevant, always refine | 64 | 0% | 0% | 0 |
| Control-relevant, decision gated | 64 | 0% | 0% | 0 |

The result demonstrates the intended mechanism: a coordinate can improve
prediction by roughly four orders of magnitude while remaining unnecessary to
resolve for a particular action. When the same coordinate changes the
common-safe action, decision gating demands it and matches the fully refined
policy's control result.

## Scope boundary

Cycle index is still the exact generating coordinate supplied to residual
analysis, fitting, validation, and decision-time refinement. The two action
maps and their safe-action sets are also supplied by the synthetic generator.
The benchmark does not discover a coordinate, learn action consequences,
estimate measurement cost, calibrate safety, or integrate the expansion
registry automatically with the production controller.

The deterministic action table also makes the control result structurally
simple once the gates are defined. The evidence lies in the separation of the
two gates and reuse of the real common-safe-action intersection—not in a claim
of difficult policy learning. Future work should learn action-outcome models
from held-out interventions and trade refinement cost against expected regret.
