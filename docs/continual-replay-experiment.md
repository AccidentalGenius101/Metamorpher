# Continual replay experiment

This mechanism benchmark trains a small NumPy multilayer perceptron on a
recurring nonlinear stream and compares no replay, reservoir-random replay, and
loss-prioritized replay. An evidence-gated method additionally treats each
current-only update as a candidate and measures its effect on retained evidence
before deciding whether replay is warranted.

## Question

Can a bounded replay memory preserve earlier competence during sequential
online updates when the task identity is not supplied at inference?

The benchmark is domain-incremental rather than task-incremental. Three input
regions carry ring, XOR, and sinusoidal-boundary rules. Their schedule is:

```text
ring → XOR → wave → ring → XOR → wave
```

Input location makes the active region observable from the two raw coordinates.
There is no task-ID feature or task-specific prediction head. Consequently, the
benchmark does not test contradictory labels for indistinguishable inputs or
autonomous discovery of an unobserved context.

## Controlled comparison

All methods receive identical generated streams, evaluation sets, model
initialization, optimizer, and model capacity. The three fixed policies use the
same batch size and number of gradient updates. Each fixed-policy update
contains 16 current examples plus 16 examples allocated as follows:

- **No replay:** repeat current examples to hold training compute constant.
- **Random replay:** sample uniformly from a 96-example reservoir.
- **Prioritized replay:** sample from the same reservoir in proportion to the
  square root of each example's last measured loss.
- **Evidence-gated replay:** apply one candidate current-only gradient step,
  audit a uniform held-out memory sample, and continue current-only training if
  retained loss remains stable. If relative retained loss rises by more than
  1%, roll back the candidate and train the batch with prioritized replay.

Both replay methods use identical reservoir replacement. This isolates sampling
policy from memory membership. Priorities are deliberately simple and may be
stale until an example is sampled again.

## Default result

Command:

```bash
python -m pip install -e ".[numpy]"
python experiments/continual_replay.py --seeds 100
```

Configuration: 100 seeds, 640 stream examples per phase, 800 held-out examples
per region, one hidden layer with 24 units, 96-example memory, and no task ID.

| Method | Final average accuracy | Final worst-region accuracy | Mean forgetting |
|---|---:|---:|---:|
| No replay | 0.5953 | 0.4070 | 0.1852 |
| Random replay | 0.7200 | 0.6192 | 0.0601 |
| Prioritized replay | 0.7312 | 0.6321 | 0.0508 |

Forgetting is computed per region as its best held-out accuracy after the first
phase that trains that region minus its final accuracy, then averaged over the
three regions.

Under this configuration, both bounded replay policies improve final retention
over current-only online SGD. Prioritized replay is modestly better on the
reported seeds at the tight memory budget. The tests require replay to improve
retention; they do not require prioritized replay to beat random replay.

## Evidence-gated result

The gated method does not assume that replay is always valuable. It spends one
probe gradient on a candidate update, uses an audit sample only to measure
retained loss, and rolls the candidate back before replay when the damage gate
fires. Audit examples are never used as training examples unless separately
sampled through the replay policy.

The benchmark also reports replay examples, audit examples, total gradient
updates, and rejected candidate batches. This exposes the complete trade-off:
gating can reduce replay bandwidth while spending additional forward-only audit
passes and probe gradients. It should not be described as a free compute
reduction.

Across 100 seeds, the default gate produced:

| Method | Final average accuracy | Mean forgetting | Replay examples | Audit examples | Gradient updates |
|---|---:|---:|---:|---:|---:|
| Always prioritized | 0.7312 | 0.0508 | 3,824 | 0 | 720 |
| Evidence-gated prioritized | 0.7269 | 0.0631 | 1,842 | 22,944 | 835 |

The gate therefore retains nearly the final accuracy of always-prioritized
replay while using about 52% fewer replay training examples. It pays for that
reduction with many cheap forward audit evaluations and about 16% more gradient
updates. Whether this is preferable depends on the relative cost of memory
retrieval, forward evaluation, and backpropagation in the target system.

## Scope and next boundary

This establishes a real neural online-learning baseline, not the full
Metamorpher continual-learning architecture. It does not yet include:

- evidence-gated model updates;
- competence-boundary detection;
- learned context or task discovery;
- offline consolidation;
- capsule transfer;
- adaptive memory allocation;
- integration with the canonical logical knowledge state.

The next integration experiment should expose prediction failures as evidence,
let the controller decide whether to continue, replay, abstain, or propose a
new context, and measure whether that governance improves the accuracy–cost–
forgetting trade-off.
