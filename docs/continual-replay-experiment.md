# Continual replay experiment

This mechanism benchmark trains a small NumPy multilayer perceptron on a
recurring nonlinear stream and compares no replay, reservoir-random replay, and
loss-prioritized replay.

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
initialization, optimizer, batch size, number of gradient updates, and model
capacity. Each update contains 16 current examples plus 16 examples allocated
as follows:

- **No replay:** repeat current examples to hold training compute constant.
- **Random replay:** sample uniformly from a 96-example reservoir.
- **Prioritized replay:** sample from the same reservoir in proportion to the
  square root of each example's last measured loss.

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
