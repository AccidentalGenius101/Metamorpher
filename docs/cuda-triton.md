# CUDA and Triton acceleration

Metamorpher is CPU-first. The scalar Python implementation is the semantic
reference, is dependency-free, and is usually faster for one small graph. CUDA
and Triton become useful when evaluating many independent controller states or
many similarly shaped candidate graphs at once.

Acceleration changes throughput, not authority or controller semantics.

## Installation and detection

```bash
python -m pip install -e ".[cuda]"
metamorpher doctor
```

The optional extra installs NumPy, PyTorch, and Triton. A usable CUDA backend also
requires a compatible NVIDIA driver and device. Installation of the extra does
not guarantee that `torch.cuda.is_available()` is true.

Expected fallback order:

1. Triton kernels on CUDA, when available and supported for the batch.
2. Vectorized NumPy on CPU, when the optional NumPy dependency is available.
3. Dependency-free CPU reference implementation for every other environment.

A missing accelerator is normal and must not prevent importing or using the
core package.

For this source snapshot, reference/NumPy parity was exercised in the CPU
development environment. A compatible CUDA runtime was unavailable, so Triton
kernel execution remains untested here. Run the backend parity command in the
README on the target CUDA system before relying on that path.

## Batch representation

A practical accelerator boundary converts independent cases into padded or
bucketed tensors:

- node features: `[batch, nodes, features]`;
- action state and hard masks: `[batch, nodes]`;
- graph edges: edge lists or CSR-style source/target arrays;
- guard truth values: discrete satisfied/violated/unknown codes;
- value scores: floating point `[batch, nodes]`;
- selected actions: one index per case plus a validity/status code.

Bucket graphs by node and edge count to reduce padding. Keep stable action IDs and
an explicit mapping between tensor positions and domain objects so traces remain
readable.

## Kernel boundary

Good accelerator candidates are pure, repeated data-plane operations:

- evaluate represented prerequisites over a batch;
- construct the certified/refinement/blocked masks;
- compute heuristic or learned scores;
- mask non-frontier actions;
- perform a segmented argmax with deterministic tie-breaking.

Keep control-plane operations on CPU:

- evidence provenance and append-only logging;
- graph transactions and cycle validation;
- equivalence-class split/merge decisions;
- domain-memory mutation;
- authorization and external execution.

This keeps accelerator failure from silently mutating structural knowledge.

## Required semantics

All backends must preserve these rules:

1. Hard masking happens before value selection.
2. Unknown is distinct from both satisfied and violated.
3. Censored evidence does not become a negative value in a tensor.
4. NaN and infinite scores cannot win an argmax.
5. Ties use the same stable action-ID order as CPU.
6. An unsupported shape or kernel error is surfaced without a partial decision;
   the caller may then explicitly retry on the CPU reference backend.
7. Accelerator output is checked for shape, range, and status validity before use.

Use integer or boolean representations for hard structural states. Floating point
scores must never decide whether a prerequisite exists. `float32` is generally
appropriate for ranking; lower precision needs explicit parity evidence.

## Compilation and caching

Triton compiles specialized kernels for shapes and parameters. First-call latency
can dominate a small workload. Warm representative buckets before benchmarking,
bound the number of accepted shape specializations, and avoid treating compile
cache behavior as algorithmic speedup.

Do not include host-to-device transfer, graph packing, or compilation in one
benchmark but exclude it from another. Report both end-to-end latency and steady
state kernel throughput.

## Parity tests

At the raw backend boundary, compare the dependency-free reference, NumPy, and
Triton outputs on:

- empty and singleton frontiers;
- all-satisfied and all-blocked encoded hard prerequisites;
- equal scores and lexicographic tie-breaking;
- NaN/positive infinity/negative infinity scores;
- padded nodes and disconnected components;

Raw backends receive already-resolved booleans and numeric features. They do not
interpret guards, alternatives, mutex relations, soft epistemic edges, censored
evidence, or graph revisions. Test those semantics one layer higher by comparing
`GraphBatchCompiler` plus each backend against the symbolic controller on:

- satisfied, violated, and unknown guards;
- alternatives, mutex relations, and soft epistemic edges;
- graph and evidence revisions between compiled batches;
- censored and conflicting evidence summaries.

Frontier/status results should match exactly. Floating point scores may use a
documented tolerance, but selected action IDs should still match unless the input
scores are within that tolerance; such cases should be reported explicitly.

## When CPU is preferable

Prefer CPU when:

- evaluating one small or rapidly changing graph;
- graph packing costs exceed scoring work;
- detailed provenance dominates runtime;
- deterministic debugging or trace inspection is the goal;
- no compatible NVIDIA environment is present.

An accelerator is a scaling path, not a prerequisite for correctness and not a
substitute for a complete graph, useful observations, or independent audits.
