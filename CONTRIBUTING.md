# Contributing

Thank you for testing Metamorpher. The most useful contributions currently are
counterexamples, adversarial tests, reproducible evaluations, CUDA/Triton parity
results, and small changes that preserve the separation between relevance,
admissibility, value, evidence, and revision.

## Before opening a pull request

1. Describe the failure or decision problem the change addresses.
2. Add a regression test that fails without the change.
3. Keep simulator truth and counterfactual labels outside runtime modules.
4. Preserve `UNKNOWN != ABSENT` and keep value scoring behind hard masks.
5. Run:

   ```bash
   python -m unittest discover -s tests -v
   python -m compileall -q src examples benchmarks
   ```

6. If changing a numerical backend, compare selected action IDs and frontier
   masks with the dependency-free reference backend.
7. Document limits and target environment; do not generalize one synthetic
   result into a safety claim.

By submitting a contribution, you represent that you have the right to submit
it and agree that accepted contributions are licensed under Apache-2.0. No
separate CLA is required.
