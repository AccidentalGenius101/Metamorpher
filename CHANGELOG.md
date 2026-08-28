# Changelog

## Unreleased

- Connected accumulated failure evidence to automatic, evidence-supported
  equivalence-class carving and version-space materialization.
- Recompute narrowed version spaces from their represented parent when novel
  contradictory evidence arrives; uninformative observations no longer count
  toward resolution support.
- Enforced hypothesis domain tags during controller decisions.
- Applied version-space common-safe masks in accelerated batch compilation.

## 0.1.0 — 2026-08-28

- Added typed, tri-state diagnostic-action graphs and certified frontiers.
- Added three-way model-relative decisions: supported, refinement required, or abstain.
- Added append-only evidence, censored observations, domain memory, audits, and unresolved version spaces.
- Added transactional graph revisions with provenance and stale-decision rejection.
- Added a dependency-free CPU reference backend, vectorized NumPy backend, and lazy CUDA/Triton kernels.
- Added deterministic Sierra and synthetic control tests, trace replay artifacts, and backend parity checks.
