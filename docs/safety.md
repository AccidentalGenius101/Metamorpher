# Safety and limitations

## Status

Metamorpher is experimental research software. It is not a production safety
system, a formal verifier, a licensed professional, or an independent source of
truth.

`supported_under_model` means only:

> Given the supplied candidate graph, recorded evidence, domain tag, and policy,
> the action is currently represented as admissible and preferred.

It does not mean the graph is complete, the observations are correct, the world
is stationary, the action is physically safe, or the result complies with law or
regulation.

## Do not use as autonomous authority

Do not rely on this prototype to autonomously control or approve:

- vehicles, machinery, power, pressure, heat, chemicals, or physical repairs;
- medical diagnosis or treatment;
- financial, legal, employment, eligibility, or access-control decisions;
- cybersecurity response or credential operations;
- any irreversible or high-consequence production workflow.

The included Sierra scenario is a controller simulation, not mechanical advice.

## Failure model

The explicit graph improves inspectability but creates no guarantee that its
contents match reality. Expected failure sources include:

- omitted or incorrect prerequisites;
- observations that are unavailable, delayed, censored, noisy, or adversarial;
- a wrong domain tag causing invalid memory reuse;
- false equivalence classes or premature partitions;
- incomplete hypotheses with an apparently common-safe action;
- learned scores that shift out of distribution;
- stale decisions after graph or evidence changes;
- prompt injection or tool-output manipulation in LLM adapters;
- accelerator, numeric, serialization, or integration defects;
- operator misunderstanding of model-relative terminology.

Independent audits help expose self-confirming loops, but an audit channel can
also fail or share the same blind spot. “Audited” is not synonymous with true.

## Integration requirements

Any serious experiment involving external systems should add safeguards outside
Metamorpher:

1. Default to simulation or dry-run.
2. Use explicit action schemas and allowlists.
3. Separate proposal, decision support, authorization, and execution identities.
4. Require human approval for irreversible or high-consequence actions.
5. Preserve independent physical/software interlocks.
6. Revalidate the decision token immediately before execution.
7. Record append-only evidence and action traces with access controls.
8. Treat missing feedback as censored, never successful by default.
9. Define rollback, escalation, incident response, and kill-switch procedures.
10. Validate within the intended domain using independent experts and tests.

No command-line flag can substitute for these controls.

## Hard constraints

A hard constraint is hard only inside the supplied model. The core prevents the
ordinary value scorer from overriding it, but cannot protect against:

- failure to represent the constraint;
- an authorized caller deleting or misclassifying it;
- a bad observation satisfying its guard;
- an executor ignoring the returned decision.

Externally governed constraints should require explicit authority and evidence
for revision. Learned correlations should remain soft until a human-defined
process promotes them.

## Accelerator safety

CUDA and Triton are optional performance paths. They must never weaken the hard
frontier, reinterpret unknown as false, or silently continue after a kernel
failure. A backend must fail closed for the affected batch; the surrounding
controller may then explicitly retry on the CPU reference path. Backend parity
tests are necessary but do not validate the real-world model.

## Language and claims

Use:

- “supported under the current model”;
- “represented prerequisite”;
- “common-safe across represented hypotheses”;
- “no invalid actions in benchmark X under assumptions Y.”

Avoid:

- “safe,” “guaranteed,” or “certified” without a precise qualification;
- extrapolating synthetic benchmark rates to real systems;
- presenting abstention as proof that every unknown was detected;
- implying that a model can know every boundary of its competence in advance.

## Production readiness

A production safety case would additionally require, at minimum, domain-specific
hazard analysis, formalized authority boundaries, threat modeling, authentication
and access control, secure audit retention, reliability engineering, independent
verification, human-factors testing, monitoring, incident response, and
regulatory review. This repository claims none of those properties.
