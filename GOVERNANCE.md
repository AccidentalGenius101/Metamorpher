# Governance

Metamorpher uses a maintainer-led open-source model. The goal is fast public
experimentation with a clearly identifiable canonical implementation.

## Canonical project

The canonical source repository is
`https://github.com/AccidentalGenius101/Metamorpher`. Gabriel Aubin-Moreau is the
founding maintainer and release steward for the 0.x series.

Forks are permitted and encouraged by Apache-2.0. A fork becomes an official
Metamorpher release only when it is accepted into the canonical repository and
published by its maintainers.

## Decisions

- Ordinary changes are discussed and reviewed through issues and pull requests.
- Maintainers decide whether a change satisfies the documented semantics,
  evidence requirements, safety boundary, and release scope.
- Changes to hard-frontier semantics, evidence censoring, external authority,
  or model-relative status language require an explicit design note and tests.
- Major architectural changes should include an ablation or counterexample that
  makes their benefit falsifiable.
- Security fixes may be developed privately and disclosed after a release is
  available.

If consensus is absent, the release steward makes the final decision and records
the rationale. This is authority over the canonical release, not over permitted
forks or independent research.

## Releases

Official releases originate from the canonical repository, have a versioned
changelog, pass the release gates, and preserve the Apache-2.0 license and
NOTICE. Pre-1.0 releases may change APIs, but semantic changes must be called out
explicitly.

## Contributions

The project uses inbound-equals-outbound licensing: accepted contributions are
provided under Apache-2.0 without a separate contributor license agreement.
Contributors must have the right to submit their work and must identify any
third-party code or data and its license.

## Governance changes

As the contributor base grows, maintainers may adopt a technical steering group
or a foundation. Any such transition must be proposed publicly and preserve the
project history, license, attribution, and availability of prior releases.
