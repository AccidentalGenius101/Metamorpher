"""Fail-closed compiler for the line-oriented Metamorpher text language."""

from __future__ import annotations

import json
import shlex

from .cognition import CandidateStructure
from .model import ActionKind, ActionNode, ClaimTier, Constraint, ConstraintKind, DomainTag
from .text import DiscourseInterpretation, TextEvidence
from .version_space import Hypothesis


class TextProgramError(ValueError):
    pass


class TextProgramCompiler:
    """Compile small, attributable model proposals into quarantined structure."""

    def compile(
        self, source_text: str, program: str, domain: DomainTag
    ) -> DiscourseInterpretation:
        evidence: list[TextEvidence] = []
        actions: dict[str, ActionNode] = {}
        suspicions: list[tuple[str, str, int, int]] = []
        before: list[tuple[str, str, int]] = []

        for line_number, raw_line in enumerate(program.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                statement, quote = self._statement_and_quote(line, source_text)
                operation, *arguments = shlex.split(statement)
                operation = operation.upper()
                if operation == "OBSERVE" and len(arguments) == 2 and quote:
                    evidence.append(
                        TextEvidence(
                            arguments[0],
                            json.loads(arguments[1]),
                            quote[1],
                            quote[2],
                        )
                    )
                elif operation == "SUSPECT" and len(arguments) == 1 and quote:
                    suspicions.append((arguments[0], quote[0], quote[1], quote[2]))
                elif operation == "ACTION" and len(arguments) == 3 and quote is None:
                    action_id, kind, label = arguments
                    if action_id in actions:
                        raise TextProgramError(f"duplicate action: {action_id}")
                    actions[action_id] = ActionNode(
                        action_id,
                        label,
                        ActionKind(kind.lower()),
                    )
                elif operation == "BEFORE" and len(arguments) == 2 and quote is None:
                    before.append((arguments[0], arguments[1], line_number))
                else:
                    raise TextProgramError("invalid statement shape")
            except (ValueError, json.JSONDecodeError) as exc:
                if isinstance(exc, TextProgramError):
                    raise
                raise TextProgramError(f"line {line_number}: {exc}") from exc

        if not evidence and not actions and not suspicions:
            raise TextProgramError("program contains no semantic statements")
        constraints: list[Constraint] = []
        for source, target, line_number in before:
            if source not in actions or target not in actions:
                raise TextProgramError(
                    f"line {line_number}: BEFORE references an unknown action"
                )
            constraints.append(
                Constraint(
                    f"candidate-before-{source}-{target}",
                    ConstraintKind.HARD_PREREQUISITE,
                    (source,),
                    target,
                    tier=ClaimTier.CANDIDATE,
                    provenance=(f"text-program-line:{line_number}",),
                )
            )
        safe_actions = frozenset(
            action.id
            for action in actions.values()
            if action.kind
            in {ActionKind.OBSERVE, ActionKind.TEST, ActionKind.AUDIT, ActionKind.ESCALATE}
        )
        hypotheses = tuple(
            Hypothesis(
                hypothesis_id,
                safe_actions,
                domain=domain,
                provenance=(
                    f"speaker_suspicion:{start}:{end}",
                    f"quote:{quote}",
                ),
            )
            for hypothesis_id, quote, start, end in suspicions
        )
        candidates = CandidateStructure(
            tuple(actions.values()),
            tuple(constraints),
            hypotheses,
            "compiled from a validated text program",
        )
        return DiscourseInterpretation(tuple(evidence), candidates)

    @staticmethod
    def _statement_and_quote(
        line: str, source_text: str
    ) -> tuple[str, tuple[str, int, int] | None]:
        if " @ " not in line:
            return line, None
        statement, raw_quote = line.rsplit(" @ ", 1)
        parsed = shlex.split(raw_quote)
        if len(parsed) != 1 or not parsed[0]:
            raise TextProgramError("source anchor must contain exactly one quotation")
        quote = parsed[0]
        start = source_text.find(quote)
        if start < 0:
            raise TextProgramError(f"source quotation not found: {quote!r}")
        if source_text.find(quote, start + 1) >= 0:
            raise TextProgramError(f"source quotation is ambiguous: {quote!r}")
        return statement, (quote, start, start + len(quote))
