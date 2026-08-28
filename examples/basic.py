"""Minimal CPU example: constrain first, then rank the admissible frontier."""

from metamorpher import (
    ActionKind,
    ActionNode,
    ActionStatus,
    Constraint,
    ConstraintKind,
    ControllerState,
    EvidenceLedger,
    HeuristicLookaheadPolicy,
    TypedActionGraph,
)

graph = TypedActionGraph()
graph.add_node(
    ActionNode(
        "inspect",
        "Inspect the cheap upstream condition",
        ActionKind.OBSERVE,
        cost=0.1,
        information_value=5.0,
    )
)
graph.add_node(
    ActionNode(
        "repair",
        "Perform the expensive intervention",
        ActionKind.REPAIR,
        cost=10.0,
        harm=2.0,
        reversible=False,
    )
)
graph.add_constraint(
    Constraint(
        "inspect-before-repair",
        ConstraintKind.HARD_PREREQUISITE,
        ("inspect",),
        "repair",
    )
)
graph.validate()

state = ControllerState()
evidence = EvidenceLedger()
policy = HeuristicLookaheadPolicy()

frontier = graph.frontier(state, evidence)
selected = policy.select(graph, state, frontier.certified)
print("initial frontier:", frontier.certified)
print("selected:", selected)

state.action_status[selected] = ActionStatus.COMPLETED
frontier = graph.frontier(state, evidence)
print("frontier after inspection:", frontier.certified)
print("model-relative only: an admissible action is not a real-world safety guarantee")
