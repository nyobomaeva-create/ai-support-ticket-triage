"""Transparent rule-based next actions for predicted tickets."""

from ticket_triage.constants import CATEGORIES, PRIORITIES

_CATEGORY_ACTIONS = {
    "billing": "Route to the billing queue and review the related transaction.",
    "technical": "Route to technical support and request reproducible steps.",
    "account": "Route to account support and verify the user's identity safely.",
    "general": "Route to customer success and provide the relevant documentation.",
}


def suggest_next_action(category: str, priority: str, uncertain: bool) -> str:
    """Return a deterministic action that never impersonates generated advice."""
    if category not in CATEGORIES:
        raise ValueError(f"Unsupported category: {category}.")
    if priority not in PRIORITIES:
        raise ValueError(f"Unsupported priority: {priority}.")

    prefixes = []
    if uncertain:
        prefixes.append("Manual review recommended.")
    if priority == "high":
        prefixes.append("Escalate immediately.")
    return " ".join([*prefixes, _CATEGORY_ACTIONS[category]])

