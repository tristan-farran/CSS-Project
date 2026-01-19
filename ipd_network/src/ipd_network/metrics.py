"""Metrics for analyzing IPD simulations."""

import axelrod as axl


def cooperation_rate(agents):
    """Compute the fraction of cooperative actions across all agents."""
    total_actions = 0
    total_cooperate = 0
    for agent in agents:
        for action in agent.history:
            total_actions += 1
            if action == axl.Action.C:
                total_cooperate += 1
    if total_actions == 0:
        return 0.0
    return total_cooperate / total_actions


def average_score(agents):
    """Return the average score per agent."""
    scores = [agent.score for agent in agents]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def score_range(agents):
    """Return the min and max agent scores."""
    scores = [agent.score for agent in agents]
    if not scores:
        return 0.0, 0.0
    return min(scores), max(scores)
