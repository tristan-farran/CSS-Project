"""Agent definitions for the IPD model."""

from mesa import Agent

from ipd_network.utils import get_logger

logger = get_logger(__name__)


class IPDAgent(Agent):
    """Mesa agent wrapper around an Axelrod strategy."""

    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy
        self.score = 0.0
        self.history = []

    def reset(self):
        """Reset scores and history for a new simulation run."""
        self.score = 0.0
        self.history = []
        self.strategy.reset()

    def choose_action(self, opponent):
        """Select an action against an opponent."""
        action = self.strategy.strategy(opponent.strategy)
        logger.debug("Agent %s plays %s", self.unique_id, action)
        return action

    def record_result(self, action, reward):
        """Update agent history and score."""
        self.history.append(action)
        self.score += reward

    def step(self):
        """Per-step hook; interactions are handled at the model level."""
        pass


def agent_scores(agents):
    """Return a mapping of agent id to cumulative score."""
    return {agent.unique_id: agent.score for agent in agents}
