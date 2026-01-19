"""Small strategy helpers for simple demos."""

import random

from ipd_network.core import Strategy


class ActionStrategy(Strategy):
    """Strategy that always plays its current action."""

    def __init__(self, action):
        self.action = action

    def decide(self, agent_id, own_history, neighbors_history):
        return self.action

    def set_action(self, action):
        self.action = action


class RandomActionStrategy(ActionStrategy):
    """Start with a random action and then behave like ActionStrategy."""

    def __init__(self, rng=None):
        rng = rng or random.Random()
        super().__init__(rng.choice(["C", "D"]))
