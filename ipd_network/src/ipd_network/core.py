"""Core interfaces for IPD simulations."""

import networkx as nx


class Interaction:
    """Store the actions taken in a single interaction."""

    def __init__(self, own_action, neighbor_action):
        self.own_action = own_action
        self.neighbor_action = neighbor_action


class Strategy:
    """Base strategy interface for IPD agents."""

    def decide(self, agent_id, own_history, neighbors_history):
        """Choose an action based on histories."""
        raise NotImplementedError


class Agent:
    """Minimal agent holding a strategy, payoff, and per-neighbor history."""

    def __init__(self, agent_id, strategy):
        self.id = agent_id
        self.strategy = strategy
        self.payoff = 0.0
        self.history = {}

    def choose_action(self, neighbors_history):
        """Select an action using the configured strategy."""
        return self.strategy.decide(self.id, self.history, neighbors_history)

    def record_interaction(self, neighbor_id, own_action, neighbor_action, reward):
        """Record interaction outcome and update payoff."""
        self.history.setdefault(neighbor_id, []).append(
            Interaction(own_action, neighbor_action)
        )
        self.payoff += reward

    def neighbors_interactions(self, neighbor_id):
        """Return the interaction history with a given neighbor."""
        return list(self.history.get(neighbor_id, []))


class Network:
    """Network wrapper around a networkx graph."""

    def __init__(self, graph):
        self.graph = graph

    def neighbors(self, node_id):
        """Return neighbor node ids."""
        return self.graph.neighbors(node_id)


def make_network(graph):
    """Wrap a networkx graph to keep the simulator graph-agnostic."""
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    return Network(graph)
