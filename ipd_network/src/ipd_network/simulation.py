"""Mesa model for Iterated Prisoner's Dilemma on networks."""

import random

import axelrod as axl
from mesa import Model
from mesa.space import NetworkGrid

from ipd_network.agents import IPDAgent
from ipd_network.core import Agent, Network
from ipd_network.network import generate_graph
from ipd_network.simple_strategies import RandomActionStrategy
from ipd_network.strategies import create_strategy
from ipd_network.utils import get_logger, set_random_seed

logger = get_logger(__name__)


DEFAULT_PAYOFFS = {
    (axl.Action.C, axl.Action.C): (3, 3),
    (axl.Action.C, axl.Action.D): (0, 5),
    (axl.Action.D, axl.Action.C): (5, 0),
    (axl.Action.D, axl.Action.D): (1, 1),
}


class SimulationConfig:
    """Configuration for running a simulation."""

    def __init__(self, num_agents, **overrides):
        self.num_agents = num_agents
        self.graph_kind = overrides.get("graph_kind", "erdos_renyi")
        self.graph_kwargs = overrides.get("graph_kwargs", {"p": 0.1})
        self.rounds = overrides.get("rounds", 10)
        self.seed = overrides.get("seed")
        self.strategies = overrides.get("strategies", ["tit_for_tat", "defector"])
        self.payoff_matrix = overrides.get("payoff_matrix", DEFAULT_PAYOFFS)


class IPDModel(Model):
    """Mesa model that runs IPD interactions on a network."""

    def __init__(self, config):
        super().__init__()
        set_random_seed(config.seed)
        self.config = config
        self.graph = generate_graph(
            config.graph_kind,
            config.num_agents,
            seed=config.seed,
            **config.graph_kwargs,
        )
        self.grid = NetworkGrid(self.graph)
        self.agents = []
        self.round = 0
        self._init_agents()

    def _init_agents(self):
        for node_id in self.graph.nodes:
            strategy_name = self.random.choice(self.config.strategies)
            strategy = create_strategy(strategy_name)
            agent = IPDAgent(unique_id=node_id, model=self, strategy=strategy)
            self.grid.place_agent(agent, node_id)
            self.agents.append(agent)
            logger.debug("Placed agent %s with %s", node_id, strategy_name)

    def _play_pair(self, agent_a, agent_b):
        action_a = agent_a.choose_action(agent_b)
        action_b = agent_b.choose_action(agent_a)

        agent_a.strategy.history.append(action_a, action_b)
        agent_b.strategy.history.append(action_b, action_a)

        payoff_a, payoff_b = self.config.payoff_matrix[(action_a, action_b)]
        agent_a.record_result(action_a, payoff_a)
        agent_b.record_result(action_b, payoff_b)

    def step(self):
        """Run one interaction round over all edges."""
        for node_u, node_v in self.graph.edges:
            agents_u = self.grid.get_cell_list_contents([node_u])
            agents_v = self.grid.get_cell_list_contents([node_v])
            if not agents_u or not agents_v:
                continue
            self._play_pair(agents_u[0], agents_v[0])

        self.round += 1
        logger.info("Completed round %s", self.round)

    def run(self):
        """Run the simulation for the configured number of rounds."""
        for _ in range(self.config.rounds):
            self.step()

    def reset(self):
        """Reset agents for a fresh run without rebuilding the network."""
        for agent in self.agents:
            agent.reset()
        self.round = 0

    def iter_agents(self):
        """Yield all agents in the model."""
        return iter(self.agents)


class GridImitationModel:
    """Imitation dynamics on a periodic grid with fixed C/D actions."""

    def __init__(self, size=20, rounds=50, seed=None, payoff_matrix=None):
        self.size = size
        self.rounds = rounds
        self.seed = seed
        self.payoff_matrix = payoff_matrix or {
            ("C", "C"): (3, 3),
            ("C", "D"): (0, 5),
            ("D", "C"): (5, 0),
            ("D", "D"): (1, 1),
        }
        self.random = random.Random(seed)
        # Periodic grid gives each node exactly four neighbors.
        self.graph = generate_graph("grid", size, m=size, periodic=True)
        self.network = Network(self.graph)
        self.agents = {}
        self._init_agents()
        self.snapshots = []

    def _init_agents(self):
        """Create agents with random starting actions."""
        for node_id in self.graph.nodes:
            strategy = RandomActionStrategy(self.random)
            self.agents[node_id] = Agent(node_id, strategy)

    def _reset_payoffs(self):
        # Payoffs are per-round for imitation.
        for agent in self.agents.values():
            agent.payoff = 0.0

    def _play_round(self):
        """Compute payoffs for the current actions."""
        self._reset_payoffs()
        # Each edge is evaluated once, updating both endpoints.
        for node_a, node_b in self.graph.edges:
            agent_a = self.agents[node_a]
            agent_b = self.agents[node_b]
            action_a = agent_a.strategy.decide(agent_a.id, agent_a.history, {})
            action_b = agent_b.strategy.decide(agent_b.id, agent_b.history, {})
            payoff_a, payoff_b = self.payoff_matrix[(action_a, action_b)]
            agent_a.record_interaction(node_b, action_a, action_b, payoff_a)
            agent_b.record_interaction(node_a, action_b, action_a, payoff_b)

    def _imitate_best_neighbor(self):
        """Copy the action of the best-performing neighbor (or self)."""
        next_actions = {}
        for node_id in self.graph.nodes:
            candidates = [node_id] + list(self.network.neighbors(node_id))
            best_score = max(self.agents[candidate].payoff for candidate in candidates)
            best_nodes = [
                candidate
                for candidate in candidates
                if self.agents[candidate].payoff == best_score
            ]
            chosen = self.random.choice(best_nodes)
            next_actions[node_id] = self.agents[chosen].strategy.action

        for node_id, action in next_actions.items():
            self.agents[node_id].strategy.set_action(action)

    def step(self):
        """Run one payoff calculation and one imitation update."""
        self._play_round()
        self._imitate_best_neighbor()
        self.snapshots.append(self.action_grid())

    def run(self):
        """Run the model for the configured number of rounds."""
        for _ in range(self.rounds):
            self.step()

    def action_grid(self):
        """Return a 2D grid of actions for visualization."""
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for node_id, agent in self.agents.items():
            # Grid nodes are relabeled to ints, so map back to row/col.
            row = node_id // self.size
            col = node_id % self.size
            grid[row][col] = 1 if agent.strategy.action == "D" else 0
        return grid
