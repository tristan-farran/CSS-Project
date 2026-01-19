"""Mesa model for Iterated Prisoner's Dilemma on networks."""

import axelrod as axl
from mesa import Model
from mesa.space import NetworkGrid

from ipd_network.agents import IPDAgent
from ipd_network.network import generate_graph
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
