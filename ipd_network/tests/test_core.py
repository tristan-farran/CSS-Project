import networkx as nx

from ipd_network.core import Agent, Network, Strategy


class AlwaysCooperate(Strategy):
    def decide(self, agent_id, own_history, neighbors_history):
        return "C"


class AlwaysDefect(Strategy):
    def decide(self, agent_id, own_history, neighbors_history):
        return "D"


def test_strategy_swappable():
    agent_c = Agent(agent_id=1, strategy=AlwaysCooperate())
    agent_d = Agent(agent_id=2, strategy=AlwaysDefect())

    assert agent_c.choose_action({}) == "C"
    assert agent_d.choose_action({}) == "D"


def test_agent_history_and_payoff():
    agent = Agent(agent_id=1, strategy=AlwaysCooperate())

    agent.record_interaction(neighbor_id=2, own_action="C", neighbor_action="D", reward=0)
    agent.record_interaction(neighbor_id=2, own_action="D", neighbor_action="C", reward=5)
    agent.record_interaction(neighbor_id=3, own_action="C", neighbor_action="C", reward=3)

    assert agent.payoff == 8
    assert len(agent.neighbors_interactions(2)) == 2
    assert len(agent.neighbors_interactions(3)) == 1
    assert agent.neighbors_interactions(4) == []


def test_network_neighbors():
    graph = nx.path_graph(3)
    network = Network(graph)

    assert set(network.neighbors(1)) == {0, 2}
