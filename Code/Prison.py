import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from dataclasses import dataclass
from IPython.display import Image
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
from matplotlib.animation import FuncAnimation, PillowWriter

payoff_matrices = {
    "Default": {
        ("C", "C"): (3, 3),
        ("C", "D"): (0, 4),
        ("D", "C"): (4, 0),
        ("D", "D"): (1, 1),
    },
    "Canonical": {
        ("C", "C"): (-1, -1),
        ("C", "D"): (-3, 0),
        ("D", "C"): (0, -3),
        ("D", "D"): (-2, -2),
    },
    "Snowdrift": {
        ("C", "C"): (500, 500),
        ("C", "D"): (200, 800),
        ("D", "C"): (800, 200),
        ("D", "D"): (0, 0),
    },
}


class ActionStrategy:
    """Strategy that always plays its current action, randomly initialized."""

    def __init__(self, rng, initial_cooperation_degree=0.5):
        self.rng = rng
        self.action = "C" if self.rng.random() < initial_cooperation_degree else "D"

    def decide(self, agent_history):
        return self.action

    def set_action(self, action):
        self.action = action


class ImitationStrategy(ActionStrategy):
    """Imitate the action with the highest mean payoff in interactions."""

    def decide(self, agent_history):
        totals = {"C": 0.0, "D": 0.0}
        counts = {"C": 0, "D": 0}
        for interactions in agent_history.values():
            for inter in interactions:
                counts[inter.own_action] += 1
                counts[inter.neighbor_action] += 1
                totals[inter.own_action] += inter.own_reward
                totals[inter.neighbor_action] += inter.neighbor_reward

        mean_C = totals["C"] / counts["C"] if counts["C"] else 0
        mean_D = totals["D"] / counts["D"] if counts["D"] else 0

        if mean_C > mean_D:
            self.action = "C"
        elif mean_D > mean_C:
            self.action = "D"
        else:  # in case of a tie, continue with the existing strategy
            pass

        return self.action


class FermiStrategy(ActionStrategy):
    """
    Pairwise Fermi imitation:
    pick random neighbor j, imitate j with probability sigmoid((pi_j - pi_i)/K),
    where pi_x is the agent's (normalized) total payoff from the last round.
    """

    def __init__(self, rng, temperature=0.1):
        super().__init__(rng)
        self.K = temperature
        self._agents_ref = None  # injected by NetworkSimulation

    def decide(self, agent_history):
        if not agent_history:
            return self.action

        if self._agents_ref is None:
            raise RuntimeError(
                "FermiStrategy needs _agents_ref injected (agents dict)."
            )

        neighbour_id = self.rng.choice(list(agent_history.keys()))
        # If we have never recorded neighbor action, do nothing
        interactions = agent_history.get(neighbour_id, [])
        if not interactions:
            return self.action

        # Correct payoffs: compare last-round total payoffs
        # (these are updated each round in NetworkSimulation._play_round)
        # Need "self" agent id; infer it from the interactions? Not stored.
        # So we compare using the payoff stored on the owning Agent object:
        # We must access "my payoff" somehow. Easiest: store it in history? not.
        #
        # Workaround: use mean of own_reward across ALL neighbours as self payoff estimate
        # and neighbour's actual .payoff from agents_ref.

        # Self payoff estimate (across all neighbours in window)
        all_own = []
        for neigh, ints in agent_history.items():
            for it in ints:
                all_own.append(it.own_reward)
        payoff_self = float(np.mean(all_own)) if all_own else 0.0

        # Neighbour payoff from neighbour's own last-round total payoff
        payoff_neigh = float(self._agents_ref[neighbour_id].payoff)

        delta = payoff_neigh - payoff_self

        if self.K == 0:
            p_switch = 1.0 if delta > 0 else 0.0
        else:
            x = delta / self.K
            x = max(min(x, 700), -700)
            p_switch = 1.0 / (1.0 + math.exp(-x))

        if self.rng.random() < p_switch:
            self.action = interactions[-1].neighbor_action

        return self.action


class ReinforcementLearningStrategy(ActionStrategy):
    """
    Q-learning strategy with epsilon-greedy action selection.
    """

    def __init__(self, rng, learning_rate=0.1, epsilon=0.1, initial_q=0.0):
        super().__init__(rng)
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.q = {"C": float(initial_q), "D": float(initial_q)}
        self._last_action = None
        self._last_reward = 0.0

    def decide(self, agent_history):
        # observe most recent reward from any interaction (if exists)
        last = None
        for interactions in agent_history.values():
            if interactions:
                cand = interactions[-1]
                if last is None:
                    last = cand
        if last is not None:
            self._last_reward = last.own_reward

        # update Q for the action we previously played
        if self._last_action is not None:
            a = self._last_action
            self.q[a] = self.q[a] + self.alpha * (self._last_reward - self.q[a])

        # choose next action (epsilon-greedy)
        if self.rng.random() < self.epsilon:
            action = "C" if self.rng.random() < 0.5 else "D"
        else:
            if self.q["C"] > self.q["D"]:
                action = "C"
            elif self.q["D"] > self.q["C"]:
                action = "D"
            else:
                action = "C" if self.rng.random() < 0.5 else "D"

        self.action = action
        self._last_action = action
        return self.action


class TitForTatStrategy(ActionStrategy):
    """
    Start random, then mirror the most recent neighbor action.
    """

    def decide(self, agent_history):
        if not agent_history:
            return self.action

        # find the most recent neighbor action across all interactions
        last_action = None
        for interactions in agent_history.values():
            if interactions:
                last_action = interactions[-1].neighbor_action

        if last_action == "D":
            self.action = "D"
        elif last_action == "C":
            self.action = "C"

        return self.action


class Agent:
    """Minimal agent holding a strategy, payoff, and history."""

    @dataclass
    class Interaction:
        own_action: str
        own_reward: float
        neighbor_action: str
        neighbor_reward: float

    def __init__(self, agent_id, strategy, history_window=5, store_history=True):
        self.id = agent_id
        self.strategy = strategy
        self.history = {}
        self.payoff = 0.0
        self.history_window = history_window
        self.store_history = store_history

    def choose_action(self):
        return self.strategy.decide(self.history)

    def record_interaction(
        self, neighbor_id, own_action, neighbor_action, own_reward, neighbor_reward
    ):
        self.payoff += own_reward
        if not self.store_history:
            return
        lst = self.history.setdefault(neighbor_id, [])
        lst.append(
            self.Interaction(own_action, own_reward, neighbor_action, neighbor_reward)
        )
        if len(lst) > self.history_window:
            del lst[: -self.history_window]


class Network:
    """NetworkX graph wrapper."""

    def generate_graph(self, kind, n, seed=None, **kwargs):
        """Generate a networkx graph by name."""
        if kind == "grid":
            side_length = int(math.isqrt(n))
            if side_length * side_length != n:
                raise ValueError(f"Grid graph requires perfect-square n, got {n}.")
            graph = nx.convert_node_labels_to_integers(
                nx.grid_2d_graph(side_length, side_length)
            )

        elif kind == "stochastic_block":
            sizes = kwargs.pop("sizes")
            p = kwargs.pop("p")
            graph = nx.stochastic_block_model(sizes, p, seed=seed, **kwargs)

        else:
            generators = {
                "erdos_renyi": nx.erdos_renyi_graph,
                "watts_strogatz": nx.watts_strogatz_graph,
                "barabasi_albert": nx.barabasi_albert_graph,
            }
            graph = generators[kind](n, seed=seed, **kwargs)

        self.kind = kind
        self.graph = graph
        self.seed = seed

    def neighbour(self, agent_id):
        """Return neighbour agent IDs."""
        if self.graph:
            return list(self.graph.neighbors(agent_id))
        else:
            raise NotImplementedError


class NetworkSimulation(Network):
    """
    Base class for running evolutionary games on any NetworkX graph.
    """

    def __init__(
        self,
        kind="grid",
        n=100,
        seed=42,
        rounds=100,
        strategy=ActionStrategy,
        strategy_kwargs=None,
        payoff_matrix=None,
        T=None,
        R=1.0,
        P=0.0,
        S=0.0,
        rng=None,
        history_window=20,
        store_history=True,
        store_snapshots=True,
        **graph_kwargs,
    ):
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs or {}
        self.rounds = rounds

        # --- NEW: define payoff inside class if payoff_matrix not provided ---
        if payoff_matrix is None:
            if T is None:
                raise ValueError("Provide either payoff_matrix or T (with R,P,S).")

            payoff_matrix = {
                ("C", "C"): (R, R),
                ("C", "D"): (S, T),
                ("D", "C"): (T, S),
                ("D", "D"): (P, P),
            }
        self.payoff_matrix = payoff_matrix

        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.history_window = history_window
        self.store_history = store_history
        self.store_snapshots = store_snapshots
        self.generate_graph(kind=kind, n=n, seed=seed, **graph_kwargs)
        self.edge_list = list(self.graph.edges())
        self.deg = dict(self.graph.degree())
        self._neighbor_index_cache = None
        self.agents = {}
        self.snapshots = []
        self._initialize_agents()

    def _initialize_agents(self):
        for agent_id in self.graph.nodes:
            strat = self.strategy(self.rng, **self.strategy_kwargs)
            self.agents[agent_id] = Agent(
                agent_id,
                strat,
                history_window=self.history_window,
                store_history=self.store_history,
            )

            for agent in self.agents.values():
                agent.strategy._agents_ref = self.agents

    def _neighbor_indices(self):
        """Cache node and neighbor indices for fast neighbor stats."""
        if self._neighbor_index_cache is not None:
            return self._neighbor_index_cache

        nodes = list(self.graph.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        neigh_idx = []
        for node in nodes:
            neigh = list(self.graph.neighbors(node))
            if not neigh:
                continue
            neigh_idx.append((index[node], [index[n] for n in neigh]))

        self._neighbor_index_cache = neigh_idx
        return neigh_idx

    def _reset_payoffs(self):
        for agent in self.agents.values():
            agent.payoff = 0.0

    # fast inner loop: precompute lookups and actions once
    def _play_round(self):
        agents = self.agents
        payoff_matrix = self.payoff_matrix
        edge_list = self.edge_list
        deg = self.deg

        for agent in agents.values():
            agent.payoff = 0.0

        actions = {node: agents[node].choose_action() for node in agents}
        for node_a, node_b in edge_list:
            action_a = actions[node_a]
            action_b = actions[node_b]
            payoff_a, payoff_b = payoff_matrix.get((action_a, action_b), (0, 0))
            ka = deg.get(node_a, 0)
            kb = deg.get(node_b, 0)
            if ka <= 0 or kb <= 0:
                continue

            # --- NEW: degree-normalized rewards ---
            payoff_a_norm = payoff_a / ka
            payoff_b_norm = payoff_b / kb

            # --- NEW: accumulate round payoff (this is what Fermi should compare) ---
            agents[node_a].payoff += payoff_a_norm
            agents[node_b].payoff += payoff_b_norm

            agents[node_a].record_interaction(
                node_b, action_a, action_b, payoff_a_norm, payoff_b_norm
            )
            agents[node_b].record_interaction(
                node_a, action_b, action_a, payoff_b_norm, payoff_a_norm
            )

            agents[node_a].record_interaction(
                node_b, action_a, action_b, payoff_a, payoff_b
            )
            agents[node_b].record_interaction(
                node_a, action_b, action_a, payoff_b, payoff_a
            )

    def _get_state(self):
        return {
            node_id: (1 if agent.strategy.action == "D" else 0)
            for node_id, agent in self.agents.items()
        }

    def state01_array(self):
        """Return state array in graph node order: 0=C, 1=D."""
        return np.fromiter(
            (
                1 if self.agents[i].strategy.action == "D" else 0
                for i in self.graph.nodes()
            ),
            dtype=np.uint8,
            count=self.graph.number_of_nodes(),
        )

    def encode_state(self):  # for speed
        arr = self.state01_array()
        return np.packbits(arr, bitorder="little").tobytes()

    def decode_state(self, packed):
        n = self.graph.number_of_nodes()
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        return bits[:n].astype(np.uint8)

    # Detect fixed points or cycles by hashing compact states
    def run_until_attractor(
        self,
        max_steps=2000,
        check_every=1,
        store_cycle_states=True,
    ):
        seen = {}
        cache = []
        for t in range(max_steps + 1):
            key = None
            if t % check_every == 0:
                key = self.encode_state()
                if key in seen:
                    t0 = seen[key]
                    period = t - t0
                    attractor = "fixed" if period == 1 else "cycle"
                    cycle_states = cache[t0:t] if store_cycle_states else None
                    return {
                        "t_end": t,
                        "t_cycle_start": t0,
                        "period": period,
                        "attractor": attractor,
                        "cycle_states": cycle_states,
                    }
                seen[key] = t

            if store_cycle_states:
                if key is None:
                    key = self.encode_state()
                cache.append(key)

            self.step()

        return {
            "t_end": max_steps,
            "t_cycle_start": None,
            "period": None,
            "attractor": "unknown",
            "cycle_states": None,
        }

    def cooperation_assortment(self, state01=None):
        """Return long-run cooperation fraction and neighbor assortment."""
        if state01 is None:
            state01 = self.state01_array()
        coop01 = 1 - state01  # 1=C, 0=D for correlation

        xs = []
        neigh_means = []
        for node_i, neigh_is in self._neighbor_indices():
            xs.append(coop01[node_i])
            neigh_means.append(float(np.mean(coop01[neigh_is])))

        if len(xs) < 2:
            r = 0.0
        else:
            xs_arr = np.array(xs, dtype=float)
            neigh_arr = np.array(neigh_means, dtype=float)
            if np.std(xs_arr) == 0 or np.std(neigh_arr) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(xs_arr, neigh_arr)[0, 1])

        return {
            "coop_frac": float((coop01 == 1).mean()),
            "assortment_r": r,
        }

    def cooperation_metrics(self, state01=None):
        """Return basic cooperator cluster metrics for the current state."""
        if state01 is None:
            state01 = self.state01_array()
        coop_nodes = [
            node for node, val in zip(self.graph.nodes(), state01) if val == 0
        ]
        if not coop_nodes:
            return {
                "coop_frac": 0.0,
                "n_coop_clusters": 0,
                "largest_coop_cluster": 0,
                "mean_coop_cluster_size": 0.0,
            }
        H = self.graph.subgraph(coop_nodes)
        comps = list(nx.connected_components(H))
        sizes = sorted([len(c) for c in comps], reverse=True)
        return {
            "coop_frac": float((state01 == 0).mean()),
            "n_coop_clusters": int(len(sizes)),
            "largest_coop_cluster": int(sizes[0]) if sizes else 0,
            "mean_coop_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
        }

    def step(self):
        self._play_round()
        if self.store_snapshots:
            self.snapshots.append(self._get_state())

    def run(self):
        for _ in range(self.rounds):
            self.step()


class PayoffMatrix:
    def __init__(self, beta_values, kbar=4.0, c=1.0):
        self.matrices, self.meta = self.generate(beta_values, kbar, c)

    def donation(self, b: float, c: float = 1.0):
        """
        Payoff builder: Donation game

        Returns:
        dictionary: payoff_matrix
        """
        R = b - c
        S = -c
        T = b
        P = 0.0
        return {
            ("C", "C"): (R, R),
            ("C", "D"): (S, T),
            ("D", "C"): (T, S),
            ("D", "D"): (P, P),
        }

    def generate(self, beta_values, kbar, c):
        """
        Returns:
        payoff_mats: dict name -> payoff_matrix
        meta_df:     DataFrame recording beta,b,c,kbar,b_over_c
        """
        matrices = {}
        rows = []
        for beta in beta_values:
            b = (
                float(beta) * float(kbar) * float(c)
            )  # b/c = beta*kbar  => b = beta*kbar*c
            name = f"beta_{beta:.2f}"  # IMPORTANT: first number is beta for parsing
            matrices[name] = self.donation(b=b, c=c)
            rows.append(
                {
                    "payoff": name,
                    "beta": float(beta),
                    "kbar": float(kbar),
                    "c": float(c),
                    "b": float(b),
                    "b_over_c": float(b / c),
                }
            )
        return matrices, pd.DataFrame(rows)


def experiment(
    network_simulation,
    strategy_class,
    strategy_kwargs=None,
    steps=100,
    seed=42,
    interval=300,
    payoff_matrix=None,
    title="",
    kind="grid",
    n=400,
    is_grid=False,
    save_gif=True,
    gif_dir="gifs",
    gif_path=None,
    **graph_kwargs,
):
    """
    Produce animations showing the network state over time.
    """
    payoff_matrix = payoff_matrix
    strategy_kwargs = strategy_kwargs or {}
    simulation = network_simulation(
        kind=kind,
        n=n,
        seed=seed,
        rounds=steps,
        payoff_matrix=payoff_matrix,
        strategy=strategy_class,
        strategy_kwargs=strategy_kwargs,
        **graph_kwargs,
    )

    graph = simulation.graph
    n_nodes = graph.number_of_nodes()

    C_COOP, C_DEFECT = "#40B0A6", "#FFBE6A"
    cmap = ListedColormap([C_COOP, C_DEFECT])
    fig, (ax_sim, ax_stats) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [4, 1]}
    )

    plot_style = globals().get(
        "PLOT_STYLE",
        {
            "title_size": 12,
            "label_size": 10,
            "tick_size": 9,
            "line_width": 1.6,
            "marker_size": 5,
        },
    )

    # -------------------------
    # Stats plot (C% and D%)
    # -------------------------
    xs, ys_c, ys_d = [], [], []

    (line_c,) = ax_stats.plot([], [], lw=2.2, label="Cooperate", color=C_COOP)
    (line_d,) = ax_stats.plot([], [], lw=2.2, label="Defect", color=C_DEFECT)

    ax_stats.set_xlim(0, steps)
    ax_stats.set_ylim(0, 100)
    ax_stats.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_stats.set_ylabel("Population share (%)")
    ax_stats.set_xlabel("Step")
    ax_stats.set_title("Population share over time", fontsize=plot_style["title_size"])
    ax_stats.grid(True, linestyle=":", alpha=0.4)
    ax_stats.legend(frameon=False, ncol=2, loc="upper right")

    # -------------------------
    # Simulation plot
    # -------------------------
    if is_grid:
        dim = int(math.isqrt(n_nodes))
        if dim * dim != n_nodes:
            raise ValueError(f"Grid mode needs square number of nodes, got {n_nodes}.")

        def state_as_grid():
            state = simulation._get_state()
            grid = [[0] * dim for _ in range(dim)]
            for node, val in state.items():
                grid[node // dim][node % dim] = val
            return grid

        sim_artist = ax_sim.imshow(state_as_grid(), cmap=cmap, vmin=0, vmax=1)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])

        def update_sim():
            sim_artist.set_data(state_as_grid())

    else:
        pos = nx.spring_layout(graph, seed=seed)
        nodelist = list(graph.nodes())
        nx.draw_networkx_edges(graph, pos, ax=ax_sim, alpha=0.3, edge_color="gray")
        state0 = simulation._get_state()
        sim_artist = nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodelist,
            node_color=[state0[i] for i in nodelist],
            cmap=cmap,
            vmin=0,
            vmax=1,
            node_size=80,
            edgecolors="gray",
            ax=ax_sim,
        )
        ax_sim.axis("off")

        def update_sim():
            state = simulation._get_state()
            sim_artist.set_array([state[i] for i in nodelist])

    # -------------------------
    # Animation update
    # -------------------------
    def update(frame):
        if frame > 0:
            simulation.step()

        display_title = title or "Simulation"
        ax_sim.set_title(f"{display_title} - Step {frame}/{steps}")

        update_sim()

        state = simulation._get_state()
        d = sum(state.values())
        c = n_nodes - d

        xs.append(frame)
        ys_c.append(100 * c / n_nodes)
        ys_d.append(100 * d / n_nodes)

        line_c.set_data(xs, ys_c)
        line_d.set_data(xs, ys_d)

        return sim_artist, line_c, line_d

    anim = FuncAnimation(
        fig,
        update,
        frames=steps + 1,
        interval=interval,
        blit=True,
        repeat=False,
    )

    if not save_gif:
        return anim

    safe_title = title or f"animation_{seed}"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe_title)

    if gif_path is None:
        gif_dir = Path(gif_dir)
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"{safe_name}.gif"
    else:
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)

    fps = max(1, int(round(1000 / interval)))
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))
    plt.close(fig)

    return Image(filename=str(gif_path))
