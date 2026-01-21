# %% [markdown]
# # Simulating the Iterated Prisoner's Dilemma
# ## Imports and configuration

# %%
import logging
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter

import networkx as nx
import math
from functools import partial


# %%
%matplotlib notebook
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["animation.embed_limit"] = 500

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )

# %% [markdown]
# ## Basic classes

# %%
payoff_matrices = { 
    "Default":          {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (0, 0) },
}

# %%
class Interaction:
    """Store the actions taken in a single interaction."""

    def __init__(self, own_action, neighbor_action):
        self.own_action = own_action
        self.neighbor_action = neighbor_action


# %%
class Strategy:
    """Base strategy interface for IPD agents."""

    def decide(self, agent_id, own_history, neighbors_history):
        """Choose an action based on histories."""
        raise NotImplementedError


class ActionStrategy(Strategy):
    """Strategy that always plays its current action."""

    def __init__(self, action):
        self.action = action

    def decide(self, agent_id, own_history, neighbors_history):
        return self.action

    def set_action(self, action):
        self.action = action


class RandomActionStrategy(ActionStrategy):
    """Start with a random action based on a specific cooperation rate."""

    def __init__(self, rng=None, c_rate=0.5):
        rng = rng or random.Random()
        # If random float [0.0, 1.0) is less than rate, choose C
        action = "C" if rng.random() < c_rate else "D"
        super().__init__(action)

# %%
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


# %%
class Network:
    """Network wrapper around a networkx graph."""

    def __init__(self, graph):
        self.graph = graph

    def neighbors(self, node_id):
        """Return neighbor node ids."""
        return self.graph.neighbors(node_id)


# %%
def generate_graph(kind, n, seed=None, **kwargs):
    """Generate a networkx graph by name."""
    if kind == "grid":
        cols = kwargs.pop("m", n)
        graph = nx.grid_2d_graph(n, cols)
        return nx.convert_node_labels_to_integers(graph)

    generators = {
        "erdos_renyi": nx.erdos_renyi_graph,
        "watts_strogatz": nx.watts_strogatz_graph,
        "barabasi_albert": nx.barabasi_albert_graph,
    }
    if kind not in generators:
        raise ValueError(f"Unknown graph kind: {kind}")
    return generators[kind](n, seed=seed, **kwargs)


# %%
def make_network(graph):
    """Wrap a networkx graph to keep the simulator graph-agnostic."""
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    return Network(graph)


# %% [markdown]
# ## Network simulation

# %%
class NetworkSimulation:
    """
    Base class for running evolutionary games on any NetworkX graph.
    """

    def __init__(self, graph, rounds=50, seed=None, payoff_matrix=payoff_matrices["Default"], initial_coop_rate=0.5):
        self.rounds = rounds
        self.seed = seed
        self.payoff_matrix = payoff_matrix
        self.initial_coop_rate = initial_coop_rate  # Store the rate
        self.random = random.Random(seed)
        
        self.graph = graph
        self.network = Network(self.graph)
        
        self.agents = {}
        self.snapshots = []
        self._init_agents()

    def _init_agents(self):
        """Create agents with random starting actions for every node in the graph."""
        for node_id in self.graph.nodes:
            # Pass the stored initial_coop_rate to the strategy
            strategy = RandomActionStrategy(self.random, c_rate=self.initial_coop_rate)
            self.agents[node_id] = Agent(node_id, strategy)

    # ... (Rest of the class remains unchanged: _reset_payoffs, _play_round, etc.)
    def _reset_payoffs(self):
        for agent in self.agents.values():
            agent.payoff = 0.0

    def _play_round(self):
        self._reset_payoffs()
        for node_a, node_b in self.graph.edges:
            agent_a = self.agents[node_a]
            agent_b = self.agents[node_b]
            
            action_a = agent_a.strategy.decide(agent_a.id, agent_a.history, {})
            action_b = agent_b.strategy.decide(agent_b.id, agent_b.history, {})
            
            payoff_a, payoff_b = self.payoff_matrix.get((action_a, action_b), (0, 0))
            
            agent_a.record_interaction(node_b, action_a, action_b, payoff_a)
            agent_b.record_interaction(node_a, action_b, action_a, payoff_b)

    def _update_strategies(self):
        raise NotImplementedError("Subclasses must implement _update_strategies")

    def get_action_state(self):
        return {
            node_id: (1 if agent.strategy.action == "D" else 0)
            for node_id, agent in self.agents.items()
        }

    def step(self):
        self._play_round()
        self._update_strategies()
        self.snapshots.append(self.get_action_state())

    def run(self):
        for _ in range(self.rounds):
            self.step()

# %%
class ImitationDynamics(NetworkSimulation):
    """
    Specific implementation of the simulation where agents 
    adopt the strategy of their most successful neighbor.
    """

    def _update_strategies(self):
        """Copy the action of the best-performing neighbor (or self)."""
        next_actions = {}
        
        # Iterate over all agents in the generic graph
        for node_id in self.graph.nodes:
            # Candidates are the node itself + its neighbors
            candidates = [node_id] + list(self.network.neighbors(node_id))
            
            # Find the max payoff among candidates
            best_score = max(self.agents[candidate].payoff for candidate in candidates)
            
            # Identify all candidates who achieved that score (tie-breaking)
            best_nodes = [
                candidate
                for candidate in candidates
                if self.agents[candidate].payoff == best_score
            ]
            
            # Randomly choose one of the best performing nodes
            chosen = self.random.choice(best_nodes)
            next_actions[node_id] = self.agents[chosen].strategy.action

        # Apply updates synchronously
        for node_id, action in next_actions.items():
            self.agents[node_id].strategy.set_action(action)

# %%
class FermiPairwiseComparison(NetworkSimulation):
    """
    Strategy where agents compare payoffs with a single random neighbor
    and switch strategies probabilistically based on the Fermi function.
    """
    
    def __init__(self, graph, rounds=50, seed=None, payoff_matrix=payoff_matrices["Default"], temperature=0.1, initial_coop_rate=0.5):
        """
        Args:
            temperature (float): Controls the noise level (K). 
                                 Lower = more rational (deterministic).
                                 Higher = more random.
        """
        super().__init__(graph, rounds, seed, payoff_matrix, initial_coop_rate)
        self.K = temperature

    def _update_strategies(self):
        """
        Update strategies using the Fermi rule:
        P(switch) = 1 / (1 + exp(-(payoff_neighbor - payoff_self) / K))
        """
        next_actions = {}
        
        for node_id in self.graph.nodes:
            # 1. Select one random neighbor
            neighbors = list(self.network.neighbors(node_id))
            if not neighbors:
                # Isolated node keeps current strategy
                next_actions[node_id] = self.agents[node_id].strategy.action
                continue
                
            target_neighbor = self.random.choice(neighbors)
            
            # 2. Compare Payoffs
            payoff_self = self.agents[node_id].payoff
            payoff_target = self.agents[target_neighbor].payoff
            delta = payoff_target - payoff_self
            
            # 3. Calculate Switching Probability (Fermi Function)
            # Clip delta/K to avoid overflow in exp() for very low K or high payoffs
            # If K is very small, delta/K can be huge.
            try:
                exponent = -delta / self.K
                # Limit exponent to avoid Math Overflow Error
                exponent = max(min(exponent, 700), -700)
                probability = 1 / (1 + math.exp(exponent))
            except ZeroDivisionError:
                # If K is 0, we act deterministically (Step function)
                probability = 1.0 if delta > 0 else 0.0
            
            # 4. Decide whether to switch
            if self.random.random() < probability:
                next_actions[node_id] = self.agents[target_neighbor].strategy.action
            else:
                next_actions[node_id] = self.agents[node_id].strategy.action

        # Apply updates synchronously
        for node_id, action in next_actions.items():
            self.agents[node_id].strategy.set_action(action)

# %%
class ReinforcementLearning(NetworkSimulation):
    """
    Strategy where agents learn from their own experience using Q-Learning.
    Agents maintain Q-values for 'C' and 'D' and update them based on rewards.
    """

    def __init__(self, graph, rounds=50, seed=None, payoff_matrix=payoff_matrices["Default"], 
                 learning_rate=0.1, epsilon=0.1, initial_q=0.0, initial_coop_rate=0.5):
        """
        Args:
            learning_rate (float): How fast new info overrides old info (alpha).
            epsilon (float): Probability of choosing a random action (exploration).
            initial_q (float): Starting value for Q-tables (optimistic vs pessimistic).
        """
        super().__init__(graph, rounds, seed, payoff_matrix, initial_coop_rate)
        self.alpha = learning_rate
        self.epsilon = epsilon
        
        # Initialize Q-tables for every agent: {node_id: {'C': val, 'D': val}}
        self.q_tables = {
            node_id: {"C": initial_q, "D": initial_q} 
            for node_id in self.graph.nodes
        }

    def _update_strategies(self):
        """
        1. Update Q-values based on the reward received in the *previous* round.
        2. Select the *next* action using epsilon-greedy policy.
        """
        
        # 1. Update Q-values (Learning Step)
        # We need to know what they *just* played and what they earned.
        for node_id, agent in self.agents.items():
            action_taken = agent.strategy.action
            reward_received = agent.payoff
            
            # Q(A) <- Q(A) + alpha * (Reward - Q(A))
            current_q = self.q_tables[node_id][action_taken]
            new_q = current_q + self.alpha * (reward_received - current_q)
            self.q_tables[node_id][action_taken] = new_q

        # 2. Select Next Action (Decision Step)
        next_actions = {}
        for node_id in self.agents:
            # Epsilon-Greedy: Explore with prob epsilon, Exploit otherwise
            if self.random.random() < self.epsilon:
                next_action = self.random.choice(["C", "D"])
            else:
                # Exploit: Choose action with highest Q-value
                q_vals = self.q_tables[node_id]
                if q_vals["C"] > q_vals["D"]:
                    next_action = "C"
                elif q_vals["D"] > q_vals["C"]:
                    next_action = "D"
                else:
                    # Tie-breaking
                    next_action = self.random.choice(["C", "D"])
            
            next_actions[node_id] = next_action

        # Apply updates synchronously
        for node_id, action in next_actions.items():
            self.agents[node_id].strategy.set_action(action)

# %% [markdown]
# ## Experiment visualization

# %%
def experiment(graph, model_class, steps=50, seed=42, interval=300, 
         payoff_matrix=None, is_grid=False, title=None):
    
    # Handle default mutable argument if necessary
    if payoff_matrix is None:
        # Assuming you have a global dict or imported default
        payoff_matrix = {} 

    # --- Configuration ---
    C_COOP = "#40B0A6"  
    C_DEFECT = "#FFBE6A"
    
    # 1. Initialize Model
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)
    
    # 2. Setup Figure
    fig, (ax_sim, ax_stats) = plt.subplots(
        2, 1, 
        figsize=(7, 9), 
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3} 
    )
    
    fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)
    cmap = ListedColormap([C_COOP, C_DEFECT])
    
    # 3. Initialize Stats Tracking & STATIC PLOT ELEMENTS
    # We initialize the lines with empty data ONCE.
    line_coop, = ax_stats.plot([], [], label='Collaborators', 
                  color=C_COOP, linewidth=2, solid_capstyle='round', 
                  path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.1), pe.Normal()])
    
    line_defect, = ax_stats.plot([], [], label='Defectors', 
                   color=C_DEFECT, linewidth=2)

    # Set formatting immediately (since we aren't clearing anymore)
    ax_stats.set_xlim(0, steps)
    ax_stats.set_ylim(0, 100)
    ax_stats.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_stats.set_ylabel("Population")
    ax_stats.grid(True, linestyle=':', alpha=0.4)
    # ax_stats.legend() # Optional: Add legend if you want it on the bottom graph too

    history_coop = []
    history_defect = []
    steps_range = []
    total_nodes = len(graph.nodes)

    def update_stats(frame):
        state = model.get_action_state()
        defector_count = sum(state.values())
        cooperator_count = len(state) - defector_count
        
        # Calculate Percentages
        pct_defect = (defector_count / total_nodes) * 100
        pct_coop = (cooperator_count / total_nodes) * 100
        
        history_defect.append(pct_defect)
        history_coop.append(pct_coop)
        steps_range.append(frame)
        
        # --- KEY FIX ---
        # Update the data of the existing lines instead of clearing/replotting
        line_coop.set_data(steps_range, history_coop)
        line_defect.set_data(steps_range, history_defect)
        
        # Return the artists that changed
        return [line_coop, line_defect]

    # 4. Define Visualization Logic
    viz_objects = {}

    if is_grid:
        grid_dim = int(math.isqrt(total_nodes))
        if grid_dim * grid_dim != total_nodes:
            raise ValueError(f"Graph has {total_nodes} nodes, not a square.")

        def get_grid_data():
            state = model.get_action_state()
            grid = [[0 for _ in range(grid_dim)] for _ in range(grid_dim)]
            for node_id, is_defector in state.items():
                row = node_id // grid_dim
                col = node_id % grid_dim
                grid[row][col] = is_defector
            return grid

        viz_objects['image'] = ax_sim.imshow(get_grid_data(), cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        
        def update_viz(frame):
            viz_objects['image'].set_data(get_grid_data())
            return [viz_objects['image']]

    else:
        pos = nx.spring_layout(graph, seed=seed)
        nodelist = list(graph.nodes())
        
        nx.draw_networkx_edges(graph, pos, ax=ax_sim, alpha=0.3, edge_color="gray")
        
        state = model.get_action_state()
        initial_colors = [state[n] for n in nodelist]

        viz_objects['nodes'] = nx.draw_networkx_nodes(
            graph, pos, nodelist=nodelist, node_color=initial_colors, 
            cmap=cmap, vmin=0, vmax=1, node_size=100, edgecolors="gray", ax=ax_sim
        )
        ax_sim.axis('off')
        ax_sim.set_aspect('auto')
        
        def update_viz(frame):
            current_state = model.get_action_state()
            new_colors = [current_state[n] for n in nodelist]
            viz_objects['nodes'].set_array(new_colors)
            return [viz_objects['nodes']]

    # 5. Shared Legend & Animation Setup
    legend_handles = [
        mpatches.Patch(color=C_COOP, label='Collaborators'), 
        mpatches.Patch(color=C_DEFECT, label='Defectors')
    ]
    ax_sim.legend(
        handles=legend_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.02),
        ncol=2, 
        frameon=False
    )
    
    ax_sim.set_title(f"{title if title else ''} (Step 0/{steps})")
    
    # Initial data load
    update_stats(0)

    def update(frame):
        if frame > 0:
            model.step()
        
        ax_sim.set_title(f"{title if title else ''} (Step {frame}/{steps})")
        
        # Collect modified artists from both subplots
        artists_stats = update_stats(frame)
        artists_sim = update_viz(frame)
        
        # Return combined list of artists (crucial for blit=True, good practice for blit=False)
        return artists_sim + artists_stats

    # Note: blit=True is generally recommended now that we have stable artists
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    
    return animation

# %% [markdown]
# ## Usage
# ### Strategies and matrices

# %%
strategies = {
    "Imitation Dynamics": ImitationDynamics,
    "Reinforcement Learning (0.2)": partial(ReinforcementLearning, learning_rate=0.2),
    "Reinforcement Learning (0.5)": partial(ReinforcementLearning, learning_rate=0.5),
    "Fermi (0.1)": partial(FermiPairwiseComparison, temperature=0.1),
    "Fermi (1.0)": partial(FermiPairwiseComparison, temperature=1.0),
}


# %%
payoff_matrices = { 
    "Default":          {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (0, 0) },
    "Canonical":        { ("C", "C"): (-1, -1), ("C", "D"): (-3, 0), ("D", "C"): (0, -3), ("D", "D"): (-2, -2) },
    "Friend or Foe":    { ("C", "C"): (1, 1), ("C", "D"): (0, 2), ("D", "C"): (2, 0), ("D", "D"): (0, 0) },
    "Snowdrift":        { ("C", "C"): (500, 500), ("C", "D"): (200, 800), ("D", "C"): (800, 200), ("D", "D"): (0, 0) },
    "Prisoners":        { ("C", "C"): (500, 500), ("C", "D"): (-200, 1200), ("D", "C"): (1200, -200), ("D", "D"): (0, 0) },
}


# %% [markdown]
# ### Network generation

# %%
size = 20
grid_graph = generate_graph("grid", size, m=size, periodic=True)

# %% [markdown]
# ### Visualization

# %%
def explore(strategy_keys, payoff_keys, graphs):
    for graph in graphs:
        for strategy in strategy_keys:
            model = strategies[strategy]
            for payoff in payoff_keys:
                matrix = payoff_matrices[payoff]
                ani = experiment(
                    graph=graph, 
                    model_class=model, 
                    steps=50, 
                    seed=42, 
                    is_grid=True,
                    title=f"{strategy} on a 2D grid using {payoff} matrix",
                    payoff_matrix=matrix,
                )
                display(ani)

# %%
# explore(strategies.keys(), payoff_matrices.keys(), graph=grid_graph)

# %%
# explore(strategies.keys(), ["Default"], graphs=[grid_graph])

# %%
# explore(["ImitationDynamics"], payoff_matrices.keys(), graph=[grid_graph])

# %%
# explore(strategies.keys(), ["Snowdrift"], graphs=[grid_graph])

# %%
ani = experiment(
    graph=grid_graph, 
    model_class=strategies["Reinforcement Learning (0.2)"], 
    steps=50, 
    seed=42, 
    is_grid=True,
    title=f"Reinforcement Learning on a 2D grid using Snowdrift matrix",
    payoff_matrix=payoff_matrices["Snowdrift"],
    )
ani.save("Test.gif")

# %%
for initial_coop in [0.3, 0.5, 0.7]:
    model = partial(ReinforcementLearning, learning_rate=0.2, initial_coop_rate=initial_coop)
    ani = experiment(
                    graph=grid_graph, 
                    model_class=model, 
                    steps=50, 
                    seed=42, 
                    is_grid=True,
                    title=f"Reinforcement Learning on a 2D grid using Snowdrift matrix (initial = {initial_coop})",
                    payoff_matrix=payoff_matrices["Snowdrift"],
                )
    display(ani)

# %%
# Erdos Renyi
# 1. Setup Parameters
n_nodes = 400       # Total agents (20x20 for clean visualization)
p_connection = 0.05 # 5% chance of an edge between any two nodes

# 2. Generate the Random Graph using NetworkX
# This creates a graph where edges are random, not a lattice.
random_graph = nx.erdos_renyi_graph(n=n_nodes, p=p_connection, seed=42)

# 3. Run the Demo
# We provide grid_size=20 so the demo can reshape the 1D list of 400 agents 
# into a 20x20 square for easier viewing.
ani = experiment(
    random_graph, 
    model_class=partial(ReinforcementLearning, learning_rate=0.2, initial_coop_rate=0.7), 
    steps=100, 
    seed=42, 
    payoff_matrix=payoff_matrices["Snowdrift"],
    title="Reinforcement Learning using Snowdrift on a Erods Renyi Network"
)
ani.save("Test-ER.gif")


