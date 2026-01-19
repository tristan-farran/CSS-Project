Using Google Gemini Pro

# Prompt 19/01/2025 18:52
Attached is our current codebase which contains GridImitationModel which is used within the demo function.

Help me deconstruct the GridImitationModel and demo function class to allow for it to be used with any graph that can be processed as a network, passed as a variable.

Also build it in such a way that we can easily add a child class with a different strategy (than the current _imitate_best_neighbor) while maintaining the overall general use. If possible make the main class as generic as possible and split the current imitate best neighbor functionality into a specific child.

When making changes output the entire function or class you deem necessary to change, explain the rationale behind the change based on my prompt and then give the new function (or class if an entire new class is generated).

## Answer
Here is the deconstruction of the `GridImitationModel` into a generic `NetworkSimulation` base class and a specific `ImitationDynamics` child class, along with the updated `demo` function.

### 1. `NetworkSimulation` (Base Class)

**Rationale:**
This class replaces the specific `GridImitationModel`. I have removed all grid-specific generation logic (like `generate_graph`) and fixed dimensions (`size`, `m`). Instead, it accepts any `networkx` graph object. It implements the core game loop (`_play_round`) but leaves the strategy update logic (`_update_strategies`) abstract. This allows you to plug in any graph topology and any update rule (e.g., Imitation, Q-Learning, Moran process).

```python
class NetworkSimulation:
    """
    Base class for running evolutionary games on any NetworkX graph.
    
    Handles:
    - Agent initialization
    - Interaction (play_round)
    - Generic stepping logic
    """

    def __init__(self, graph, rounds=50, seed=None, payoff_matrix=DEFAULT_PAYOFFS):
        self.rounds = rounds
        self.seed = seed
        self.payoff_matrix = payoff_matrix
        self.random = random.Random(seed)
        
        # Accept any pre-built networkx graph
        self.graph = graph
        self.network = Network(self.graph)
        
        self.agents = {}
        self.snapshots = []
        self._init_agents()

    def _init_agents(self):
        """Create agents with random starting actions for every node in the graph."""
        for node_id in self.graph.nodes:
            # We default to RandomActionStrategy, but this could also be parameterized
            strategy = RandomActionStrategy(self.random)
            self.agents[node_id] = Agent(node_id, strategy)

    def _reset_payoffs(self):
        """Reset payoffs for the new round."""
        for agent in self.agents.values():
            agent.payoff = 0.0

    def _play_round(self):
        """
        Compute payoffs for the current actions across all edges.
        This is topology-agnostic; it just iterates over edges.
        """
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
        """
        Define how agents update their strategies after a round.
        Must be implemented by subclasses (e.g., Imitation, Learning).
        """
        raise NotImplementedError("Subclasses must implement _update_strategies")

    def get_action_state(self):
        """Return a dictionary mapping node_id to current action (0 for C, 1 for D)."""
        return {
            node_id: (1 if agent.strategy.action == "D" else 0)
            for node_id, agent in self.agents.items()
        }

    def step(self):
        """Run one full step: play game -> update strategies -> snapshot."""
        self._play_round()
        self._update_strategies()
        self.snapshots.append(self.get_action_state())

    def run(self):
        """Run the simulation for the configured number of rounds."""
        for _ in range(self.rounds):
            self.step()

```

---

### 2. `ImitationDynamics` (Child Class)

**Rationale:**
This class encapsulates the specific "imitate best neighbor" logic. By isolating this, you can easily create other classes like `MoranDynamics` or `ReinforcementLearningDynamics` without rewriting the network management code. It inherits from `NetworkSimulation` and implements the abstract `_update_strategies` method.

```python
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

```

---

### 3. `demo` (Updated Function)

**Rationale:**
The `demo` function is now decoupled from graph generation.

1. **Arguments:** It now accepts a `graph` object and a `model_class`.
2. **Flexibility:** You can pass *any* networkx graph.
3. **Visualization:** The previous `demo` relied on `imshow` (which requires a 2D grid). To preserve the visual output for grids while allowing generic graphs, I added a logic check: if the graph looks like a grid (nodes are integers 0..N), we attempt to reshape the data into a grid for `imshow`. If `grid_size` is not provided, you might need a generic network visualizer (like `nx.draw`), but to keep the demo close to your original style, I retained the grid visualization logic specifically for grid-compatible graphs.

```python
def demo(graph, model_class=ImitationDynamics, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, grid_size=None):
    """
    Run a simulation demo.
    
    Args:
        graph: A NetworkX graph object.
        model_class: The class of simulation to run (default: ImitationDynamics).
        grid_size: (int) Optional. If provided, visualizes as a 2D grid. 
                   If None, visualization logic would need to change (omitted for brevity).
    """
    # Initialize the specific model class with the provided graph
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)

    # Visualization Setup
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Helper to convert generic node dict to 2D grid for imshow
    def get_grid_data():
        state = model.get_action_state()
        if grid_size:
            # Reconstruct 2D array from node dict assuming row-major integer node IDs
            grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            for node_id, is_defector in state.items():
                row = node_id // grid_size
                col = node_id % grid_size
                grid[row][col] = is_defector
            return grid
        else:
            # Fallback or different viz for non-grids could go here
            return list(state.values())

    cmap = ListedColormap(["#e6f2e4", "#d1495b"])
    
    # Initialize image
    initial_data = get_grid_data()
    image = ax.imshow(initial_data, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_title(f"Sim: {model_class.__name__}")
    ax.legend(handles=[mpatches.Patch(color='#e6f2e4', label='Collaborators'), 
                       mpatches.Patch(color='#d1495b', label='Defectors')],
              loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        model.step()
        image.set_data(get_grid_data())
        ax.set_xlabel(f"Step {frame + 1}/{steps}")
        return [image]

    logger.info("Starting demo for %s steps on %s", steps, model_class.__name__)
    animation = FuncAnimation(fig, update, frames=steps, interval=interval, blit=True, repeat=False)
    return animation

```

### Example Usage

You would now call the code like this, explicitly creating the graph first:

```python
# Create the graph explicitly
size = 20
grid_graph = generate_graph("grid", size, m=size, periodic=True)

# Pass graph to demo (providing grid_size ensures correct visualization)
ani = demo(grid_graph, model_class=ImitationDynamics, steps=50, seed=42, grid_size=size)
display(ani)

```


# Prompt 19/01/2025 18:56

Why does the following yield an initialized grid with step=1 being all defectors? prisoners = {
            ("C", "C"): (500, 500),
            ("C", "D"): (-200, 1200),
            ("D", "C"): (1200, -200),
            ("D", "D"): (0, 0),
        }
ani = demo(grid_graph, model_class=ImitationDynamics, steps=50, seed=42, grid_size=size, payoff_matrix=prisoners)
display(ani)

## Answer

This occurs because your **Temptation payoff (1200)** is so high that Defectors become mathematically unbeatable in almost any standard configuration.

Here is the breakdown of why the grid turns entirely red (Defectors) in a single step:

### 1. The Mathematical Cap

In a grid, every agent has exactly **4 neighbors**. This creates a "mathematical speed limit" for how much a Cooperator can possibly earn.

* **Max Cooperator Score:** Even if a Cooperator lives in a perfect utopia surrounded by 4 other Cooperators, their maximum possible score is .

### 2. The Defector Advantage

Now look at the Defector's payoff. They earn **1200** for every Cooperator they exploit.

* **One Victim:** If a Defector is next to just 1 Cooperator, they earn 1200.
* **Two Victims:** If a Defector is next to 2 Cooperators, they earn .

### 3. The "Invincibility" Threshold

Because **2400 (Defector with 2 victims) > 2000 (Perfect Cooperator)**, a Defector only needs to find **2 Cooperators** to score higher than the *best possible* Cooperator anywhere on the grid.

In a random initialization (roughly 50/50 split), statistically almost every Defector will have at least 2 Cooperator neighbors. This makes them "Super Defectors" with scores that no Cooperator can beat.

### 4. The Imitation Dynamic

The simulation uses the rule: *"Copy the neighbor with the highest score."*

1. Every agent looks at their neighbors.
2. They see these "Super Defectors" earning 2400 or 3600.
3. Even the happiest Cooperators (earning 2000) look at their Defector neighbor (earning 2400) and decide to switch strategies.
4. Result: **Total collapse of cooperation.**

### Summary

Your payoff matrix creates a scenario where **Defection is too lucrative.** For cooperation to survive, the Temptation payoff usually needs to be lower (often less than ), or the network needs clusters where Cooperators are shielded from seeing successful Defectors.

If you want to see a more interesting battle, try lowering the Temptation (D vs C) to **700 or 800**, or increasing the Reward (C vs C).

# Prompt
Add frame 0, so the initial configuration in the demo plotting.

## Answer
To include the initial configuration (Frame 0) in the animation, we need to make two small adjustments to the `demo` function:

1. **Increase Frame Count:** Set `frames=steps + 1` so the animation includes the 0th state plus all 50 simulation steps.
2. **Conditional Stepping:** Inside the `update` function, only call `model.step()` if `frame > 0`. This preserves the initial state for the first frame.

Here is the updated `demo` function:

```python
def demo(graph, model_class=ImitationDynamics, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, grid_size=None):
    """
    Run a simulation demo including the initial frame (Frame 0).
    """
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)

    fig, ax = plt.subplots(figsize=(6, 6))
    
    def get_grid_data():
        state = model.get_action_state()
        if grid_size:
            grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            for node_id, is_defector in state.items():
                row = node_id // grid_size
                col = node_id % grid_size
                grid[row][col] = is_defector
            return grid
        return list(state.values())

    cmap = ListedColormap(["#e6f2e4", "#d1495b"])
    
    # Initialize with Frame 0 data
    initial_data = get_grid_data()
    image = ax.imshow(initial_data, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_title(f"Sim: {model_class.__name__}")
    ax.legend(handles=[mpatches.Patch(color='#e6f2e4', label='Collaborators'), 
                       mpatches.Patch(color='#d1495b', label='Defectors')],
              loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        # Only advance the model if we are past the initial frame (0)
        if frame > 0:
            model.step()
        
        image.set_data(get_grid_data())
        ax.set_xlabel(f"Step {frame}/{steps}")
        return [image]

    logger.info("Starting demo for %s steps", steps)
    
    # frames=steps + 1 ensures we see 0 to 50 (51 total frames)
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=True, repeat=False)
    return animation

```

# Prompt
Provide an example that utilizes a random Erdos Renyi network using the already implemented functionality.

## Answer

To use an **Erdős-Rényi (Random) Graph** with the existing `demo` function, we essentially just swap out the graph generation step.

Since the `demo` function relies on `imshow` (a 2D heat map) for visualization, we will set the number of nodes to a perfect square (e.g., 400) and pass `grid_size=20`.

**Note on Visualization:** In the resulting animation, the "grid" you see is just a visual container for the 400 agents. Unlike the grid graph, **neighbors in the image are not necessarily neighbors in the network.** Interactions are happening across long-distance random connections.

```python
# %%
# 1. Setup Parameters
n_nodes = 400       # Total agents (20x20 for clean visualization)
p_connection = 0.05 # 5% chance of an edge between any two nodes

# 2. Generate the Random Graph using NetworkX
# This creates a graph where edges are random, not a lattice.
random_graph = nx.erdos_renyi_graph(n=n_nodes, p=p_connection, seed=42)

# 3. Run the Demo
# We provide grid_size=20 so the demo can reshape the 1D list of 400 agents 
# into a 20x20 square for easier viewing.
ani = demo(
    random_graph, 
    model_class=ImitationDynamics, 
    steps=50, 
    seed=10, 
    grid_size=20, # sqrt(400)
    payoff_matrix=DEFAULT_PAYOFFS
)

display(ani)

```

### What to expect

You will likely see the simulation resolve much faster than the Grid model.

* **Small World Effect:** Because random graphs have a short average path length (you are only a few hops from everyone else), "Defector" or "Cooperator" waves propagate through the population explosively rather than travelling slowly across a physical grid surface.

# Prompt
Provide the necessary changes to properly plot this Network using Networkx's plotting capabilities.

## Answer
To properly visualize generic networks (like Erdős-Rényi or Barabási-Albert) where nodes don't fit into a neat grid, we must switch from `ax.imshow` (heatmap) to `nx.draw` (node-link diagram).

Here is the updated `demo` function. It calculates a layout (position) for the nodes once and then updates only the node colors during the animation.

### Changes Made:

1. **Layout Calculation:** Added `pos` calculation (using `spring_layout` by default) to determine where nodes sit in 2D space.
2. **NetworkX Drawing:** Replaced `imshow` with `nx.draw_networkx_nodes` and `nx.draw_networkx_edges`.
3. **Stable Ordering:** We explicitly create a `nodelist` to ensure the color array always matches the correct nodes during updates.
4. **Efficient Updates:** The animation updates the `nodes` collection using `set_array` rather than redrawing the whole graph every frame.

### The Updated Code

```python
def demo(graph, model_class=ImitationDynamics, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, layout_func=nx.spring_layout):
    """
    Run a simulation demo using NetworkX plotting for generic graphs.
    """
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)

    # 1. Setup Layout
    # Compute positions once so nodes don't jump around during animation.
    # Use the provided seed for the layout so it's reproducible.
    if layout_func:
        pos = layout_func(graph, seed=seed)
    else:
        # Fallback if no layout provided
        pos = nx.spring_layout(graph, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 7))
    
    # 2. Setup Visualization Elements
    cmap = ListedColormap(["#e6f2e4", "#d1495b"])
    
    # Draw edges once (static background)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, edge_color="gray")
    
    # We need a fixed list of nodes to ensure colors match the correct node index
    nodelist = list(graph.nodes())
    
    # Get initial state (Frame 0)
    state = model.get_action_state()
    initial_colors = [state[n] for n in nodelist]

    # Draw nodes (returns a PathCollection we can update)
    nodes = nx.draw_networkx_nodes(
        graph, 
        pos, 
        nodelist=nodelist,
        node_color=initial_colors, 
        cmap=cmap, 
        vmin=0, 
        vmax=1, 
        node_size=100,
        edgecolors="gray", # Thin border around nodes
        ax=ax
    )
    
    # Legend
    ax.legend(handles=[mpatches.Patch(color='#e6f2e4', label='Collaborators'), 
                       mpatches.Patch(color='#d1495b', label='Defectors')],
              loc='upper right')
    ax.set_title(f"Sim: {model_class.__name__} (Step 0/{steps})")
    ax.axis('off')

    # 3. Animation Update Rule
    def update(frame):
        if frame > 0:
            model.step()
        
        # Fetch new states and map to the fixed nodelist
        current_state = model.get_action_state()
        new_colors = [current_state[n] for n in nodelist]
        
        # Update only the color array of the existing nodes
        nodes.set_array(new_colors)
        
        ax.set_title(f"Sim: {model_class.__name__} (Step {frame}/{steps})")
        return [nodes]

    logger.info("Starting demo for %s steps", steps)
    # blit=False is often more stable for PathCollections in notebooks
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    return animation

```

### Example Usage (Erdős-Rényi)

Now you can run the Random Graph example, and it will plot a proper "ball and stick" network graph instead of a misleading grid.

```python
# Create a random graph
random_graph = nx.erdos_renyi_graph(n=100, p=0.05, seed=42)

# Run demo (no grid_size needed anymore)
ani = demo(random_graph, model_class=ImitationDynamics, steps=50, seed=10)
display(ani)

```

# Prompt
Provide a demo function that both allows for complex networks as the simplified grid we had before with a grid boolean toggle

## Answer
Here is the unified `demo` function. It includes an `is_grid` toggle that automatically switches between the heatmap visualization (for grids) and the network graph visualization (for everything else).

### Key Features

* **`is_grid=True`:** Calculates the square root of the node count to determine dimensions and renders a clean 2D heatmap.
* **`is_grid=False`:** Renders the complex node-link diagram using a spring layout.
* **Frame 0 Support:** Includes the initial state in the animation logic.

```python
import math

def demo(graph, model_class=ImitationDynamics, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, is_grid=False):
    """
    Unified demo function for both Grid and Complex Networks.
    
    Args:
        graph: A NetworkX graph object.
        model_class: Simulation class (default: ImitationDynamics).
        steps: Number of simulation rounds.
        seed: Random seed for reproducibility.
        interval: Animation speed in ms.
        payoff_matrix: Dictionary of payoffs.
        is_grid: If True, visualizes as a 2D heatmap (requires square node count).
                 If False, visualizes as a node-link network graph.
    """
    # 1. Initialize Model
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)
    
    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = ListedColormap(["#e6f2e4", "#d1495b"])
    legend_handles = [
        mpatches.Patch(color='#e6f2e4', label='Collaborators'), 
        mpatches.Patch(color='#d1495b', label='Defectors')
    ]
    
    # 3. Define Visualization Logic based on Mode
    viz_objects = {}

    if is_grid:
        # --- Grid Visualization (Heatmap) ---
        n_nodes = len(graph.nodes)
        grid_dim = int(math.isqrt(n_nodes))
        
        if grid_dim * grid_dim != n_nodes:
            raise ValueError(f"Graph has {n_nodes} nodes, which is not a perfect square. Cannot view as grid.")

        def get_grid_data():
            state = model.get_action_state()
            # Convert dict {node_id: action} to 2D array
            # Assumes node_ids correspond to row-major order (0..N)
            grid = [[0 for _ in range(grid_dim)] for _ in range(grid_dim)]
            for node_id, is_defector in state.items():
                row = node_id // grid_dim
                col = node_id % grid_dim
                grid[row][col] = is_defector
            return grid

        # Draw initial frame
        viz_objects['image'] = ax.imshow(get_grid_data(), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        def update_viz(frame):
            viz_objects['image'].set_data(get_grid_data())
            return [viz_objects['image']]

    else:
        # --- Network Visualization (Node-Link) ---
        # Calculate positions once
        pos = nx.spring_layout(graph, seed=seed)
        nodelist = list(graph.nodes())
        
        # Draw static edges
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, edge_color="gray")
        
        # Get initial colors
        state = model.get_action_state()
        initial_colors = [state[n] for n in nodelist]

        # Draw nodes
        viz_objects['nodes'] = nx.draw_networkx_nodes(
            graph, pos, nodelist=nodelist, node_color=initial_colors, 
            cmap=cmap, vmin=0, vmax=1, node_size=100, edgecolors="gray", ax=ax
        )
        ax.axis('off')
        
        def update_viz(frame):
            # Fetch new states mapped to fixed nodelist
            current_state = model.get_action_state()
            new_colors = [current_state[n] for n in nodelist]
            viz_objects['nodes'].set_array(new_colors)
            return [viz_objects['nodes']]

    # 4. Common Animation Logic
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_title(f"Sim: {model_class.__name__} (Step 0/{steps})")

    def update(frame):
        if frame > 0:
            model.step()
        
        ax.set_title(f"Sim: {model_class.__name__} (Step {frame}/{steps})")
        return update_viz(frame)

    logger.info("Starting %s demo for %s steps", "Grid" if is_grid else "Network", steps)
    
    # Blit must be False for reliable networkx updates in some environments
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    return animation

```

### Example 1: Using it as a Grid

```python
# Create a Grid Graph
grid_graph = nx.grid_2d_graph(20, 20, periodic=True)
# Relabel to integers for compatibility with the simplistic row/col logic
grid_graph = nx.convert_node_labels_to_integers(grid_graph)

ani = demo(grid_graph, is_grid=True, steps=50, seed=42)
display(ani)

```

### Example 2: Using it as a Complex Network

```python
# Create a Random Graph
random_graph = nx.erdos_renyi_graph(100, 0.05, seed=42)

ani = demo(random_graph, is_grid=False, steps=50, seed=42)
display(ani)

```

Would you like me to refine the `update` logic to handle specific edge-case topologies (like Scale-Free networks) differently?

# Prompt
Suggest alternative strategies to the ImitationDynamics, do not implement any of them yet. Just describe their behaviour.

## Answer

Here are four alternative dynamics strategies commonly used in Evolutionary Game Theory. These would be implemented as new child classes of `NetworkSimulation`, replacing the deterministic `ImitationDynamics`.

### 1. Probabilistic Pairwise Comparison (The "Fermi Rule")

This is a "bounded rationality" model. Instead of scanning *all* neighbors and strictly copying the best, an agent picks **one** neighbor at random and compares payoffs.

* **Behavior:** The agent calculates the payoff difference between themselves and the chosen neighbor. They switch to the neighbor's strategy with a probability defined by a sigmoid (Fermi) function.
* **Key Feature:** It introduces **noise** (temperature). Even if a neighbor is doing slightly better, the agent might not switch. Conversely, there is a small, non-zero chance an agent might copy a *worse* performing neighbor (modeling mistakes or exploration). This prevents the "instant crystallization" you saw with the deterministic imitation.

### 2. Myopic Best Response (Rational Agents)

This abandons "imitation" entirely in favor of calculation. Agents don't care how successful their neighbors are; they only care about what their neighbors are *doing*.

* **Behavior:** In every round, the agent looks at the current actions of all its neighbors. It calculates: *"If I play C next turn, what is my payoff? If I play D, what is my payoff?"* It then chooses the action that maximizes its immediate reward against the current configuration.
* **Key Feature:** These agents are "selfish optimizers." This dynamic often leads to Nash Equilibrium states very quickly. Unlike imitation, a Defector surrounded by Cooperators might switch to Cooperation if the payoff matrix encourages coordination (like in the Stag Hunt game), or they might stick to Defection if it's the dominant strategy (like in Prisoner's Dilemma).

### 3. Reinforcement Learning (Q-Learning)

This models individual learning rather than social learning. Agents don't look at their neighbors' scores at all; they only look at their own history.

* **Behavior:** Each agent maintains a generic "Q-value" (expected reward) for "C" and "D". If they play "C" and get a high reward, they update their internal Q-value for "C" upwards. They pick their next action based on which Q-value is currently higher (often with a small "epsilon" chance of random exploration).
* **Key Feature:** Agents develop unique personalities based on their specific interaction history. A Cooperator in a hostile neighborhood learns that "C is bad" and switches to "D". A Cooperator in a friendly neighborhood learns "C is good" and sticks with it.

### 4. The Moran Process (Survival of the Fittest)

This is a biological evolution model rather than a social strategy model.

* **Behavior:** Selection is stochastic and global (or local). Instead of every node updating simultaneously, updates happen asynchronously.
1. A node is chosen for "reproduction" proportional to its fitness (payoff).
2. Its strategy is cloned onto a random neighbor, "killing" the neighbor's current strategy.


* **Key Feature:** This is highly probabilistic. A weak Defector can get lucky and reproduce, or a strong Cooperator can get unlucky and be replaced. It is useful for studying the "fixation probability"—the chance that a single mutant strategy eventually takes over the entire graph.

# Prompt

Implement option 1, the Fermi Rule Pairwise Comparison first. Do this in a similar fashion to how you implemented the ImitationDynamics utilizing subclasses.

## Answer

Here is the implementation of the `FermiPairwiseComparison` class.

### `FermiPairwiseComparison` Implementation

**Rationale:**
This strategy introduces stochasticity (randomness) into the decision-making process.

1. **Selection:** Instead of scanning *all* neighbors, an agent picks **one** neighbor at random.
2. **Comparison:** The agent calculates the payoff difference: .
3. **Probability:** The probability of switching to the neighbor's strategy is given by the Fermi function:



where  (temperature) controls the "noise" or "irrationality."
* **High  (High Noise):** Agents switch almost randomly, regardless of payoff (approaches 50/50 as ).
* **Low  (Low Noise):** Agents act almost deterministically, only switching if the neighbor is strictly better (approaches step function as ).



This prevents the "instant crystallization" of the grid and allows for more dynamic equilibrium states.

```python
import math

class FermiPairwiseComparison(NetworkSimulation):
    """
    Strategy where agents compare payoffs with a single random neighbor
    and switch strategies probabilistically based on the Fermi function.
    """
    
    def __init__(self, graph, rounds=50, seed=None, payoff_matrix=DEFAULT_PAYOFFS, temperature=0.1):
        """
        Args:
            temperature (float): Controls the noise level (K). 
                                 Lower = more rational (deterministic).
                                 Higher = more random.
        """
        super().__init__(graph, rounds, seed, payoff_matrix)
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

```

### Example Usage

To verify the difference, use a slightly higher temperature (e.g., `0.5` or `1.0`) to see "noisy" transitions.

Since `FermiPairwiseComparison` takes an extra argument (`temperature`) that `NetworkSimulation` does not, we use a `lambda` to inject the parameter when passing it to `demo`.

```python
# Create a grid graph
grid_graph = nx.grid_2d_graph(20, 20, periodic=True)
grid_graph = nx.convert_node_labels_to_integers(grid_graph)

# Run the Fermi simulation with K=0.5
ani = demo(
    grid_graph, 
    model_class=lambda g, **kwargs: FermiPairwiseComparison(g, temperature=0.5, **kwargs),
    steps=50, 
    seed=42, 
    is_grid=True,
    payoff_matrix=DEFAULT_PAYOFFS
)

display(ani)

```

# Prompt
Which strategy most closely resembles real life interactions of cooperation?

## Answer
The most realistic strategy for modeling human cooperation is generally considered to be **Reinforcement Learning (Q-Learning)** or **probabilistic bounded rationality (like the Fermi Rule)**, rather than strict Imitation Dynamics.

While "Imitation of the Best" (the first model you used) is powerful for simple biological organisms or viral spread, it often fails to capture the nuance of human social interaction. Real-life human cooperation is usually a mix of habit, learning from experience, and occasional irrationality.

### Why Reinforcement Learning (Q-Learning) is often most realistic:

* **Personal Experience vs. Peer Pressure:** In real life, you don't always know what your neighbors are earning. You often don't know if your neighbor's success is due to their strategy or luck. Instead, you rely on *your own* history: "I cooperated last time and got burned, so I won't do it again." Q-Learning models this internal memory.
* **Bounded Rationality:** Humans are not perfect optimizers. We develop "heuristics" or habits. If a strategy works "good enough" (satisficing), we stick with it, even if it's not mathematically optimal.
* **Distinct Personalities:** Q-Learning allows agents to develop different "personalities" based on their unique history. One agent might become a "grudger" because they were exploited early on, while another becomes a "saint" because they happened to interact with other cooperators initially.

### Why the Fermi Rule (Pairwise Comparison) is a strong contender:

* **Noisy Decision Making:** The Fermi rule introduces "temperature" (noise). In real economies, people sometimes make mistakes or act irrationally (e.g., trying a new strategy just out of curiosity, or failing to switch to a better one due to inertia).
* **Social Comparison:** It models the very human trait of "keeping up with the Joneses." We tend to compare ourselves to specific peers rather than the global maximum. If your neighbor drives a slightly better car, you might switch strategies, but if the difference is negligible, you likely won't bother.

### Summary Comparison

| Strategy | Realism Factor | Why? |
| --- | --- | --- |
| **Imitation (Best Neighbor)** | Low for humans | Assumes perfect information about others' success and zero loyalty to one's own past habits. Better for bacteria or simple firms. |
| **Fermi Rule (Pairwise)** | **Medium-High** | Captures "social comparison" and "bounded rationality" (we are lazy optimizers who sometimes make mistakes). |
| **Reinforcement Learning** | **High** | Captures "learning from experience." Real humans care more about their own pain/pleasure history than abstract neighbor stats. |

If you are trying to model a social network (like Twitter/X or a village), **Fermi Pairwise Comparison** is likely the best balance of simplicity and realism. If you are modeling complex negotiation or long-term trust (like business partners), **Reinforcement Learning** is superior.

# Prompt
Implement reinforcement learning

## Answer
Here is the implementation of the `ReinforcementLearning` (Q-Learning) strategy as a child class of `NetworkSimulation`.

### `ReinforcementLearning` Implementation

**Rationale:**
This strategy shifts from "social learning" (copying neighbors) to "individual learning" (learning from personal history).

1. **Memory (-Table):** Each agent maintains a -value for "C" (Cooperate) and "D" (Defect). This represents their *estimated* future reward for that action.
2. **Decision (-Greedy):**
* **Exploitation ():** Pick the action with the highest -value.
* **Exploration ():** Pick a random action to discover new possibilities.


3. **Update Rule:** After receiving a payoff (), the agent updates their belief about the chosen action ():



where  is the learning rate.

```python
class ReinforcementLearning(NetworkSimulation):
    """
    Strategy where agents learn from their own experience using Q-Learning.
    Agents maintain Q-values for 'C' and 'D' and update them based on rewards.
    """

    def __init__(self, graph, rounds=50, seed=None, payoff_matrix=DEFAULT_PAYOFFS, 
                 learning_rate=0.1, epsilon=0.1, initial_q=0.0):
        """
        Args:
            learning_rate (float): How fast new info overrides old info (alpha).
            epsilon (float): Probability of choosing a random action (exploration).
            initial_q (float): Starting value for Q-tables (optimistic vs pessimistic).
        """
        super().__init__(graph, rounds, seed, payoff_matrix)
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

```

### Example Usage

To use this, we inject the class and its specific parameters (like `learning_rate`) via a lambda function into the `demo`.

```python
# Create grid graph
grid_graph = nx.grid_2d_graph(20, 20, periodic=True)
grid_graph = nx.convert_node_labels_to_integers(grid_graph)

# Run RL Simulation
ani = demo(
    grid_graph, 
    model_class=lambda g, **kwargs: ReinforcementLearning(
        g, 
        learning_rate=0.1, 
        epsilon=0.05,    # 5% chance of random exploration
        initial_q=5.0,   # "Optimistic initialization" encourages trying both early
        **kwargs
    ),
    steps=100,           # RL often needs more steps to converge
    seed=42, 
    is_grid=True,
    payoff_matrix=DEFAULT_PAYOFFS
)

display(ani)

```

# Prompt

How to extract the model_class into a separate variable?

## Answer

To extract the `model_class` into a separate variable, you can assign the lambda function (or a `partial` function) to a variable name before passing it to `demo`. This makes the `demo` call much cleaner and easier to read.

Here are two ways to do this:

### Option 1: Using `lambda` (Simple)

This is exactly what we did inline, just saved to a variable first.

```python
# 1. Define the strategy configuration
# We capture 'g' (graph) and '**kwargs' (rounds, seed, payoff) and pass them through
rl_config = lambda g, **kwargs: ReinforcementLearning(
    g, 
    learning_rate=0.1, 
    epsilon=0.05, 
    initial_q=5.0, 
    **kwargs
)

# 2. Pass the variable to demo
ani = demo(
    grid_graph, 
    model_class=rl_config,  # <--- Clean variable here
    steps=100, 
    seed=42, 
    is_grid=True
)
display(ani)

```

### Option 2: Using `functools.partial` (Cleaner/Standard)

The `partial` function is the standard Pythonic way to "pre-fill" arguments for a class or function. It handles the passing of the graph and other arguments automatically without writing `lambda g, **kwargs`.

```python
from functools import partial

# 1. Create a partial class with your specific settings pre-filled
fermi_config = partial(FermiPairwiseComparison, temperature=0.5)

# 2. Pass it to demo
# The demo function will call fermi_config(graph, rounds=..., seed=...) automatically
ani = demo(
    grid_graph, 
    model_class=fermi_config, 
    steps=50, 
    seed=42, 
    is_grid=True
)
display(ani)

```

### Why do this?

* **Reusability:** You can reuse `fermi_config` across multiple demos without retyping the parameters.
* **Readability:** Your `demo(...)` call isn't cluttered with hyperparameter details.
* **Experimentation:** You can easily create list of configs to loop over:
```python
strategies = [
    partial(FermiPairwiseComparison, temperature=0.1),
    partial(FermiPairwiseComparison, temperature=1.0),
    partial(ReinforcementLearning, learning_rate=0.2)
]

```

# Prompt
Attached is the updated code. Expand the experiment function with the following features:
1. Move the legend to be to the right side outside of the main plot area, instead of inside the grid. Only necessary for the grid.  
2. Add a subplot on the bottom that tracks the number of Collaborating and Defecting nodes over time in a lineplot where a new point gets added during each step.

## Answer

Here is the updated `experiment` function. I have modified it to include a secondary subplot for tracking statistics and adjusted the legend placement for grid visualizations.

### Changes Made

1. **Dual Subplots:** The figure now initializes with two subplots (`nrows=2`) using `gridspec_kw={'height_ratios': [4, 1]}`. This gives the main simulation 80% of the vertical space and the stats tracker 20%.
2. **External Legend:** When `is_grid=True`, the legend is anchored outside the plot area (`bbox_to_anchor=(1, 0.5)`) to the right, ensuring it doesn't obscure the grid cells.
3. **Real-Time Line Plot:** I added a `update_stats` helper function that runs every frame. It counts the current Cooperators/Defectors and updates the line plot on the bottom axes (`ax_stats`).

```python
# %%
def experiment(graph, model_class, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, is_grid=False, title=None):
    """
    Unified experiment function with real-time population tracking.
    
    Args:
        graph: A NetworkX graph object.
        model_class: Simulation class
        steps: Number of simulation rounds.
        seed: Random seed for reproducibility.
        interval: Animation speed in ms.
        payoff_matrix: Dictionary of payoffs.
        is_grid: If True, visualizes as a 2D heatmap.
                 If False, visualizes as a node-link network graph.
        title: Optional title for the plot.
    """
    # 1. Initialize Model
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)
    
    # 2. Setup Figure with 2 Subplots (Top: Sim, Bottom: Stats)
    # height_ratios=[4, 1] makes the simulation view much larger than the line plot
    fig, (ax_sim, ax_stats) = plt.subplots(
        2, 1, 
        figsize=(7, 9), 
        gridspec_kw={'height_ratios': [4, 1]}
    )
    
    cmap = ListedColormap(["#e6f2e4", "#d1495b"])
    
    # Legend setup
    legend_handles = [
        mpatches.Patch(color='#e6f2e4', label='Collaborators'), 
        mpatches.Patch(color='#d1495b', label='Defectors')
    ]
    
    # 3. Initialize Stats Tracking
    # We'll store counts here
    history_coop = []
    history_defect = []
    steps_range = []

    def update_stats(frame):
        state = model.get_action_state()
        defectors = sum(state.values())
        cooperators = len(state) - defectors
        
        history_defect.append(defectors)
        history_coop.append(cooperators)
        steps_range.append(frame)
        
        ax_stats.clear()
        ax_stats.plot(steps_range, history_coop, label='Collaborators', color='green', alpha=0.7)
        ax_stats.plot(steps_range, history_defect, label='Defectors', color='red', alpha=0.7)
        
        ax_stats.set_xlim(0, steps)
        ax_stats.set_ylim(0, len(graph.nodes))
        ax_stats.set_xlabel("Step")
        ax_stats.set_ylabel("Count")
        ax_stats.legend(loc='center right', fontsize='small')
        ax_stats.grid(True, linestyle='--', alpha=0.5)

    # 4. Define Visualization Logic based on Mode
    viz_objects = {}

    if is_grid:
        # --- Grid Visualization (Heatmap) ---
        n_nodes = len(graph.nodes)
        grid_dim = int(math.isqrt(n_nodes))
        
        if grid_dim * grid_dim != n_nodes:
            raise ValueError(f"Graph has {n_nodes} nodes, which is not a perfect square. Cannot view as grid.")

        def get_grid_data():
            state = model.get_action_state()
            # Convert dict {node_id: action} to 2D array
            grid = [[0 for _ in range(grid_dim)] for _ in range(grid_dim)]
            for node_id, is_defector in state.items():
                row = node_id // grid_dim
                col = node_id % grid_dim
                grid[row][col] = is_defector
            return grid

        # Draw initial frame
        viz_objects['image'] = ax_sim.imshow(get_grid_data(), cmap=cmap, vmin=0, vmax=1)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        
        # Move legend outside to the right
        ax_sim.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
        
        def update_viz(frame):
            viz_objects['image'].set_data(get_grid_data())
            return [viz_objects['image']]

    else:
        # --- Network Visualization (Node-Link) ---
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
        
        # Standard legend inside for network view (usually enough space)
        ax_sim.legend(handles=legend_handles, loc='upper right')
        
        def update_viz(frame):
            current_state = model.get_action_state()
            new_colors = [current_state[n] for n in nodelist]
            viz_objects['nodes'].set_array(new_colors)
            return [viz_objects['nodes']]

    # 5. Common Animation Logic
    ax_sim.set_title(f"{title if title else ''} (Step 0/{steps})")
    
    # Initialize stats for step 0
    update_stats(0)

    def update(frame):
        if frame > 0:
            model.step()
        
        ax_sim.set_title(f"{title if title else ''} (Step {frame}/{steps})")
        update_stats(frame)
        return update_viz(frame) + list(ax_stats.lines)

    logger.info("Starting %s demo for %s steps", "Grid" if is_grid else "Network", steps)
    
    plt.tight_layout()
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    return animation

```

# Prompt

Visualization issues:
1. The subplot does not use the custom cmap for collaborators and defectors. 
2. There are now two legends. For both versions I want the legend with patches anchored to be centered below the subplot while inline

## Answer

Here is the updated `experiment` function.

### Changes Made

1. **Consistent Colors:** I defined `C_COOP` and `C_DEFECT` variables with your specific hex codes and applied them to both the heatmap/network nodes **and** the line plot (which previously used default green/red).
2. **Unified Legend Placement:** The main legend (patches) is now created once, positioned centered below the simulation plot (`bbox_to_anchor=(0.5, -0.02)`), and laid out horizontally (`ncol=2`).
3. **Layout Adjustments:** I increased the vertical spacing in `gridspec_kw` (`'hspace': 0.3`) to make room for the legend between the two plots.

```python
# %%
def experiment(graph, model_class, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, is_grid=False, title=None):
    """
    Unified experiment function with consistent styling and positioning.
    """
    # --- Configuration ---
    # Define colors centrally to ensure the line plot matches the grid/network
    C_COOP = "#e6f2e4"   # Light green
    C_DEFECT = "#d1495b" # Red
    
    # 1. Initialize Model
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)
    
    # 2. Setup Figure
    # hspace=0.3 gives room for the legend between the top and bottom plot
    fig, (ax_sim, ax_stats) = plt.subplots(
        2, 1, 
        figsize=(7, 9), 
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3} 
    )
    
    cmap = ListedColormap([C_COOP, C_DEFECT])
    
    # 3. Initialize Stats Tracking
    history_coop = []
    history_defect = []
    steps_range = []

    def update_stats(frame):
        state = model.get_action_state()
        defectors = sum(state.values())
        cooperators = len(state) - defectors
        
        history_defect.append(defectors)
        history_coop.append(cooperators)
        steps_range.append(frame)
        
        ax_stats.clear()
        
        # Use the exact same hex colors as the map
        # Added edgecolors to lines to make light colors (like the coop green) visible against white
        ax_stats.plot(steps_range, history_coop, label='Collaborators', 
                      color=C_COOP, linewidth=2, solid_capstyle='round', path_effects=[mpatches.pe.Stroke(linewidth=3, foreground='black', alpha=0.1), mpatches.pe.Normal()])
        
        ax_stats.plot(steps_range, history_defect, label='Defectors', 
                      color=C_DEFECT, linewidth=2)
        
        ax_stats.set_xlim(0, steps)
        ax_stats.set_ylim(0, len(graph.nodes))
        ax_stats.set_ylabel("Count")
        
        # We don't really need a second legend here if the top one is clear, 
        # but a small one helps identify the lines specifically.
        ax_stats.legend(loc='center right', fontsize='x-small', framealpha=0.8)
        ax_stats.grid(True, linestyle=':', alpha=0.4)

    # 4. Define Visualization Logic
    viz_objects = {}

    if is_grid:
        # --- Grid Visualization ---
        n_nodes = len(graph.nodes)
        grid_dim = int(math.isqrt(n_nodes))
        
        if grid_dim * grid_dim != n_nodes:
            raise ValueError(f"Graph has {n_nodes} nodes, not a square.")

        def get_grid_data():
            state = model.get_action_state()
            grid = [[0 for _ in range(grid_dim)] for _ in range(grid_dim)]
            for node_id, is_defector in state.items():
                row = node_id // grid_dim
                col = node_id % grid_dim
                grid[row][col] = is_defector
            return grid

        viz_objects['image'] = ax_sim.imshow(get_grid_data(), cmap=cmap, vmin=0, vmax=1)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        
        def update_viz(frame):
            viz_objects['image'].set_data(get_grid_data())
            return [viz_objects['image']]

    else:
        # --- Network Visualization ---
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
        
        def update_viz(frame):
            current_state = model.get_action_state()
            new_colors = [current_state[n] for n in nodelist]
            viz_objects['nodes'].set_array(new_colors)
            return [viz_objects['nodes']]

    # 5. Shared Legend & Animation Setup
    # Create one central legend anchored below the main plot
    legend_handles = [
        mpatches.Patch(color=C_COOP, label='Collaborators'), 
        mpatches.Patch(color=C_DEFECT, label='Defectors')
    ]
    ax_sim.legend(
        handles=legend_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.02), # Just below the simulation axis
        ncol=2, 
        frameon=False
    )
    
    ax_sim.set_title(f"{title if title else ''} (Step 0/{steps})")
    update_stats(0)

    def update(frame):
        if frame > 0:
            model.step()
        
        ax_sim.set_title(f"{title if title else ''} (Step {frame}/{steps})")
        update_stats(frame)
        return update_viz(frame) + list(ax_stats.lines)

    logger.info("Starting %s demo for %s steps", "Grid" if is_grid else "Network", steps)
    
    # We do NOT use tight_layout here because we manually set hspace and legend positions
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    return animation

```

# Prompt 

Provided stack error

## Answer

The error occurs because I incorrectly tried to access `patheffects` through `mpatches`. It needs to be imported directly from `matplotlib`.

Here is the corrected `experiment` function with the proper import and usage.

```python
import matplotlib.patheffects as pe  # Import PathEffects

def experiment(graph, model_class, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, is_grid=False, title=None):
    """
    Unified experiment function with consistent styling and positioning.
    """
    # --- Configuration ---
    C_COOP = "#e6f2e4"   # Light green
    C_DEFECT = "#d1495b" # Red
    
    # 1. Initialize Model
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)
    
    # 2. Setup Figure
    fig, (ax_sim, ax_stats) = plt.subplots(
        2, 1, 
        figsize=(7, 9), 
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3} 
    )
    
    cmap = ListedColormap([C_COOP, C_DEFECT])
    
    # 3. Initialize Stats Tracking
    history_coop = []
    history_defect = []
    steps_range = []

    def update_stats(frame):
        state = model.get_action_state()
        defectors = sum(state.values())
        cooperators = len(state) - defectors
        
        history_defect.append(defectors)
        history_coop.append(cooperators)
        steps_range.append(frame)
        
        ax_stats.clear()
        
        # Corrected PathEffects usage using 'pe' imported above
        ax_stats.plot(steps_range, history_coop, label='Collaborators', 
                      color=C_COOP, linewidth=2, solid_capstyle='round', 
                      path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.1), pe.Normal()])
        
        ax_stats.plot(steps_range, history_defect, label='Defectors', 
                      color=C_DEFECT, linewidth=2)
        
        ax_stats.set_xlim(0, steps)
        ax_stats.set_ylim(0, len(graph.nodes))
        ax_stats.set_ylabel("Count")
        
        ax_stats.legend(loc='center right', fontsize='x-small', framealpha=0.8)
        ax_stats.grid(True, linestyle=':', alpha=0.4)

    # 4. Define Visualization Logic
    viz_objects = {}

    if is_grid:
        # --- Grid Visualization ---
        n_nodes = len(graph.nodes)
        grid_dim = int(math.isqrt(n_nodes))
        
        if grid_dim * grid_dim != n_nodes:
            raise ValueError(f"Graph has {n_nodes} nodes, not a square.")

        def get_grid_data():
            state = model.get_action_state()
            grid = [[0 for _ in range(grid_dim)] for _ in range(grid_dim)]
            for node_id, is_defector in state.items():
                row = node_id // grid_dim
                col = node_id % grid_dim
                grid[row][col] = is_defector
            return grid

        viz_objects['image'] = ax_sim.imshow(get_grid_data(), cmap=cmap, vmin=0, vmax=1)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        
        def update_viz(frame):
            viz_objects['image'].set_data(get_grid_data())
            return [viz_objects['image']]

    else:
        # --- Network Visualization ---
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
    update_stats(0)

    def update(frame):
        if frame > 0:
            model.step()
        
        ax_sim.set_title(f"{title if title else ''} (Step {frame}/{steps})")
        update_stats(frame)
        return update_viz(frame) + list(ax_stats.lines)

    logger.info("Starting %s demo for %s steps", "Grid" if is_grid else "Network", steps)
    
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    return animation

```

# Prompt

I have removed the ax_stat.legend(). What changes are required to make the y axis in the ax_stat into a percentage instead of a count?

## Answer

To change the y-axis to a percentage, we need to modify the `update_stats` function to calculate the percentage relative to the total number of nodes, set the y-limits to `(0, 100)`, and update the axis label.

Here is the revised `experiment` function:

```python
import matplotlib.patheffects as pe

def experiment(graph, model_class, steps=50, seed=42, interval=300, 
         payoff_matrix=DEFAULT_PAYOFFS, is_grid=False, title=None):
    """
    Unified experiment function with percentage-based tracking.
    """
    # --- Configuration ---
    C_COOP = "#e6f2e4"   # Light green
    C_DEFECT = "#d1495b" # Red
    
    # 1. Initialize Model
    model = model_class(graph, rounds=steps, seed=seed, payoff_matrix=payoff_matrix)
    
    # 2. Setup Figure
    fig, (ax_sim, ax_stats) = plt.subplots(
        2, 1, 
        figsize=(7, 9), 
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3} 
    )
    
    cmap = ListedColormap([C_COOP, C_DEFECT])
    
    # 3. Initialize Stats Tracking
    history_coop = []
    history_defect = []
    steps_range = []
    
    # Pre-calculate total nodes for percentage calculation
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
        
        ax_stats.clear()
        
        # Plot Percentages
        ax_stats.plot(steps_range, history_coop, label='Collaborators', 
                      color=C_COOP, linewidth=2, solid_capstyle='round', 
                      path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.1), pe.Normal()])
        
        ax_stats.plot(steps_range, history_defect, label='Defectors', 
                      color=C_DEFECT, linewidth=2)
        
        ax_stats.set_xlim(0, steps)
        ax_stats.set_ylim(0, 100)  # Fixed to 0-100%
        ax_stats.set_ylabel("Population (%)") # Updated Label
        
        # Legend removed as requested
        ax_stats.grid(True, linestyle=':', alpha=0.4)

    # 4. Define Visualization Logic
    viz_objects = {}

    if is_grid:
        # --- Grid Visualization ---
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

        viz_objects['image'] = ax_sim.imshow(get_grid_data(), cmap=cmap, vmin=0, vmax=1)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        
        def update_viz(frame):
            viz_objects['image'].set_data(get_grid_data())
            return [viz_objects['image']]

    else:
        # --- Network Visualization ---
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
    update_stats(0)

    def update(frame):
        if frame > 0:
            model.step()
        
        ax_sim.set_title(f"{title if title else ''} (Step {frame}/{steps})")
        update_stats(frame)
        return update_viz(frame) + list(ax_stats.lines)

    logger.info("Starting %s demo for %s steps", "Grid" if is_grid else "Network", steps)
    
    animation = FuncAnimation(fig, update, frames=steps + 1, interval=interval, blit=False, repeat=False)
    return animation

```

# Prompt

My current code has the following function: 
class RandomActionStrategy(ActionStrategy):
    """Start with a random action and then behave like ActionStrategy."""

    def __init__(self, rng=None):
        rng = rng or random.Random()
        super().__init__(rng.choice(["C", "D"]))


How can I change this so a certain rate of C can be passed as a paremeter? Currently it is 50/50 always

## Answer

Provided code.