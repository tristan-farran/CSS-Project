**Agent: ChatGPT 5.2 (Codex)**

---

##### Prompt:
Create a Python project skeleton for an agent-based Iterated Prisoner’s Dilemma on networks.

Requirements:
    Use Networkx for graph representation and generation, mesa for the agent based piece, and axelrod for the prisoner's dilemma strategies.
    Keep the code modular: separate network, agents, strategies, simulation, metrics, utils.
    Add docstrings and simple logging.

##### Result:
Created a new modular project skeleton under /Users/tristanfarran/Documents/Code/2026/Code/CSS-Project/ipd_network with separate modules for network generation, agents, strategies, simulation, metrics, and utils, plus basic packaging/docs. The core model is in simulation.py, agents wrap Axelrod strategies in agents.py, and graph creation lives in network.py, all with type hints, docstrings, and simple logging via utils.py.

##### Evaluation:
Looks like a robust skeleton, though significantly too complicated for our purposes and includes abstractions I am unfamiliar with, I will add the basic functionality and then ask it to simplify.

---

##### Prompt:
Implement the core interfaces for the framework:
    Strategy base class: decides action {C,D} based on history and neighbours history.
    Agent class: has id, strategy, payoff, and stores interaction history per neighbor.
    Network wrapper class: holds a networkx graph and provides neighbor queries.

Design constraints:
    Strategies must be swappable without changing simulator code.
    Network type must be swappable without changing agent code.
    Provide minimal unit tests using pytest verifying correctness

##### Result:
Implemented a clean core interface layer that keeps strategies and network implementations swappable, plus minimal pytest coverage for strategy swap, agent history/payoff, and network neighbor queries. Core types live in core.py, are exported from the package in __init__.py, and tests are in test_core.py.

##### Evaluation:
Looks like a reasonable first pass, still quite confusing to read as no experience with Mesa or Axelrod, and it overdid the complexity. Time to simplify.

---

##### Prompt:
Simplify everything as much as possible, while retaining functionality and robustness, in order to make this as compact, and above all readable, so I can start editing it with minimal confusion.

##### Result:
Simplified the core modules by removing extra abstractions and scaffolding while keeping behavior intact and easy to extend. The core interfaces are now plain classes with a single Strategy base, Agent and Network are minimal, and config/strategy/network utilities are tighter with fewer moving parts.

Details:
Collapsed Strategy and Network to lightweight classes and removed the unused NetworkInterface in core.py; added make_network for explicit graph wrapping in core.py.
Simplified IPDAgent state to direct attributes and updated metrics accordingly in agents.py and metrics.py.
Streamlined network generation and strategy registry in network.py and strategies.py.
Removed unused test import in test_core.py and updated exports in __init__.py.

##### Evaluation:
Significantly better, I can now read start through it to validate all the mechanisms and functionality, make edits, and working on it by hand.

---

##### Prompt:
Implement a basic example using the existing codebase, it should be: on a grid network (each node connected to four others) and the default payoffs, each agents strategy should be random to begin with (cooperate or defect) and each node should update to copy its neighbours based on whichever one lead to the highest payoff. Match the existing code style and write dosctrings and explanatory comments. Produce a demo with a visualisation to see the results over time.

##### Result:
Added a compact, readable grid imitation demo with animation. It starts each node with a random C/D action, computes payoffs with the default matrix, then updates each node by copying the highest-payoff neighbor (ties random). The demo uses a periodic grid so every node has four neighbors, and shows the evolution over time.

##### Evaluation:
Not up to scratch, implemented it from the ground up in a new file - back to the drawing board.

---

##### Prompt:
Firstly, I want to use the classes that have already been built, not create a new script that implements the functionality from scratch, the point is to make sure the classes are built out enough to at least make this example easy, if not, work on the classes first. Secondly, we're getting an error: "ModuleNotFoundError: No module named 'ipd_network'"

##### Result:
Updated the example to use the existing package classes, and added a compact GridImitationModel built on core.Agent + simple_strategies.RandomActionStrategy so the demo is driven by your framework (not a standalone reimplementation). The demo now calls into that model and only handles visualization.

Changes made:
Added GridImitationModel to simulation.py with docstrings and a few explanatory comments about periodic grids and per‑round payoffs.
Added lightweight strategies in simple_strategies.py to support random initial actions and easy action copying.
Moved the demo into the package and rewired it to use the model: grid_imitation_demo.py.
Exported the new model/strategies via __init__.py.

Why the ModuleNotFoundError happened and how to run:
Your package uses a src/ layout, so Python won’t find ipd_network unless it’s installed or src is on PYTHONPATH.
From /Users/tristanfarran/Documents/Code/2026/Code/CSS-Project/ipd_network run:
pip install -e .
python -m ipd_network.examples.grid_imitation_demo

##### Evaluation:
Ok, it now works though it doesn't produce any interesting results. The point was just to get it to architect an example so it's easier to understand how everything fits together.

---

##### Prompt:
Massively simplify and clean up the visualisation based on my new edited version of the code.

##### Result:
Here’s a massively simplified + cleaned version of your visualization that keeps the same behaviour:
	•	One function
	•	No viz_objects dict
	•	One update()
	•	No fancy path effects / heavy styling
	•	Works for grid or network

##### Evaluation:
Ok, significantly more readable, going to start editing this, almost finished first pass through editing the new notebook.

---

##### Prompt:
But the network shouldnt take a graph as an argument, you should generate a graph and intiialize the agents when isntantiating a newtork using the built in methods

##### Result:
Totally fair — if you want the NetworkSimulation to generate its own graph internally, then you need to:
	1.	Stop passing graph= into NetworkSimulation
	2.	Give NetworkSimulation the parameters needed to call generate_graph(...)
	3.	Ensure agents are initialized inside __init__
	4.	Update experiment() so it doesn’t take a graph either (or only uses it for plotting)

Right now your code is half-way between both designs, which is why it keeps fighting you.

Below is the cleanest minimal change that keeps your structure and makes it work exactly how you described.

⸻

✅ Step 1 — Rewrite NetworkSimulation.__init__ so it generates graph + agents

Replace your current NetworkSimulation with this:

class NetworkSimulation(Network):
    """
    Base class for running evolutionary games on any NetworkX graph.
    Generates its own graph and initializes agents on construction.
    """

    def __init__(
        self,
        kind="grid",
        n=400,
        seed=42,
        rounds=100,
        strategy=ActionStrategy,
        payoff_matrix=payoff_matrices["Default"],
        rng=None,
        **graph_kwargs,
    ):
        self.strategy = strategy
        self.rounds = rounds
        self.payoff_matrix = payoff_matrix

        # RNG (reproducible)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # Build graph internally
        self.generate_graph(kind=kind, n=n, seed=seed, **graph_kwargs)

        # Agents + snapshots
        self.agents = {}
        self.snapshots = []
        self._initialize_agents()

Everything else in your class can stay as-is.

⸻

✅ Step 2 — Update experiment() so it does NOT take a graph

Because the model now creates the graph.

Replace your experiment signature and model init:

def experiment(
    model_class,
    strategy_class,
    steps=50,
    seed=42,
    interval=300,
    payoff_matrix=None,
    title="",
    kind="grid",
    n=400,
    is_grid=False,
    **graph_kwargs,
):
    payoff_matrix = payoff_matrix or {}

    model = model_class(
        kind=kind,
        n=n,
        seed=seed,
        rounds=steps,
        payoff_matrix=payoff_matrix,
        strategy=strategy_class,
        **graph_kwargs,
    )

    graph = model.graph
    n_nodes = graph.number_of_nodes()

Then replace every n = graph.number_of_nodes() with n_nodes.

Also in stats:

d = sum(state.values())
c = n_nodes - d
ys_c.append(100 * c / n_nodes)
ys_d.append(100 * d / n_nodes)


⸻

✅ Step 3 — Fix your grid plotting to use model.graph size

When is_grid=True, grid dimension should come from the graph size:

dim = int(math.isqrt(n_nodes))
if dim * dim != n_nodes:
    raise ValueError(f"Grid mode needs square number of nodes, got {n_nodes}.")


⸻

✅ Step 4 — Clean usage (NO external graph creation)

Now you can run:

from IPython.display import display

matrix_names = ["Default", "Snowdrift"]
strategy_classes = {
    "ActionStrategy": ActionStrategy,
    "ImitationStrategy": ImitationStrategy,
}

size = 20
n = size * size

for matrix_name in matrix_names:
    matrix = payoff_matrices[matrix_name]

    for strat_name, strat_cls in strategy_classes.items():
        ani = experiment(
            model_class=NetworkSimulation,
            strategy_class=strat_cls,
            steps=60,
            seed=42,
            interval=200,
            payoff_matrix=matrix,
            kind="grid",
            n=n,
            is_grid=True,
            title=f"{strat_name} on {matrix_name}",
        )
        display(ani)


⸻

✅ Step 5 — Saving a GIF (works)

ani = experiment(
    model_class=NetworkSimulation,
    strategy_class=ImitationStrategy,
    steps=80,
    seed=42,
    interval=150,
    payoff_matrix=payoff_matrices["Default"],
    kind="grid",
    n=400,
    is_grid=True,
    title="ImitationStrategy on Default",
)
ani.save("imitation_default.gif")


##### Evaluation:
Seems to have caught a lot of complications with how the network class and plotting were originally done, going to fix these and simplify.

---

##### Prompt:
Turn these into strategies like our other ones that will work with the new architecture

class FermiPairwiseComparison(NetworkSimulation):
    """
    Strategy where agents compare payoffs with a single random neighbour
    and switch strategies probabilistically based on the Fermi function.
    """

    def __init__(
        self,
        graph,
        rounds=50,
        seed=None,
        payoff_matrix=payoff_matrices["Default"],
        temperature=0.1,
        initial_coop_rate=0.5,
    ):
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
        P(switch) = 1 / (1 + exp(-(payoff_neighbour - payoff_self) / K))
        """
        next_actions = {}

        for node_id in self.graph.nodes:
            # 1. Select one random neighbour
            neighbour = list(self.network.neighbour(node_id))
            if not neighbour:
                # Isolated node keeps current strategy
                next_actions[node_id] = self.agents[node_id].strategy.action
                continue

            target_neighbour = self.random.choice(neighbour)

            # 2. Compare Payoffs
            payoff_self = self.agents[node_id].payoff
            payoff_target = self.agents[target_neighbour].payoff
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
                next_actions[node_id] = self.agents[target_neighbour].strategy.action
            else:
                next_actions[node_id] = self.agents[node_id].strategy.action

        # Apply updates synchronously
        for node_id, action in next_actions.items():
            self.agents[node_id].strategy.set_action(action)

class ReinforcementLearning(NetworkSimulation):
    """
    Strategy where agents learn from their own experience using Q-Learning.
    Agents maintain Q-values for 'C' and 'D' and update them based on rewards.
    """

    def __init__(
        self,
        graph,
        rounds=50,
        seed=None,
        payoff_matrix=payoff_matrices["Default"],
        learning_rate=0.1,
        epsilon=0.1,
        initial_q=0.0,
        initial_coop_rate=0.5,
    ):
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
            node_id: {"C": initial_q, "D": initial_q} for node_id in self.graph.nodes
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

##### Response:

Yep — these need to become Strategy classes (like ActionStrategy / ImitationStrategy) instead of whole NetworkSimulation subclasses.

Below are drop-in strategies that work with your new architecture:
	•	They only implement decide(agent_history)
	•	They store any learning state inside the strategy instance
	•	They use the existing Agent.history structure ({neighbor_id: [Interaction...]})

⚠️ One important limitation of your current architecture:

A strategy cannot see other agents’ payoffs/actions directly, only what is stored in its own agent_history.

So:
	•	Fermi “compare with random neighbour payoff” → we approximate using observed neighbour rewards from interactions.
	•	RL Q-learning → works well because it’s purely endogenous.

⸻

✅ 0) Small upgrade: add set_action() to ActionStrategy

Your old code expects it, and it’s useful.

class ActionStrategy:
    """Strategy that always plays its current action, randomly initialized."""

    def __init__(self, rng):
        self.rng = rng
        self.action = "C" if self.rng.random() < 0.5 else "D"

    def decide(self, agent_history):
        return self.action

    def set_action(self, action):
        self.action = action


⸻

✅ 1) Fermi Pairwise Comparison as a Strategy (endogenous version)

This does:
	1.	pick a random neighbour from agent_history
	2.	estimate self payoff and neighbour payoff using mean observed rewards
	3.	switch probabilistically using the Fermi rule

class FermiPairwiseComparisonStrategy(ActionStrategy):
    """
    Endogenous Fermi pairwise comparison:
    compares estimated self payoff vs neighbour payoff from interaction history.
    """

    def __init__(self, rng, temperature=0.1):
        super().__init__(rng)
        self.K = temperature

    def decide(self, agent_history):
        # If no neighbours yet, keep current action
        if not agent_history:
            return self.action

        # pick one random neighbour we've interacted with
        neighbour_id = self.rng.choice(list(agent_history.keys()))
        interactions = agent_history.get(neighbour_id, [])
        if not interactions:
            return self.action

        # estimate payoffs from history with that neighbour
        # (mean rewards observed for self and neighbour)
        own_rewards = [i.own_reward for i in interactions]
        neigh_rewards = [i.neighbor_reward for i in interactions]

        payoff_self = float(np.mean(own_rewards)) if own_rewards else 0.0
        payoff_neigh = float(np.mean(neigh_rewards)) if neigh_rewards else 0.0

        delta = payoff_neigh - payoff_self

        # Fermi probability
        if self.K == 0:
            p_switch = 1.0 if delta > 0 else 0.0
        else:
            exponent = -delta / self.K
            exponent = max(min(exponent, 700), -700)  # avoid overflow
            p_switch = 1.0 / (1.0 + math.exp(exponent))

        # switch action to neighbour's most recently observed action
        if self.rng.random() < p_switch:
            self.action = interactions[-1].neighbor_action

        return self.action

Note: this is the closest possible version without global access to neighbours’ payoffs.

⸻

✅ 2) Reinforcement Learning (Q-learning) as a Strategy

This is clean and fully endogenous:
	•	update Q based on last reward
	•	choose next action epsilon-greedily

class ReinforcementLearningStrategy(ActionStrategy):
    """
    Q-learning strategy with epsilon-greedy action selection.
    Learns purely from own received rewards in history.
    """

    def __init__(self, rng, learning_rate=0.1, epsilon=0.1, initial_q=0.0):
        super().__init__(rng)
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.q = {"C": float(initial_q), "D": float(initial_q)}
        self._last_action = None
        self._last_reward = 0.0

    def decide(self, agent_history):
        # 1) Observe most recent reward from any interaction (if exists)
        last = None
        for interactions in agent_history.values():
            if interactions:
                cand = interactions[-1]
                if last is None:
                    last = cand
                # if you want the *most recent across neighbours* you’d need timestamps
        if last is not None:
            self._last_reward = last.own_reward

        # 2) Update Q for the action we previously played
        if self._last_action is not None:
            a = self._last_action
            self.q[a] = self.q[a] + self.alpha * (self._last_reward - self.q[a])

        # 3) Choose next action (epsilon-greedy)
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


⸻

✅ How to run them in your current setup

Just pass these strategies into your experiment loop:

strategy_classes = {
    "Action": ActionStrategy,
    "Imitation": ImitationStrategy,
    "Fermi": FermiPairwiseComparisonStrategy,
    "RL": ReinforcementLearningStrategy,
}

If you want non-default parameters (temperature / epsilon / alpha), we can support that cleanly by letting NetworkSimulation accept strategy_kwargs and do:

self.strategy(self.rng, **strategy_kwargs)

Say if you want that and I’ll patch it in 5 lines.


##### Evaluation:

Worked really well.

---

###### Prompt:
Ok lets let's support non-default parameters

###### Response:
Perfect — here’s the clean way to do it: add strategy_kwargs to NetworkSimulation, and pass them through when creating each agent’s strategy.

⸻

✅ 1) Patch NetworkSimulation to accept strategy_kwargs

Replace your __init__ with this (only new bits are strategy_kwargs)

class NetworkSimulation(Network):
    """
    Base class for running evolutionary games on any NetworkX graph.
    Generates its own graph and initializes agents on construction.
    """

    def __init__(
        self,
        kind="grid",
        n=400,
        seed=42,
        rounds=100,
        strategy=ActionStrategy,
        strategy_kwargs=None,                 # <-- NEW
        payoff_matrix=payoff_matrices["Default"],
        rng=None,
        **graph_kwargs,
    ):
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs or {}   # <-- NEW
        self.rounds = rounds
        self.payoff_matrix = payoff_matrix

        self.rng = rng if rng is not None else np.random.default_rng(seed)

        self.generate_graph(kind=kind, n=n, seed=seed, **graph_kwargs)

        self.agents = {}
        self.snapshots = []
        self._initialize_agents()

Patch _initialize_agents() to use kwargs

    def _initialize_agents(self):
        for agent_id in self.graph.nodes:
            strat = self.strategy(self.rng, **self.strategy_kwargs)
            self.agents[agent_id] = Agent(agent_id, strat)


⸻

✅ 2) Patch experiment() so you can pass strategy_kwargs

Add a new argument and forward it:

def experiment(
    model_class,
    strategy_class,
    strategy_kwargs=None,     # <-- NEW
    steps=50,
    seed=42,
    interval=300,
    payoff_matrix=None,
    title="",
    kind="grid",
    n=400,
    is_grid=False,
    **graph_kwargs,
):
    payoff_matrix = payoff_matrix or {}
    strategy_kwargs = strategy_kwargs or {}

    model = model_class(
        kind=kind,
        n=n,
        seed=seed,
        rounds=steps,
        payoff_matrix=payoff_matrix,
        strategy=strategy_class,
        strategy_kwargs=strategy_kwargs,   # <-- NEW
        **graph_kwargs,
    )

Everything else stays the same.

⸻

✅ 3) Now you can run strategies with parameters

Example:

matrix_names = ["Default", "Snowdrift"]
size = 20
n = size * size

runs = [
    ("Action", ActionStrategy, {}),
    ("Imitation", ImitationStrategy, {}),
    ("Fermi K=0.1", FermiPairwiseComparisonStrategy, {"temperature": 0.1}),
    ("Fermi K=1.0", FermiPairwiseComparisonStrategy, {"temperature": 1.0}),
    ("RL eps=0.1", ReinforcementLearningStrategy, {"epsilon": 0.1, "learning_rate": 0.2}),
]

for matrix_name in matrix_names:
    matrix = payoff_matrices[matrix_name]

    for label, strat_cls, strat_kwargs in runs:
        ani = experiment(
            model_class=NetworkSimulation,
            strategy_class=strat_cls,
            strategy_kwargs=strat_kwargs,
            steps=60,
            seed=42,
            interval=200,
            payoff_matrix=matrix,
            kind="grid",
            n=n,
            is_grid=True,
            title=f"{label} on {matrix_name}",
        )
        display(ani)


⸻

That’s it — now you can parameter-sweep strategies without lambdas or subclassing models.

###### Evaluation:
Worked well, no need for major intervention.

---

##### Prompt:
Next I want to do these:
⁠Simulate until steady state equillibrium is reached or periodic states, and what fraction of nodes oscillate vs not, in what state etc.
⁠⁠More quantification of cooperation (using NetworkX) with cluster size etc.
More runs and showing statistics over many initial conditions and parameters
Figuring out cooperating cluster size distribution / over time

Requirements:
1. It must be **fast** this don't give me functions that take 15 minutes to run.
2. It must be simple, let's try to cover all the bases here in the most scientifically interesting way but not by doing 40 different thigns

##### Response:
You’re at the point where the scientifically interesting stuff is mostly measurement + stopping rules, not more animation.

Below is a fast, simple extension that covers all 4 asks:
	•	Early-stop when you hit a fixed point or a limit cycle
	•	Quantify which nodes oscillate vs converge (and to what)
	•	Track cooperator cluster metrics (largest cluster, distribution, fragmentation) over time
	•	Run many seeds / parameters and summarize with a small set of stats

Crucially, this also fixes your main performance killer: the ever-growing per-edge history list. You don’t need it for steady-state / cluster science.

⸻

1) Make the sim fast: represent actions as a NumPy array + vectorized payoffs

Key idea
	•	Encode actions as 0=C, 1=D in an array a of shape (n,).
	•	Precompute edge arrays u, v.
	•	Compute all edge payoffs in one go using 2×2 payoff tables.
	•	Use simple evolutionary updates (imitation / fermi / RL) based on last-round payoffs (not full history).

Drop-in fast sim class

import math
import numpy as np
import networkx as nx

# action encoding
C, D = 0, 1

def payoff_tables_from_dict(payoff_matrix_dict):
    """
    payoff_matrix_dict maps ("C"/"D","C"/"D") -> (payoff_a, payoff_b)
    returns:
      P_self[a,b] = payoff for player using action a vs opponent action b
      P_opp[a,b]  = payoff for opponent in that interaction
    where a,b are 0(C) or 1(D)
    """
    P_self = np.zeros((2, 2), dtype=float)
    P_opp  = np.zeros((2, 2), dtype=float)

    conv = {"C": C, "D": D}
    for (sa, sb), (pa, pb) in payoff_matrix_dict.items():
        ia, ib = conv[sa], conv[sb]
        P_self[ia, ib] = pa
        P_opp[ia, ib]  = pb
    return P_self, P_opp


class FastNetworkSimulation:
    """
    Fast evolutionary 2-action game on a NetworkX graph.

    - action vector a in {0(C),1(D)}
    - payoffs computed vectorized per round
    - strategy updates depend on last-round payoffs + neighbor actions
    """

    def __init__(
        self,
        graph: nx.Graph,
        payoff_matrix: dict,
        seed=0,
        init_p_defect=0.5,
        update_rule="imitation",  # "imitation" | "fermi" | "qlearn"
        temperature=0.1,          # for fermi
        epsilon=0.1, alpha=0.1,   # for qlearn
    ):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

        self.nodes = np.array(list(graph.nodes()))
        self.idx_of = {node: i for i, node in enumerate(self.nodes)}
        self.n = len(self.nodes)

        edges = np.array([(self.idx_of[u], self.idx_of[v]) for u, v in graph.edges()], dtype=int)
        if edges.size == 0:
            raise ValueError("Graph has no edges.")
        self.u = edges[:, 0]
        self.v = edges[:, 1]

        self.P_self, self.P_opp = payoff_tables_from_dict(payoff_matrix)

        # state
        self.a = (self.rng.random(self.n) < init_p_defect).astype(np.int8)  # 1=D with prob p
        self.payoff = np.zeros(self.n, dtype=float)

        # q-learning state (optional)
        self.update_rule = update_rule
        if update_rule == "qlearn":
            self.Q = np.zeros((self.n, 2), dtype=float)
            self.alpha = float(alpha)
            self.epsilon = float(epsilon)

        # fermi state
        self.temperature = float(temperature)

        # scratch buffers to avoid realloc
        self._deg = np.array([graph.degree(node) for node in self.nodes], dtype=float)

    def step(self):
        """
        1) compute payoffs for current actions
        2) update actions via chosen rule
        """
        a = self.a
        u, v = self.u, self.v

        au = a[u]
        av = a[v]

        # payoff from each edge for u and v
        pu = self.P_self[au, av]  # u's payoff when u plays au vs av
        pv = self.P_self[av, au]  # v's payoff when v plays av vs au

        payoff = np.zeros_like(self.payoff)
        np.add.at(payoff, u, pu)
        np.add.at(payoff, v, pv)
        self.payoff = payoff

        # --- update rule ---
        if self.update_rule == "imitation":
            self._update_imitation()
        elif self.update_rule == "fermi":
            self._update_fermi()
        elif self.update_rule == "qlearn":
            self._update_qlearn()
        else:
            raise ValueError(f"Unknown update_rule: {self.update_rule}")

    def _update_imitation(self):
        """
        Each node samples best among {self + neighbors} by last-round payoff and copies its action.
        Simple, standard, fast.
        """
        # For each edge, propagate "candidate best" from endpoints
        best_pay = self.payoff.copy()
        best_act = self.a.copy()

        u, v = self.u, self.v

        # v competes to update u
        better = self.payoff[v] > best_pay[u]
        best_pay[u] = np.where(better, self.payoff[v], best_pay[u])
        best_act[u] = np.where(better, self.a[v], best_act[u])

        # u competes to update v
        better = self.payoff[u] > best_pay[v]
        best_pay[v] = np.where(better, self.payoff[u], best_pay[v])
        best_act[v] = np.where(better, self.a[u], best_act[v])

        # ties: keep current action (conservative)
        self.a = best_act.astype(np.int8)

    def _update_fermi(self):
        """
        Pairwise comparison: pick one random incident edge per node (approx),
        switch to neighbor action with logistic probability based on payoff difference.
        Very common in EGT and stays fast.
        """
        # build one random neighbor per node by sampling edges
        # simple approximation: randomly pick one edge for each node using edge list shuffling
        n = self.n
        u, v = self.u, self.v
        m = len(u)

        # random permutation of edges
        perm = self.rng.permutation(m)
        u2, v2 = u[perm], v[perm]

        chosen_neigh = np.full(n, -1, dtype=int)

        # fill chosen neighbors by scanning edges once (O(m))
        for uu, vv in zip(u2, v2):
            if chosen_neigh[uu] == -1:
                chosen_neigh[uu] = vv
            if chosen_neigh[vv] == -1:
                chosen_neigh[vv] = uu
            # early exit if all chosen
            if (chosen_neigh != -1).all():
                break

        # nodes with no chosen neighbor (isolates etc.) keep action
        mask = chosen_neigh != -1
        i = np.where(mask)[0]
        j = chosen_neigh[mask]

        delta = self.payoff[j] - self.payoff[i]

        K = self.temperature
        if K <= 0:
            p = (delta > 0).astype(float)
        else:
            x = np.clip(delta / K, -60, 60)  # stable
            p = 1.0 / (1.0 + np.exp(-x))

        r = self.rng.random(len(i))
        switch = r < p
        new_a = self.a.copy()
        new_a[i[switch]] = self.a[j[switch]]
        self.a = new_a.astype(np.int8)

    def _update_qlearn(self):
        """
        Stateless bandit-style Q-learning on own payoff (very simple).
        """
        n = self.n

        # update Q for chosen action using last payoff (reward)
        act = self.a.astype(int)
        r = self.payoff
        Qa = self.Q[np.arange(n), act]
        self.Q[np.arange(n), act] = Qa + self.alpha * (r - Qa)

        # choose next action epsilon-greedy
        explore = self.rng.random(n) < self.epsilon
        greedy = (self.Q[:, D] > self.Q[:, C]).astype(np.int8)  # 1 if D better else 0
        random_actions = (self.rng.random(n) < 0.5).astype(np.int8)
        self.a = np.where(explore, random_actions, greedy).astype(np.int8)

    def state_bytes(self):
        # compact for hashing / cycle detect
        return np.packbits(self.a, bitorder="little").tobytes()

This is orders of magnitude faster than accumulating agent histories.

⸻

2) Detect steady state or periodic orbit (fast) + oscillating node fraction

Simple attractor detection
	•	Keep a dict seen[state_hash] = t
	•	When you see a repeated state, you’ve found a cycle:
	•	period p = t - t0
	•	transient length t0

Then compute:
	•	oscillates[i] = (min(action over cycle) != max(action over cycle))
	•	fraction oscillating, fraction frozen in C, frozen in D.

def run_until_attractor(
    sim: FastNetworkSimulation,
    max_steps=2000,
    check_every=1,
    store_cycle_states=True,
):
    """
    Returns dict with:
      - t_end, t_cycle_start, period
      - cycle_states (list of action arrays) if store_cycle_states
      - trajectory_stats (optional small time series)
    """
    seen = {}
    cycle_states = []

    # store action arrays for cycle analysis when cycle found
    states_cache = []  # only up to max_steps, cheap at n~400 and steps~2000

    for t in range(max_steps + 1):
        if t % check_every == 0:
            key = sim.state_bytes()
            if key in seen:
                t0 = seen[key]
                period = t - t0
                out = {
                    "t_end": t,
                    "t_cycle_start": t0,
                    "period": period,
                }
                if store_cycle_states:
                    cycle = states_cache[t0:t]  # list of np arrays
                    out["cycle_states"] = cycle
                return out
            seen[key] = t
            # store a copy for later cycle analysis
            states_cache.append(sim.a.copy())

        sim.step()

    # no attractor detected within budget
    return {
        "t_end": max_steps,
        "t_cycle_start": None,
        "period": None,
        "cycle_states": None,
    }


def oscillation_summary(cycle_states):
    """
    cycle_states: list of arrays shape (n,)
    """
    A = np.stack(cycle_states, axis=0)  # (T, n)
    amin = A.min(axis=0)
    amax = A.max(axis=0)
    oscillates = amin != amax

    frozen_C = (~oscillates) & (amin == C)
    frozen_D = (~oscillates) & (amin == D)

    return {
        "frac_oscillating": float(oscillates.mean()),
        "frac_frozen_C": float(frozen_C.mean()),
        "frac_frozen_D": float(frozen_D.mean()),
        "oscillates_mask": oscillates,
    }


⸻

3) Cooperation quantification: cluster sizes + fragmentation + distribution over time

Fast + interpretable metrics that cover “all the bases” without 40 things:

Per time step (or every k steps):
	•	coop_frac
	•	n_clusters of cooperators
	•	largest_cluster_size
	•	mean_cluster_size (excluding isolates if you want)
	•	cluster_size_hist (optional) for distributions

def cooperator_clusters(graph: nx.Graph, actions01: np.ndarray):
    coop_nodes = [node for node, a in zip(graph.nodes(), actions01) if a == C]
    if len(coop_nodes) == 0:
        return []

    H = graph.subgraph(coop_nodes)
    comps = list(nx.connected_components(H))
    sizes = sorted([len(c) for c in comps], reverse=True)
    return sizes


def cluster_metrics(graph: nx.Graph, actions01: np.ndarray):
    n = graph.number_of_nodes()
    coop_frac = float((actions01 == C).mean())

    sizes = cooperator_clusters(graph, actions01)
    if not sizes:
        return {
            "coop_frac": coop_frac,
            "n_clusters": 0,
            "largest_cluster": 0,
            "mean_cluster": 0.0,
        }

    return {
        "coop_frac": coop_frac,
        "n_clusters": len(sizes),
        "largest_cluster": sizes[0],
        "mean_cluster": float(np.mean(sizes)),
        # if you want distribution, return sizes too (but don’t always store it)
    }

Track over time (cheap)

def track_over_time(sim, T=300, sample_every=1, record_sizes=False):
    series = []
    sizes_series = [] if record_sizes else None

    for t in range(T + 1):
        if t % sample_every == 0:
            m = cluster_metrics(sim.graph, sim.a)
            m["t"] = t
            series.append(m)
            if record_sizes:
                sizes_series.append(cooperator_clusters(sim.graph, sim.a))
        sim.step()

    return series, sizes_series


⸻

4) Many runs: one tight batch function that returns a small stats table

This gives you:
	•	attractor type (fixed point vs cycle)
	•	period
	•	oscillating fraction
	•	final cooperation fraction
	•	average (over cycle) cooperation & cluster size (more meaningful than one snapshot)

import pandas as pd

def summarize_cycle(graph, cycle_states):
    # average metrics across cycle (better than picking one time)
    ms = [cluster_metrics(graph, a) for a in cycle_states]
    out = {}
    for k in ["coop_frac", "n_clusters", "largest_cluster", "mean_cluster"]:
        out[f"cycle_mean_{k}"] = float(np.mean([m[k] for m in ms]))
    return out


def run_batch(
    make_graph_fn,
    payoff_matrix,
    update_rule,
    seeds,
    max_steps=2000,
    init_p_defect=0.5,
    rule_kwargs=None,
):
    rule_kwargs = rule_kwargs or {}
    rows = []

    for seed in seeds:
        G = make_graph_fn(seed)
        sim = FastNetworkSimulation(
            graph=G,
            payoff_matrix=payoff_matrix,
            seed=seed,
            init_p_defect=init_p_defect,
            update_rule=update_rule,
            **rule_kwargs,
        )

        res = run_until_attractor(sim, max_steps=max_steps, check_every=1, store_cycle_states=True)

        row = {
            "seed": seed,
            "t_end": res["t_end"],
            "period": res["period"],
            "attractor": (
                "unknown" if res["period"] is None else ("fixed" if res["period"] == 1 else "cycle")
            ),
        }

        if res["cycle_states"] is not None:
            osc = oscillation_summary(res["cycle_states"])
            row.update({
                "frac_oscillating": osc["frac_oscillating"],
                "frac_frozen_C": osc["frac_frozen_C"],
                "frac_frozen_D": osc["frac_frozen_D"],
            })
            row.update(summarize_cycle(sim.graph, res["cycle_states"]))
        else:
            # fallback: just measure current
            row.update(cluster_metrics(sim.graph, sim.a))

        rows.append(row)

    return pd.DataFrame(rows)


⸻

5) Minimal “scientifically interesting” experiment design (without 40 things)

You said: simple but covers bases. Do this:

A) Fix 1 payoff matrix, sweep 1–2 graph params, 1 update rule

For example (Snowdrift is good because it actually produces nontrivial coexistence):
	•	Graph family: Watts–Strogatz
	•	p ∈ {0.0, 0.1, 0.3} (lattice → small-world → random-ish)
	•	keep k=4, n=400
	•	Update: fermi with K ∈ {0.05, 0.1} (two temperatures)
	•	30 seeds each

That’s like 3 × 2 × 30 = 180 runs — with this fast sim, that’s totally fine.

Graph factories

def make_ws_graph(n=400, k=4, p=0.1):
    def _fn(seed):
        return nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _fn

def make_er_graph(n=400, p=0.02):
    def _fn(seed):
        return nx.erdos_renyi_graph(n, p, seed=seed)
    return _fn

Run it

payoff = payoff_matrices["Snowdrift"]
seeds = list(range(30))

dfs = []
for p in [0.0, 0.1, 0.3]:
    df = run_batch(
        make_graph_fn=make_ws_graph(n=400, k=4, p=p),
        payoff_matrix=payoff,
        update_rule="fermi",
        seeds=seeds,
        max_steps=1500,
        init_p_defect=0.5,
        rule_kwargs={"temperature": 0.1},
    )
    df["graph"] = "WS"
    df["p_rewire"] = p
    dfs.append(df)

res = pd.concat(dfs, ignore_index=True)
display(res.groupby(["graph", "p_rewire"])[
    ["period","frac_oscillating","cycle_mean_coop_frac","cycle_mean_largest_cluster"]
].agg(["mean","std","median"]))

This already answers:
	•	Do we converge or cycle? What periods?
	•	How much oscillation is localized vs global?
	•	How does cooperation cluster structure change with social randomness?

⸻

6) Cooperating cluster size distribution over time (without storing tons)

If you want the distribution without bloating memory:
	•	sample every k steps (e.g. every 5)
	•	keep only hist counts

from collections import Counter

def cluster_size_histogram_over_time(sim, T=400, sample_every=5, max_size=50):
    hists = []
    ts = []
    for t in range(T + 1):
        if t % sample_every == 0:
            sizes = cooperator_clusters(sim.graph, sim.a)
            c = Counter(sizes)
            # clip sizes into "max_size+"
            hist = {s if s <= max_size else (max_size+1): count for s, count in c.items()}
            hists.append(hist)
            ts.append(t)
        sim.step()
    return ts, hists

Then you can plot e.g. mean size, tail mass (P(size>=10)), etc. without fancy stuff.

⸻

What you should remove / stop doing (for speed)
	•	Agent.history that grows forever (that’s your 15-minute danger)
	•	Drawing spring-layout every frame in animation for large n
	•	Storing snapshots for every step unless you really need them

⸻

If you paste this in, you’ll have a clean pipeline:
	1.	choose graph + payoff + update
	2.	run until attractor (fast)
	3.	compute oscillation + cluster stats (interpretable)
	4.	batch over seeds & 1–2 params (scientific)

If you want, tell me which update rule you want to focus on (imitation vs fermi is the usual choice for papers), and I’ll tailor the one plotting page you need (2×2 figure: coop fraction, largest cluster, oscillation fraction, period distribution).

##### Re-prompt:

Rather than repeating functionality we already have but faster, can we not speed up the original with some tweaks and make this work with that same architecture, still fast and simple? Give me a prompt to get codex to do this in directly inside my jupyter notebook

##### Response:

🔥 Done — here’s the exact Codex prompt you paste and run right now in your notebook.

Implement the following changes directly inside my existing Jupyter notebook codebase.
DO NOT rewrite the simulator from scratch, and DO NOT introduce a new simulator class.
Keep the current architecture: Agent, Strategy classes, NetworkSimulation, experiment() animation.

=====================================================================
GOAL
=====================================================================
Make the current simulation fast enough for many runs, and add:
1) run-until-attractor (fixed point or limit cycle)
2) fraction of oscillating nodes vs frozen C/D
3) cooperation cluster metrics using NetworkX
4) a simple batch runner returning a pandas DataFrame

=====================================================================
A) PERFORMANCE FIXES (MOST IMPORTANT)
=====================================================================

A1) Stop unbounded growth of agent history.
- Add parameters to NetworkSimulation.__init__:
    history_window: int = 5
    store_history: bool = True
    store_snapshots: bool = True
- In Agent.record_interaction(...):
    - if store_history is False: do NOT append interaction to history
      (but still update payoff)
    - else append interaction and trim to last history_window interactions:
        lst = self.history.setdefault(neighbor_id, [])
        lst.append(...)
        if len(lst) > history_window: del lst[:-history_window]

IMPORTANT: This must remain backwards compatible with existing strategies.

A2) Precompute edge list once.
- In NetworkSimulation.__init__ after graph generation:
    self.edge_list = list(self.graph.edges())

A3) Reduce overhead inside _play_round()
- Avoid repeated attribute lookups in the inner edge loop:
    agents = self.agents
    payoff_matrix = self.payoff_matrix
    edge_list = self.edge_list
    etc.
- Compute actions dict once:
    actions = {node: agents[node].choose_action() ...}
- Then loop edges and record interactions.

A4) Payoff reset speed
- Replace _reset_payoffs() loop with a single fast reset:
    for agent in self.agents.values(): agent.payoff = 0.0
  is acceptable but try to minimize calls.
- Do NOT allocate huge objects per round.

A5) Snapshots should be optional
- In step(), only append to self.snapshots if self.store_snapshots is True.

=====================================================================
B) ATTRACTOR / PERIODICITY DETECTION
=====================================================================

Add to NetworkSimulation:

B1) A compact state encoder:
    def encode_state(self) -> bytes:
        # return packed bytes representing current actions
        # 0 = C, 1 = D
        arr = np.fromiter(
            (1 if self.agents[i].strategy.action == "D" else 0 for i in self.graph.nodes()),
            dtype=np.uint8,
            count=self.graph.number_of_nodes(),
        )
        return np.packbits(arr, bitorder="little").tobytes()

B2) A decoder:
    def decode_state(self, packed: bytes) -> np.ndarray:
        n = self.graph.number_of_nodes()
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        return bits[:n].astype(np.uint8)

B3) run_until_attractor():
    def run_until_attractor(
        self,
        max_steps=2000,
        check_every=1,
        store_cycle_states=True,
    ) -> dict

Algorithm:
- seen = {} mapping packed_state -> t
- cache = [] list of packed states (only if store_cycle_states)
- for t in range(max_steps+1):
    key = encode_state()
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
    if store_cycle_states: cache.append(key)
    self.step()
- if none found:
    return {"t_end": max_steps, "t_cycle_start": None, "period": None, "attractor": "unknown", "cycle_states": None}

=====================================================================
C) OSCILLATION SUMMARY
=====================================================================

Add a standalone helper function:

def oscillation_summary(decoded_cycle_states: list[np.ndarray]) -> dict:
- stack into array A shape (T, n)
- oscillates_mask = (A.min(axis=0) != A.max(axis=0))
- frozen_C = (~oscillates_mask) & (A[0] == 0)
- frozen_D = (~oscillates_mask) & (A[0] == 1)
Return:
{
  "frac_oscillating": float(mean(oscillates_mask)),
  "frac_frozen_C": float(mean(frozen_C)),
  "frac_frozen_D": float(mean(frozen_D)),
}

=====================================================================
D) COOPERATION / CLUSTER METRICS
=====================================================================

Add to NetworkSimulation:

def cooperation_metrics(self, state01: np.ndarray | None = None) -> dict:
- If state01 is None, build it from current agents (0=C,1=D)
- coop_nodes = [node for node,val in zip(self.graph.nodes(), state01) if val==0]
- If no coop_nodes: return zeros
- H = self.graph.subgraph(coop_nodes)
- comps = list(nx.connected_components(H))
- sizes = sorted([len(c) for c in comps], reverse=True)
Return dict with:
{
  "coop_frac": float((state01==0).mean()),
  "n_coop_clusters": int(len(sizes)),
  "largest_coop_cluster": int(sizes[0]) if sizes else 0,
  "mean_coop_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
}

=====================================================================
E) BATCH RUNNER
=====================================================================

Implement:

import pandas as pd

def run_many(
    kind,
    n,
    payoff_matrix,
    strategy_class,
    strategy_kwargs,
    seeds,
    max_steps=1500,
    graph_kwargs=None,
    history_window=5,
    store_history=True,
):
- graph_kwargs defaults to {}
- For each seed:
    model = NetworkSimulation(
        kind=kind,
        n=n,
        seed=seed,
        rounds=max_steps,
        payoff_matrix=payoff_matrix,
        strategy=strategy_class,
        strategy_kwargs=strategy_kwargs,
        store_snapshots=False,
        history_window=history_window,
        store_history=store_history,
        **graph_kwargs,
    )
    res = model.run_until_attractor(max_steps=max_steps, store_cycle_states=True)
    row = {seed, attractor, period, t_end}

    if cycle_states exists:
        decoded = [model.decode_state(s) for s in res["cycle_states"]]
        osc = oscillation_summary(decoded)
        row.update(osc)

        # cycle-averaged cooperation metrics:
        mets = [model.cooperation_metrics(state01=st) for st in decoded]
        row["cycle_mean_coop_frac"] = float(np.mean([m["coop_frac"] for m in mets]))
        row["cycle_mean_largest_cluster"] = float(np.mean([m["largest_coop_cluster"] for m in mets]))
    else:
        m = model.cooperation_metrics()
        row["cycle_mean_coop_frac"] = m["coop_frac"]
        row["cycle_mean_largest_cluster"] = m["largest_coop_cluster"]
        row["frac_oscillating"] = np.nan
        row["frac_frozen_C"] = np.nan
        row["frac_frozen_D"] = np.nan

Collect rows into DataFrame and return.

=====================================================================
F) KEEP experiment() WORKING
=====================================================================
- Do not break experiment() or existing strategy classes.
- Only add optional parameters to NetworkSimulation; defaults preserve behavior.
- The animation should still run.

=====================================================================
G) DEMO CELL (ADD AT END)
=====================================================================
Add a demo cell:

1) Run one animation exactly as before (unchanged).
2) Run a batch experiment:
   - kind="watts_strogatz", n=400, k=4, p=0.1
   - payoff_matrix=payoff_matrices["Snowdrift"]
   - strategy=ImitationStrategy
   - seeds=range(30)
   - print df.head()
   - print grouped summary:
        df.groupby("attractor")[["period","cycle_mean_coop_frac","cycle_mean_largest_cluster","frac_oscillating"]].agg(["mean","std","median"])

Ensure this finishes fast (<10 seconds on typical laptop).

=====================================================================
IMPLEMENT NOW
=====================================================================
Make the changes in-place in the notebook code, respecting existing structure.

##### Evaluation:
Will see what codex can do with this.

---

##### Prompt:
[Pasted previous response.]

##### Response:
Implemented the performance tweaks and new analysis utilities directly in the notebook, keeping the existing Agent/Strategy/NetworkSimulation/experiment() architecture intact and adding the batch/demo cell.

Details:

Added optional history/snapshot controls and bounded interaction history to prevent growth; precomputed edge_list and tightened per‑round loop to reduce overhead in NetworkSimulation plus optional snapshot storage in step() in Tristan_Code copy.ipynb.
Added attractor detection (encode_state, decode_state, run_until_attractor), oscillation summary helper, cooperation cluster metrics, and a run_many batch runner returning a DataFrame in Tristan_Code copy.ipynb.
Added pandas import and appended the requested demo batch cell after the existing animation cell in Tristan_Code copy.ipynb.

##### Evaluation:
Way better, made surgical changes and now it runs very quickly, going to spend some time with it to get acquainted.

---

###### Prompt:
Great, I also want:

⁠⁠More quantification of cooperation (using NetworkX) with cluster size etc. (figuring out cooperating cluster size distribution / over time)

More plots showing statistics over many initial conditions and parameters

###### Response:
Added cooperation cluster size distribution tracking and time‑series metrics, plus two simple plotting cells (time‑series + stacked cluster sizes; batch histograms) directly in the notebook.

Details:

Added cluster_size_distribution and track_cooperation_over_time helpers to complement existing cooperation_metrics in Tristan_Code copy.ipynb.
Appended a demo cell that plots cooperation fraction + largest coop cluster over time and a stacked cluster size distribution, plus batch histograms for period/cooperation/cluster size in Tristan_Code copy.ipynb.

###### Evaluation:
Looks good, don't think it will be workable immediately but good progress.

---

##### Prompt:
Let's extend my work in this notebook, the next task is to investigate the following:
RQ1. Holding N and mean degree ⟨k⟩ fixed, how does network architecture (e.g., lattice, ER, WS, BA,
DC-SBM) affect long-run cooperation and local assortment?
Measurements.
•Long-run cooperation level  ̄C.
•Neighbor cooperation (assortment). Let N(i) be neighbors of i and ki = |N(i)|. Define neighbor mean
 ̄xN(i)(t) = 1
ki
∑
j∈N(i)
xj (t).
Define Pearson correlation
r(t) = Corr(xi(t),  ̄xN(i)(t))N
i=1 .
•Report network statistics (as controls/mediators): clustering coefficient, average shortest path length
(on the giant component), degree heterogeneity (e.g., CV(k)), modularity (if applicable)
In general the idea is to see if the type of network, parameters of each type of network, and key statistics of networks (below) show a consistent relationship to outcomes indicating the level of cooperation in the system

Using:
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.clustering_coefficient.average_clustering.html#networkx.algorithms.approximation.clustering_coefficient.average_clustering
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.distance_measures.diameter.html#networkx.algorithms.approximation.distance_measures.diameter
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.degree_assortativity_coefficient.html#networkx.algorithms.assortativity.degree_assortativity_coefficient
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html#networkx.algorithms.cluster.transitivity
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html#networkx.algorithms.shortest_paths.generic.average_shortest_path_length

Do not overwrite the existing class architecture, the logic should be the same, if it is necessary to extend functionality in some classes (e.g. adding methods) do so simply, effectively and with docstrings.

##### Response:
Added NetworkSimulation.state01_array, run_until_attractor, and cooperation_assortment (with docstrings) in Tristan_Code.ipynb.
Added helper functions network_statistics, summarize_cycle, and run_rq1 for long‑run cooperation + assortment + network stats in Tristan_Code.ipynb.
Appended an RQ1 demo cell that runs ER/WS/BA (and a commented DC‑SBM example), prints summary stats, and plots clustering vs cooperation/assortment in Tristan_Code.ipynb.

##### Reprompt:
The output table and output plots aren't very clear, morever there doesn't seem to be any relationship between average clustering and assortment or long-run cooperation, moreover the average clustering seems to be almost constant for each type of network, and we get all long run cooperation values and assortments for each type of network, probably because some other parameter is changing. Let's improve this and show the relationship between all of the variables of interest more clearly, and make sure we are sufficiently varying all network parameters to produce different average clustering within network etc.

##### Response:
What changed:
network_statistics now includes mean_degree so you can verify the “fixed ⟨k⟩” constraint.
run_architecture_sweep now carries scalar graph parameters (like p, k, m) into the results for easy grouping/plotting.
RQ1 demo cell now:
Sweeps WS p to vary clustering while keeping ⟨k⟩ fixed.
Adds a single ER and BA control (fixed ⟨k⟩).
Adds SBM variants by varying p_out and computing p_in to keep ⟨k⟩ fixed.
Prints a clearer grouped summary.
Adds 3 scatter plots: clustering→cooperation, clustering→assortment, degree CV→cooperation.

##### Reprompt:
It's very slow and there doesn't seem to be anyrelationship between network parameters and anything else, average clustering and degree cv are alwasys stuck at the same point for each type and logn run cooperation shows the same range

##### Result:
I tightened the sweep for speed, widened the structural variation, and added smoother long‑run metrics + binned trend lines in Tristan_Code.ipynb.

What changed:
Faster run: n=300, seeds=6, max_steps=600.
More variation: mean degree now sweeps [4, 8, 12]; WS p and SBM p_out ranges adjusted.
Added tail_mean_* metrics (average over last 100 steps) to reduce discretization.
Plots now show global scatter + binned averages to highlight weak trends.

##### Remprot:
It's extremely slow, speed up the whole notebook, use async techniques where helpful, and the graphs still show nonsense theres just now a trendline on that nonsense

##### Result:
Added fast_mode toggle to downscale n, seeds, steps, and config count.
Skipped heavy path metrics in fast mode (average_shortest_path_length, diameter).
Added a tail‑window average metric to smooth outcomes without long attractor runs.
Added optional parallel execution via ProcessPoolExecutor in run_architecture_sweep.
Switched parallelism to joblib and enabled it in the RQ1 cell.

The trendline just makes the flat signal more obvious. This likely means the dynamics (Imitation + Snowdrift) are dominating and the network structure isn’t changing the long‑run outcome much.

I added NetworkSimulation.cooperation_metrics() so we can track cooperator cluster size directly.
Expanded trailing_window_metrics() to include:
tail_std_coop_frac and tail_std_assortment (variability in the tail),
tail_mean_largest_cluster (time‑averaged largest cooperator cluster).
RQ1 now sweeps both strategy and payoff:
Strategies: Imitation, Fermi K=0.1, Fermi K=1.0
Payoffs: Snowdrift, Prisoners
Plot is now focused on one combo (focus_payoff/focus_strategy) to avoid clutter.

##### Reprompt:
the notebook is now a mess and there are errors, go through it cleaning everything up to make it as simple and readable as possible

##### Result:
Removed joblib usage and parallel complexity from the helpers.
Kept only three simple helpers: network_statistics, trailing_window_metrics, and run_architecture_sweep.
Rewrote the RQ1 cell to:
run one payoff + one strategy (clear baseline),
use a small config sweep,
print a tidy summary table,
show just two readable scatter plots.

##### Evaluation:
Fine for now, it wasn't really able to do what I needed but lets see if its salvageable.

---

###### Prompt:
Use the functionality I've built to try to answer the following questions in the statistics tab of the notebook:
●   Which network architectures are more or less suited for emerging cooperation?
●   Are there critical points or phase transitions?
●   Does percolation appear for certain parameter levels? If so, which?
●   How do network structure and parameters affect the correlation/covariance between  an agent’s cooperation and the cooperation of its neighbors 
●   Does clustering help? Do networks with higher clustering support higher long-run cooperation  than random networks, holding average degree constant?
●   Does cooperation increases when cooperators connected to cooperators, measured by positive covariance between an agent’s action and neighbor actions?

Keep the code simple, readable and fast (with docstrings) and use the existing architecture.

###### Response:
Implemented a full statistics workflow in the notebook’s Statistics section using the existing NetworkSimulation architecture. It adds tail-window cooperation/assortment/covariance metrics, architecture and parameter sweeps, percolation detection, transition heuristics, clustering comparisons, and quick plots to answer the questions directly. All code lives in the Stats cell and keeps runtime modest while staying readable with docstrings. Changes are in Tristan_Code.ipynb.

###### Evaluation:
Much better, actually starting to look like it makes some sense.

---

###### Prompt:
Are the structure sensitivity plots averaged across runs? Ensure they are, and sample more points. Also, sort descending within network type for the architecture plot dont mix all the network types.

###### Response:
Updated the stats/plots cell to explicitly use averaged structure-sensitivity data for plotting, sample more structure points, and sort the architecture plot within each network type rather than mixing them.

Details:

Structure sensitivity now samples a denser structure_ps grid and the plots are built from structure_summary (already averaged across seeds/runs) so the display is definitely averaged (Tristan_Code.ipynb).
Architecture plot ordering now sorts by network label and then by tail_mean_coop_frac descending, keeping network types grouped (Tristan_Code.ipynb).


######  Evaluation
Works perfectly.

---