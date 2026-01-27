# Iterated Prisoner’s Dilemma On Networks (IPDN)

A small simulation framework to study how **cooperation emerges (or collapses)** in the Iterated Prisoner’s Dilemma when agents interact on different **network topologies** and evolve under different **update rules**.

This project was built for a **Complex Systems Simulation** setting and focuses on both **aggregate outcomes** (cooperation rate over time) and **spatial/network structure** (clusters, interfaces, and domain dynamics).

This project:
- Simulates repeated IPD interactions between agents connected by a graph
- Supports multiple **network models** (e.g. grid / random / small-world / BA)
- Supports multiple **strategy update rules** (e.g. imitation, Tit-for-Tat, RL-style)
- Produces:
  - **time-series plots** of cooperation fraction
  - **animated network visualizations** (GIFs) showing strategy spread over time
