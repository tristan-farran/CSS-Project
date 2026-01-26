##### Tristan

Removed helpers, run_til_attractor, and cooperation_metrics, and WIP section as not relelvant to my part and were just a quick sketch not fit for purpose.

Added tit for tat.
Change default payoff to be the usual
NetworkSimulation now owns neighbor-index caching plus cooperation_neighbor_stats, and cooperation_assortment uses the cache.


##### Zheng

Adjusted the payoff scales of Snowdrift and Prisoners to match the Fermi K = 0.5

Changed the class "NetworkSimulation':
(1) Normalize payoffs by node degree to avoid the bias in hub-related networks(BA,SBM). Without normalization, high-degree nodes systematically accumulate larger payoffs simply because they play more games.
Location: Inside NetworkSimulation._play_round() before calling record_interaction().

(2) Cache node degrees once during initialization: 
    self.deg = dict(self.graph.degree())
to avoid recomputing degrees inside the inner simulation loop; also needed for payoff normalization.
Location: nn NetworkSimulation.__init__() after graph generation.

(3) Accumulate each agent’s per-round payoff into agent.payoff to provide a consistent “fitness/payoff signal” for update rules like Fermi / imitation that should compare total payoff of agents (self vs neighbor) rather than only the payoff from one pairwise interaction.
Location: in NetworkSimulation._play_round() inside the edge loop.

Changed the class "FermiStrategy"
The original FermiStrategy compared payoffs using only the interaction history between i and a sampled neighbor j, where neighbor_reward is the payoff of j against i, not j’s overall payoff. This biases imitation and can cause systematic collapse (e.g., rapid all-D). It should compare i’s overall payoff (e.g., last round or moving average) and j’s overall payoff. This correction requires access to neighbor’s payoff (agent.payoff) or neighbor’s full history; adding payoff accumulation enables this.

