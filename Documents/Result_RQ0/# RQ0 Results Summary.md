# RQ0 Results Summary 
*(based on mean long-run cooperation \bar{C} = mean_barC after burn-in; 10 seeds; networks × payoff matrices × strategies)*

## 0. What we measured (links to RQ0 + H0)
**RQ0 (persistence):** For a given network, payoff matrix, and update rule, does the system reach **persistent (non-trivial) cooperation** rather than collapse to defection?

We summarize persistence using:
- **Long-run cooperation level:** `mean_barC` (your \bar{C}) after burn-in over an averaging window.
- **Fixation/absorption:** `Pr_allD`, `Pr_allC` if recorded.  
  - “Collapse to defection” ↔ high `Pr_allD` and `mean_barC ≈ 0`.
  - “Trivial all-cooperation” ↔ high `Pr_allC` and `mean_barC ≈ 1`.
  - “Persistent non-trivial cooperation” ↔ intermediate `mean_barC` and low fixation.

**Hypothesis H0.1 (existence):** Under sufficiently favorable payoff conditions and imitative dynamics, \bar{C} stays above a non-trivial level on structured networks.

---

## 1. Core patterns across payoffs (answers RQ0; evaluates H0.1)

### 1.1 Canonical (PD-like) payoff — cooperation mostly *not* persistent under payoff-responsive rules
**Observed:**  
- `Fermi(K=0.5)` produces **low** cooperation on all networks (roughly ~0.006–0.097).  
- `ImitationStrategy` also yields **low** cooperation overall (roughly ~0.06–0.17), with the highest value appearing on **WS_k8_p0.01** (~0.175).  
- `RLStrategy` is also **low** in this payoff (~0.05–0.06 range).  
- `ActionStrategy` is ~0.514 everywhere (see note below).

**Interpretation (RQ0):**  
For a Prisoner’s Dilemma incentive structure, the system tends toward **defection-dominant outcomes**, and sustained cooperation is difficult. The only notable “relative improvement” is that **more structured / locally clustered networks** (e.g., WS with very low rewiring) can sometimes support *slightly* higher cooperation under imitation, consistent with cluster protection intuition—but still not “high cooperation.”

**Hypothesis link:**  
This *partially supports* H0.1 only in the weak sense: structured networks help *some* update rules, but under strongly PD-like incentives, cooperation remains low and does not look robustly persistent.

---

### 1.2 Prisoners (strong PD) payoff — cooperation collapses almost completely
**Observed:**  
- `Fermi(K=0.5)` ≈ **0** across networks.  
- `ImitationStrategy` ≈ **0** across networks.  
- `RLStrategy` ≈ **0.05** across networks.  
- `ActionStrategy` again ≈ 0.514.

**Interpretation (RQ0):**  
This payoff matrix strongly favors defection; under imitation/Fermi dynamics, the system behaves like it reaches (or approaches) **all-D / near-all-D**. This is exactly what you’d expect in a strong PD regime.

**Hypothesis link:**  
This outcome **contradicts** the “persistence in PD” expectation unless “favorable conditions” are introduced (e.g., weaker temptation, different b/c, different noise, etc.). It supports the broader theoretical point: **payoff regime is the primary driver** of whether cooperation can persist.

---

### 1.3 Default payoff — mixed but still generally low cooperation under Fermi/Imitation; RL moderate
**Observed:**  
- `Fermi(K=0.5)` is mostly **very low**, but network-dependent (e.g., ~0.006–0.118).  
- `ImitationStrategy` is mostly **near 0**, with one larger case on WS_k8_p0.01 (~0.075).  
- `RLStrategy` is **moderate** and fairly stable across networks (~0.37–0.39).  
- `ActionStrategy` ≈ 0.514.

**Interpretation (RQ0):**  
Under this payoff matrix, imitation/Fermi still do not sustain high cooperation, but the RL learner maintains a **non-trivial** cooperation fraction. This suggests that your RL rule is not simply converging to “always defect” in this payoff and may be balancing exploration/exploitation to reach an intermediate behavior.

**Hypothesis link:**  
This gives a case where “non-trivial \bar{C}” exists (for RL), but it is not clearly driven by network clustering in this first sweep (values are similar across networks).

---

### 1.4 Snowdrift (coexistence game) — clear persistent intermediate cooperation
**Observed:**  
- `Fermi(K=0.5)` yields **intermediate** cooperation (~0.22–0.43), depending on network.  
- `ImitationStrategy` yields **moderate-to-high** cooperation (~0.33–0.56).  
  - Highest imitation levels appear on **WS** and **SBM strong**.
- `RLStrategy` is consistently **high-ish** (~0.53–0.57).  
- `ActionStrategy` ≈ 0.514.

**Interpretation (RQ0):**  
Snowdrift is a regime where cooperation and defection can coexist stably, so **persistent intermediate cooperation is expected**. Your results match this: \bar{C} is substantially above 0 for all payoff-responsive strategies, often around 0.4–0.6. This is strong evidence that **persistence depends heavily on payoff regime**.

**Hypothesis link:**  
This **supports** the “existence of persistent cooperation under favorable payoff conditions” part of H0.1.  
It also suggests that **network structure can modulate** the equilibrium level (e.g., WS/SBM yielding higher imitation-based cooperation than some other networks).

---

### 1.5 Friend or Foe — Fermi moderate, imitation near zero, RL moderate
**Observed:**  
- `Fermi(K=0.5)` produces **moderate** cooperation (~0.05–0.21), with **WS_k8_p0.01** highest (~0.205).  
- `ImitationStrategy` is **near 0** across networks.  
- `RLStrategy` is **moderate** (~0.35–0.36).  
- `ActionStrategy` ≈ 0.514.

**Interpretation (RQ0):**  
In this payoff, Fermi produces non-trivial cooperation in some networks (again WS low rewiring looks best), while pure imitation collapses. That contrast highlights that **update rule matters**: even with the same network and payoff, different dynamics can yield very different long-run behavior.

**Hypothesis link:**  
This gives partial support to “structured networks help,” but only under certain dynamics (Fermi here, not imitation).

---

## 2. Network effects (answers RQ about “which architectures suit cooperation?”; tests clustering hypothesis)

### 2.1 Consistent qualitative signal: **WS with low rewiring (p=0.01)** often boosts cooperation under imitation/Fermi
Across several payoff regimes, the **WS_k8_p0.01** condition tends to produce relatively higher \bar{C} for imitation and sometimes Fermi.

**Interpretation:**  
Low rewiring preserves local neighborhoods + clustering, allowing cooperators to form and maintain local support (cooperative clusters) more easily than in highly random networks.

### 2.2 SBM community structure: helps in Snowdrift imitation, not in PD
In Snowdrift, SBM strong shows high \bar{C} under imitation (~0.50).  
In Canonical/Prisoners, SBM does not rescue cooperation.

**Interpretation:**  
Communities can protect cooperative clusters *when the game allows coexistence*, but cannot overcome a strong PD incentive structure alone.

### 2.3 BA heterogeneity
BA shows:
- Canonical: Fermi/Imitation around ~0.09–0.096 (not outstanding),
- Snowdrift: imitation lower than many others (~0.33) while RL high (~0.57).

**Interpretation:**  
Degree heterogeneity may not automatically promote cooperation under imitation in your current setup; hubs can also amplify defection depending on payoff/update specifics.

---

## 3. Strategy/update-rule effects (answers “requirements for emergence?”)

### 3.1 ActionStrategy is not payoff-responsive → treat as baseline, not evidence of cooperation
You consistently see `ActionStrategy ≈ 0.51425` across *all* networks and payoffs.

**Interpretation:**  
This invariance indicates that ActionStrategy is likely generating near-random actions with a fixed bias (or fixed initial proportion) rather than adapting to payoffs. Therefore:
- It is a **control** / sanity baseline for “non-learning behavior,”
- It **should not** be used as evidence that cooperation persists due to payoff incentives.

### 3.2 Imitation vs Fermi
- Under PD-like payoffs: both tend toward low cooperation, but imitation sometimes benefits more from clustering (WS p=0.01).
- Under Snowdrift: imitation becomes strong (0.43–0.56), reflecting coexistence dynamics.

### 3.3 RLStrategy produces stable intermediate levels across multiple payoffs
RL shows:
- Default: ~0.37–0.39
- Friend or Foe: ~0.35–0.36
- Snowdrift: ~0.53–0.57
- Prisoners/Canonical: low (~0.05–0.06)

**Interpretation:**  
RL is adapting, but its learned behavior still depends on payoff regime—consistent with the general theory that incentives dominate.

---

## 4. Direct answers to research questions (v1 scope)

### RQ0: Does the system reach persistent (non-trivial) cooperation?
**Answer (v1):**  
- **Yes**, in payoff regimes that allow coexistence (notably **Snowdrift**) where \bar{C} is consistently moderate-to-high across networks and strategies (especially imitation/RL).  
- **Mostly no**, in strongly PD-like regimes (**Prisoners**, and largely **Canonical**) where Fermi/Imitation collapse to near-zero \bar{C} and RL stays low.

### RQ1: Which network architectures are more suited for emerging cooperation?
**Answer (v1):**  
- Evidence suggests **structured / clustered networks** (especially **WS with low rewiring**) can raise \bar{C} under imitation (and sometimes Fermi), relative to more random conditions—*but mainly when payoffs are not strongly PD-like*.
- SBM communities can increase \bar{C} in Snowdrift under imitation, suggesting community structure can help cluster persistence in coexistence games.

### RQ2: What are requirements for emergence?
**Answer (v1):**  
- **Payoff regime is the primary requirement**: Snowdrift supports coexistence → persistent cooperation emerges; strong PD does not.  
- **Update rule matters**: imitation can be fragile in some payoffs (Friend or Foe), while Fermi can maintain moderate cooperation there. RL yields intermediate behavior in multiple payoffs.

*(RQ3 phase transitions / percolation and RQ4 rewiring are not directly tested in this v1 table since you did not yet sweep parameters like T, b/c, K, w or compute component size statistics.)*

---

## 5. Hypotheses evaluation (what is supported / not supported)

### H0.1 (existence under favorable conditions + imitation + structured networks)
- **Supported** in favorable payoff regimes (Snowdrift): imitation yields high \bar{C}, especially on structured networks like WS low rewiring and SBM strong.
- **Not supported** in strong PD regimes (Prisoners, mostly Canonical): imitation does not maintain non-trivial \bar{C} even on structured networks.

### “Clustering helps” hypothesis (your earlier hypothesis)
- **Partially supported**: WS_k8_p0.01 often improves \bar{C} for imitation/Fermi compared to more random settings, but the effect is payoff-dependent and does not overcome strong PD incentives.

### “Cooperators connected to cooperators” / assortment hypothesis
- **Not directly tested yet**: you would need local assortment / covariance measurements or CC edge fractions to validate this (planned next).

---

## 6. Notes / caveats for v1 reporting
1. **ActionStrategy constant ~0.514** across all conditions indicates it is not payoff-driven; treat as baseline control rather than “successful cooperation.”
2. Your current summary is based on \bar{C} only. For a stronger RQ0 claim, include:
   - `Pr_allD`, `Pr_allC` (fixation probabilities),
   - time series plots C(t) for a few seeds to show stabilization vs collapse.
3. Because you ran only one K (0.5) for Fermi and fixed RL hyperparameters, “requirements” are preliminary; later sweeps over K, b/c (or T), and network parameters will strengthen conclusions.

---

## 7. Next steps (to align with remaining RQs)
- **Parameter sweeps** (RQ3): vary b/c (or T), K, and network parameters (WS p, SBM p_in/p_out) and look for sharp changes in \bar{C}, fixation rates, and cooperative component size.
- **Cluster/percolation metrics** (RQ3): compute S_max and cluster size distribution n_s(t).
- **Assortment metrics** (RQ2): compute edge-type fractions f_CC, f_CD, f_DD and local assortment/covariance between node action and neighbor actions.
