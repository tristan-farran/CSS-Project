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
Under this payoff matrix, imitation/Fermi still do not sustain high cooperation, but the RL learner maintains a **non-trivial** cooperation fraction. This suggests that the RL rule is not simply converging to “always defect” in this payoff and may be balancing exploration/exploitation to reach an intermediate behavior.

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
- It is a **control** baseline for “non-learning behavior,”
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

## 4. Direct answers to research questions

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


---

## 5. Hypotheses evaluation (what is supported / not supported)

### H0.1 (existence under favorable conditions + imitation + structured networks)
- **Supported** in favorable payoff regimes (Snowdrift): imitation yields high \bar{C}, especially on structured networks like WS low rewiring and SBM strong.
- **Not supported** in strong PD regimes (Prisoners, mostly Canonical): imitation does not maintain non-trivial \bar{C} even on structured networks.

### “Clustering helps” hypothesis 
- **Partially supported**: WS_k8_p0.01 often improves \bar{C} for imitation/Fermi compared to more random settings, but the effect is payoff-dependent and does not overcome strong PD incentives.

---

## 6. Time to absorption (fixation speed) across conditions

### 6.1 What `mean_time_to_absorption` means in our runs
- `mean_time_to_absorption` is the **average number of rounds until the system reaches an absorbing state** (all-C or all-D) **for those runs that actually absorbed**.
- If `mean_time_to_absorption` is **blank/NaN**, it means **no absorbing state was reached within the simulated horizon** in that condition (i.e., the process remained in a mixed state for the entire run). This should be treated as a **right-censored** absorption time: “time to absorption > simulation length”.

Because absorption may occur only in some seeds, it is important to interpret absorption together with:
- `Pr_allD`, `Pr_allC` (how often absorption happened and which absorbing state),
- `mean_barC` (whether the non-absorbing runs maintain non-trivial cooperation).

### 6.2 High-level patterns (across payoffs and strategies)

**(A) Snowdrift: no absorption observed**  
Across all networks and payoff-responsive strategies (Imitate / Fermi / RL), `Pr_allD = Pr_allC = 0` and `mean_time_to_absorption` is NaN, consistent with a **coexistence regime** where mixed states persist.

**(B) Strong PD (Prisoners): absorption happens quickly**  
- Under **Imitation**, absorption to all-D is very fast: mean time ≈ **12 rounds** (11–14 across networks).  
- Under **Fermi(K=0.5)**, absorption also occurs frequently but more slowly: mean time ≈ **64 rounds** (≈43–94 depending on network).  
This aligns with the idea that strong PD incentives create a strong pull toward defection, and payoff-based update rules reach fixation rapidly.

**(C) Weak-PD-like (Default, Friend or Foe) and PD-like (Canonical): absorption is slower, especially for Fermi**  
- **Fermi(K=0.5)** shows frequent all-D absorption but often only after **hundreds of rounds** (≈224–317 on average, payoff-dependent).
- **Imitation** (when it absorbs at all) does so quickly (≈18–27 rounds) and only in some networks.

**(D) RL and Action show no absorption in this experiment window**  
For RL (and Action baseline), `Pr_allD = Pr_allC = 0` and `mean_time_to_absorption` is NaN across payoffs/networks, meaning these dynamics **did not fixate** within the run length used in this sweep.

### 6.3 Quantitative summary tables (absorption frequency + speed)

### Canonical — absorption frequency and time

| Strategy | mean \bar{C} | mean Pr(all-D) | mean Pr(all-C) | mean Pr(absorb) | mean time to absorption | min | max | networks with absorption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action | 0.514 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Imitate | 0.090 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Fermi | 0.029 | 0.889 | 0.011 | 0.900 | 224.1 | 158.7 | 302.0 | 9/9 |
| RL | 0.055 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |

### Default — absorption frequency and time

| Strategy | mean \bar{C} | mean Pr(all-D) | mean Pr(all-C) | mean Pr(absorb) | mean time to absorption | min | max | networks with absorption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action | 0.514 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Imitate | 0.020 | 0.089 | 0.000 | 0.089 | 24.1 | 21.0 | 27.0 | 5/9 |
| Fermi | 0.045 | 0.722 | 0.000 | 0.722 | 258.4 | 182.3 | 352.3 | 9/9 |
| RL | 0.379 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |

### Friend or Foe — absorption frequency and time

| Strategy | mean \bar{C} | mean Pr(all-D) | mean Pr(all-C) | mean Pr(absorb) | mean time to absorption | min | max | networks with absorption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action | 0.514 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Imitate | 0.014 | 0.133 | 0.000 | 0.133 | 20.8 | 18.0 | 24.0 | 4/9 |
| Fermi | 0.118 | 0.511 | 0.000 | 0.511 | 316.8 | 248.5 | 383.3 | 9/9 |
| RL | 0.358 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |

### Prisoners — absorption frequency and time

| Strategy | mean \bar{C} | mean Pr(all-D) | mean Pr(all-C) | mean Pr(absorb) | mean time to absorption | min | max | networks with absorption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action | 0.514 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Imitate | 0.002 | 0.711 | 0.000 | 0.711 | 12.0 | 11.0 | 14.0 | 9/9 |
| Fermi | 0.000 | 0.922 | 0.000 | 0.922 | 64.3 | 43.5 | 94.1 | 9/9 |
| RL | 0.051 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |

### Snowdrift — absorption frequency and time

| Strategy | mean \bar{C} | mean Pr(all-D) | mean Pr(all-C) | mean Pr(absorb) | mean time to absorption | min | max | networks with absorption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action | 0.514 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Imitate | 0.477 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| Fermi | 0.380 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |
| RL | 0.554 | 0.000 | 0.000 | 0.000 |  |  |  | 0/9 |


### 6.4 Network effects on absorption time (where absorption was observed)

Below are the **fastest** and **slowest** absorption networks (based on the reported `mean_time_to_absorption`) for conditions that reached absorption:

- **Canonical / Fermi**: fastest = `BA_m4` (158.7 rounds, Pr(all-D)=0.9); slowest = `WS_k8_p0.01` (302.0 rounds, Pr(all-D)=0.8).
- **Default / Fermi**: fastest = `BA_m4` (182.3 rounds, Pr(all-D)=1.0); slowest = `WS_k8_p0.01` (352.3 rounds, Pr(all-D)=0.3).
- **Default / Imitate**: fastest = `BA_m4` (21.0 rounds, Pr(all-D)=0.1); slowest = `ER_p0.02` (27.0 rounds, Pr(all-D)=0.1).
- **Friend or Foe / Fermi**: fastest = `BA_m4` (248.5 rounds, Pr(all-D)=0.8); slowest = `Grid_20x20` (383.3 rounds, Pr(all-D)=0.7).
- **Friend or Foe / Imitate**: fastest = `BA_m4` (18.0 rounds, Pr(all-D)=0.1); slowest = `WS_k8_p0.5` (24.0 rounds, Pr(all-D)=0.1).
- **Prisoners / Fermi**: fastest = `ER_p0.015` (43.5 rounds, Pr(all-D)=0.6); slowest = `SBM_2block_strong` (94.1 rounds, Pr(all-D)=1.0).
- **Prisoners / Imitate**: fastest = `BA_m4` (11.0 rounds, Pr(all-D)=0.9); slowest = `WS_k8_p0.01` (14.0 rounds, Pr(all-D)=0.1).

**Interpretation:**  
- In multiple payoff regimes, **WS_k8_p0.01 (high clustering / low rewiring)** tends to **delay** absorption under Fermi, and often coincides with higher `mean_barC`. This is consistent with “cluster protection”: cooperators can survive longer (or indefinitely) in locally clustered neighborhoods.
- In contrast, **BA_m4** frequently shows **faster** absorption times (especially for Fermi/Imitation in Canonical/Default/Friend-or-Foe), suggesting that degree heterogeneity (hubs) can accelerate convergence to fixation under certain update rules.

### 6.5 How this strengthens the RQ0 conclusion (“can cooperation persist?”)

Combining `mean_barC` with absorption timing gives a sharper answer:

- **Persistent cooperation** is most convincingly supported when:
  - `mean_barC` is **intermediate/high**, **and**
  - `Pr_absorb` is **low**, with `mean_time_to_absorption` **not observed** (NaN) → indicating a stable mixed regime within the simulation horizon.  
  This is exactly what we see in **Snowdrift** across networks.

- **Collapse to defection** is supported when:
  - `Pr_allD` is high, **and**
  - absorption happens **quickly** (small mean time), particularly in **Prisoners** under imitation (~12 rounds) and Fermi (~64 rounds).

- **Slow drift toward fixation** appears when:
  - `Pr_allD` is moderate-to-high but absorption takes **hundreds** of rounds (Canonical/Default/Friend-or-Foe under Fermi).  
  Here, some runs remain mixed for a long time, but the long-run trend still favors defection in many seeds.
