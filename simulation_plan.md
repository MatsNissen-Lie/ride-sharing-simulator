# Simulation Plan: Data & Plot Generation for Computational Study

**Goal:** Generate the specific plots, tables, and numerical results required for Section 5 (Computational Study) of the report. This document outlines the data extraction and visualization tasks needed to validate the baseline model.

The following outputs are required to demonstrate the Alonso-Mora framework's performance on the Manhattan road network.

Here is a recommended structure and content outline for Section 5:

---

### 5.1 Dataset and Experimental Setup

This section should detail the inputs mentioned in your contributions.

* **The Road Network:** Describe the graph-based representation of Manhattan. Mention the number of nodes and edges.
* **Request Data:** Specify the data source (likely NYC Taxi & Limousine Commission data).
* **Preprocessing:** How did you clean the data? (e.g., removing trips outside Manhattan, snapping GPS coordinates to graph nodes).
* **Demand Profile:** Show a plot of request volume over the simulation period (e.g., a 1-hour peak period vs. off-peak).
* **Simulation Parameters:** Define the constraints and variables used in the logic described in Section 4:
    * **Batch time ($\tau$):** 30 seconds.
    * **Fleet size ($|V|$):** What fleet sizes did you test?
    * **Capacity ($C_{max}$):** 4 passengers.
    * **Constraints:** Define the specific values used for maximum wait time ($\Omega$) and maximum detour delay ($\Delta$).

### 5.2 Performance Metrics

Define how you measure success. Since you aim to "establish a rigorous optimization baseline", you need standard metrics:

* **Service Rate:** The percentage of total requests served ($N_{served} / N_{total}$).
* **Wait Times:** Average time between request generation ($t^r$) and pickup.
* **Detour/Delay:** Average additional time passengers spent in the car compared to a direct trip.
* **Fleet Utilization:** Percentage of time vehicles are active vs. idle.
* **Occupancy Rate:** The average number of passengers in moving vehicles (validating the "ride-sharing" aspect ).

### 5.3 Computational Performance (Scalability Analysis)

This is crucial because your Solution Method spent significant effort detailing "pruning logic" and "vectorized operations". You must prove the system is computationally tractable.

* **Graph Sizes:** Report the average number of edges in the RV graph ($E_{RV}$) and RTV graph ($E_{RTV}$) per batch. This validates that your pruning works.
* **Runtime Breakdown:** Show the average time spent in each phase (Pairwise checking, Trip generation, ILP solve).
* *Goal:* Prove the total runtime is < $\tau$ (30s), making it a real-time system.

### 5.4 Baseline Results (The "Status Quo")

Present the main results of your simulation.

* **Service Quality vs. Fleet Size:** A classic plot showing that as Fleet Size increases, Service Rate increases and Wait Time decreases.
* **Occupancy Analysis:** Show how often rides were actually shared (e.g., % of trips with 1 vs. 2 vs. 3 passengers). This addresses the goal of reducing "low vehicle occupancy" mentioned in the intro.
* **Effect of Rebalancing:** If possible, compare a run *with* rebalancing vs. *without*. This validates the strategy described in Section 4.5.

### 5.5 Sensitivity Analysis (Optional but Recommended)

To fully address the "status quo" contribution, briefly show how the model reacts to changes.

* **Varying $\Omega$ and $\Delta$:** What happens if you allow passengers to wait longer? (Usually, service rate goes up, and occupancy goes up).

