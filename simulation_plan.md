# Section 5: Computational Study - Simulation Plan

Based on the context established in your **Introduction** and **Solution Method**, the **Computational Study** (Section 5) needs to validate your baseline model and answer the first part of your research question: *"How can a baseline model for centralized ride-sharing be established using real-world data?"*.

It must demonstrate that your implementation of the Alonso-Mora framework (RV/RTV graphs + ILP) works on the Manhattan road network and establishes the "status quo" you promised in the contributions.

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

---

### Suggested Content for Subsection 5.1 (Drafting Start)

**5.1 Data Description and Experimental Setup**
To validate the proposed solution method, we utilize a historical dataset of taxi trips from New York City. As outlined in the objectives, Manhattan serves as the topological testbed due to its high density and grid-like road network.

**5.1.1 Road Network and Demand Data**
The simulation environment is built upon a directed graph representation of the Manhattan road network extracted from OpenStreetMap (OSM).

* **Nodes:** [Insert Number] (representing intersections).
* **Edges:** [Insert Number] (representing road segments).
Travel times are static and calculated based on free-flow speed limits, consistent with the foundational work by \textcite{alonsomora2017pnas}.

Demand is derived from the NYC Taxi & Limousine Commission (TLC) trip records. We selected [Date/Time Range] to represent a high-demand scenario. Raw GPS coordinates were mapped to the nearest graph nodes.

**5.1.2 Simulation Configuration**
The baseline model is executed with the following parameters, chosen to reflect standard urban mobility constraints:
\begin{table}[h]
\centering
\begin{tabular}{l|l}
\textbf{Parameter} & \textbf{Value} \\ \hline
Batch Duration ($\tau$) & 30 seconds \\
Vehicle Capacity ($C_{max}$) & 4 passengers \\
Max Wait Time ($\Omega$) & [e.g., 5 minutes] \\
Max Detour Factor ($\Delta$) & [e.g., 1.5x direct trip] \\
Fleet Size ($|V|$) & Varied [e.g., 100 - 3000] \\
\end{tabular}
\caption{Baseline Simulation Parameters}
\label{tab:sim_params}
\end{table}

---

**Would you like me to generate the LaTeX code for the "Performance Metrics" section next, defining the equations for Service Rate and Average Delay?**
