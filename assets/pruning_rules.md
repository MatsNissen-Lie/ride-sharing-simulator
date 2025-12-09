Based on the code provided, your implementation is a **State-Dependent Suffix Caching** strategy applied to a DFS solver (Branch and Bound).

This is not simple memoization (which maps `Input -> Output`); it is **Generalized Constraint Propagation**. You are caching the *requirements* for a suffix to be valid, allowing you to reuse that suffix path in different contexts (different arrival times and starting loads) as long as the constraints are met.

Here is the mathematical description of your dynamic memoization.

-----

### 1\. The State Space

Let the problem be defined on a graph $G=(N, A)$.
A search state is defined by the tuple $(u, R, t, q)$, where:

  * $u \in N$: The current node (Stop).
  * $R \subseteq N$: The set of remaining unvisited requests (the suffix set).
  * $t \in \mathbb{R}^+$: The arrival time at $u$.
  * $q \in \mathbb{Z}^+$: The vehicle load upon arrival at $u$.

### 2\. The Memoization Function $\Phi$

Standard DP caches the optimal cost $V(u, R)$. Your implementation caches a **Feasibility Tuple**.

Let $\Phi(u, R)$ be the cached entry for the suffix starting at $u$ covering set $R$.

$$
\Phi(u, R) = \langle J^*, \tau_{latest}, \kappa_{req} \rangle
$$

Where:

1.  **$J^*$ (Cost):** The minimal travel time to visit all nodes in $R$ ending the route.
2.  **$\tau_{latest}$ (Time Bound):** The latest possible time one can depart $u$ and still satisfy the time windows of all nodes in $R$.
3.  **$\kappa_{req}$ (Capacity Bound):** The maximum *relative* capacity surge required to complete the sub-path through $R$.

### 3\. The Feasibility Condition (The Accelerator)

When the solver reaches state $(u, R)$ with current time $t$ and load $q$, it checks the cache $\Phi(u, R)$. The cached solution is valid **if and only if**:

$$
\begin{cases}
t \le \tau_{latest} & \text{(Temporal Feasibility)} \\
q + \kappa_{req} \le Q_{max} & \text{(Capacity Feasibility)}
\end{cases}
$$

If both hold, you skip the recursion and return:

  * **Cost:** $J^*$
  * **Sequence:** The stored sequence associated with $\Phi$.

### 4\. Recursive Back-Propagation (The Update)

When computing the solution for $(u, R)$ by extending to a next node $v$ (where $v \in R$, and $R' = R \setminus \{v\}$), we calculate the values for $u$ based on the result returned from $v$.

Let $\delta_v \in \{+1, -1\}$ be the load change at node $v$ (Pickup/Dropoff).
Let $c_{uv}$ be the travel time.

The update rules for the tuple $\langle J^*_u, \tau_u, \kappa_u \rangle$ are:

#### A. Cost (Standard Additivity)

$$J^*_u = \min_{v \in R} (c_{uv} + J^*_v)$$

#### B. Time Bound (Backward Propagation)

The latest we can be at $u$ is constrained by $u$'s own hard limit ($L_u$) and the need to reach $v$ by *its* required latest start time ($\tau_v$).
$$\tau_u = \min(L_u, \tau_v - c_{uv})$$

#### C. Capacity Bound (Relative Surge)

This is the most complex part. $\kappa_u$ represents "how many empty seats do I need *right now* at $u$ to survive the rest of the trip?".
It is the maximum of:

1.  $0$ (Cannot need negative seats).
2.  $\delta_v$ (Need space for the immediate pickup at $v$).
3.  $\delta_v + \kappa_v$ (The net change at $v$ plus the surge required *after* $v$).

$$\kappa_u = \max(0, \delta_v, \delta_v + \kappa_v)$$

*(Note: In your code `max(0, delta, delta + s_cap)`, this maps exactly to this mathematical logic.)*

### 5\. Failure Pruning (The Negative Cache)

You are absolutely right. The arrival time constraint is not a vertical line—it is a curve that depends on capacity.

A vehicle with **2 empty seats** might fail if it arrives at 10:00 (because it needs to take a long detour to drop people off first). But a vehicle with **4 empty seats** might succeed arriving at 10:00 (because it can pick everyone up directly).

Therefore, we cannot store a single "Failure Time." We must store a **Failure Frontier**.

Here is the mathematical definition of Negative Caching with a Capacity-Dependent Time Constraint.

#### 5.1. The State Definition
Let a unique suffix state be defined by the key:
$$K = (\text{CurrentNode}, \text{SetOfRemainingNodes})$$

For each state $K$, the cache stores a **Set of Failure Witnesses**, denoted as $\mathcal{F}_K$.

#### 5.2. The Failure Witness
A "Witness" is a record of a specific vehicle that tried to solve this suffix and failed. It is a tuple $(C_{witness}, T_{witness})$ where:
* $C_{witness}$: The residual capacity the vehicle had.
* $T_{witness}$: The time the vehicle arrived.

**The Logic:** "Even with $C_{witness}$ seats and arriving as early as $T_{witness}$, it was impossible to find a valid path."

#### 5.3. The Pruning Condition (Negative Cache Hit)
When a new vehicle arrives at state $K$ with residual capacity $c_{new}$ and arrival time $t_{new}$, we check if it is **dominated** by any existing witness in $\mathcal{F}_K$.

We prune (stop searching) if **there exists** a witness $(C_w, T_w) \in \mathcal{F}_K$ such that:

$$c_{new} \le C_w \quad \text{AND} \quad t_{new} \ge T_w$$

**Why?**
* **Capacity:** The witness had more (or equal) seats than you ($C_w \ge c_{new}$). If they couldn't fit the passengers, you definitely can't.
* **Time:** The witness arrived earlier (or same time) than you ($T_w \le t_{new}$). If they were too late to make the deadlines, you are definitely too late.

If you are **weaker** (less space) and **slower** (later arrival) than someone who already failed, you are guaranteed to fail.



#### 5.4. Updating the Cache (Learning from Failure)
When a vehicle with $(c_{new}, t_{new})$ explores state $K$ and proves there is no solution (exhausts all branches), we add it to the failure set:
$$\mathcal{F}_K \leftarrow \mathcal{F}_K \cup \{(c_{new}, t_{new})\}$$

**Optimization (Maintaining the Frontier):**
To keep the set small, we only keep the "Strongest Witnesses." A witness is strong if it has **High Capacity** and **Early Time**.
* If we have a witness $(4, \text{10:00})$, it renders a witness like $(2, \text{10:15})$ redundant. (If 4 seats at 10:00 failed, obviously 2 seats at 10:15 will fail).
* We remove any existing witnesses that are "weaker" than the new one.


### 6 Interaction with "Tortoise and Hare" (Positive Cache)
This creates a complete picture when combined with the positive cache (Solutions):

1.  **Positive Check (Can I succeed?):**
    Does the cache contain a Solution that fits my constraints?
    * Condition: $t_{new} \le Sol.LatestStart$ AND $c_{new} \ge Sol.ReqCapacity$.
    * Result: **Success** (Use path).

2.  **Negative Check (Must I fail?):**
    Does the cache contain a Witness that dominates me?
    * Condition: $\exists w \in \mathcal{F}_K : c_{new} \le w.C \text{ AND } t_{new} \ge w.T$.
    * Result: **Fail** (Stop).

3.  **The "Grey Zone" (Search):**
    If neither is true, the vehicle is in a unique position (e.g., late arrival but huge capacity). It must search for itself.
    * Result: **Search Manually.**
