# Ride-sharing simulator for Manhattan

This simulator replays real New York taxi and Uber/Lyft trips on the Manhattan
street graph. Every 10 seconds it assigns the waiting riders to shared
vehicles (up to 4 seats) and moves idle vehicles toward demand.

The baseline is the batch assignment method of Alonso-Mora et al. (2017):
an RV graph, an RTV graph, an assignment ILP and a rebalancing step. Each step
is a component that you can replace. Thus you can compare different methods on
the same trips and the same fleet.

## What you can compare

- **Assignment**: the paper's ILP (Gurobi or HiGHS), a joint ILP that also
  moves empty vehicles, and variants that add sampled future riders.
- **Rebalancing**: the paper's reactive step, forecast-based steps, a step
  that follows a zone-level fleet plan, and a perfect-information benchmark.
- **Demand forecasts**: monthly averages, past-only history, a nowcast and
  gradient boosting.
- **Objectives**: rider delay or operator profit.
- **Fleet size, time window and traffic speed**.

Each run writes per-batch statistics, plots and its full configuration, so
you can compare runs with the scripts in `scripts/analysis/`.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), about 4 GB of
free memory for a full Manhattan run. The street graph and the taxi zones are
in the repository. The trip data is not.

### 1. Install

```sh
git clone https://github.com/MatsNissen-Lie/ride-sharing-simulator
cd ride-sharing-simulator
uv sync
```

### 2. Get the trips and build the travel-time cache

```sh
# NYC TLC trip records for January 2025 (Uber/Lyft and yellow cabs), into data/riders/
OUT_DIR=data/riders scripts/data_prep/download_tlc.sh 2025-01

# Snap each trip to street nodes; writes data/riders/trips_2025-01_enriched*.parquet
uv run python scripts/data_prep/generate_rider_locations_multi.py

# All-pairs travel times and distances on the street graph; writes data/duckDB/route_db.duckdb
uv run python scripts/compute_all_sp_fast.py
```

The rider script reads only `fhvhv_tripdata_2025-*` and
`yellow_tripdata_2025-*` files.

### 3. Run one hour with HiGHS (no licence needed)

```sh
uv run python main.py --solver highs --start 2025-01-16T17:45 --hours 1 --fleet 1000
```

This simulates 15 minutes of warm-up and then one measured hour, with 1000
vehicles, from Thursday 16 January 2025 at 17:45. The results go to
`reports/<timestamp>_sim_standard/`.

Without `--hours`, a run simulates 48 hours. Without `--solver`, the run uses
Gurobi, which needs a licence for problems of this size.

### 4. Optional: a year of demand history

The default demand forecast (`nowcast-past-day-of-week`) and the other
`past-*` and `gbm` forecasts use trip counts from the year before the
simulated day. Only the forecast-based rebalancers and the look-ahead solvers
use a forecast. The baseline run above does not.

```sh
scripts/data_prep/download_tlc.sh 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 \
  2024-07 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01     # into data/history/raw/
uv run python scripts/data_prep/build_demand_history.py       # writes data/history/zone_counts_15min.parquet
```

Without this history, add `--demand-model historical`. That model uses the
other days of the month in `data/riders/`.

## More examples

```sh
uv run python main.py --solver highs --rtv vehicle                  # per-vehicle RTV builder (paper's Algorithm 1)
uv run python main.py --solver highs --rebalancing oracle           # rebalance to the actual future requests
uv run python main.py --solver highs --rebalancing forecast-optimal --demand-model historical
uv run python main.py --solver joint --vehicle-value mpc --rebalancing mpc
uv run python main.py --objective profit --cost-per-km 0.25 --solver highs
uv run python main.py --help                                        # all options
```

From Python, for example to run a grid of experiments:

```python
from src.experiment import ExperimentConfig, run_experiment

run_dir = run_experiment(
    ExperimentConfig(fleet=750, assignment_solver="highs", rebalancing="oracle", hours=1)
)
```

## Configuration

Two places hold the settings:

- `config/config.py`: fixed constants. Examples are the maximum wait (3 min),
  the maximum delay (5 min), the batch length (10 s), the vehicle capacity, the
  fare model, the traffic factor and the default time window.
- `src/experiment/config.py`: `ExperimentConfig`, the components and settings
  of one run. `main.py` exposes most fields as command-line options.

### Components

Each step of the pipeline is an interface in `src/interfaces/`. The
implementations are registered by name in `src/experiment/registry.py`. A
field of `ExperimentConfig` selects one by name.

| Config field (CLI option) | Options | What the options do |
|---|---|---|
| `assignment_solver` (`--solver`) | `gurobi`, `highs` | The paper's assignment ILP, solved with Gurobi or with HiGHS (through SciPy). |
| | `joint` | One ILP (HiGHS) that chooses, for each vehicle, a trip, an empty move to a nearby zone, or no change. It uses `--vehicle-value` to score where each option leaves the vehicle. |
| | `sampled`, `sampled-chain` | The joint ILP plus virtual future riders sampled from the demand forecast. A virtual rider counts with a small weight (`--virtual-weight`). `-chain` also lets a vehicle serve a virtual rider after its current trip. |
| | `sampled-matched`, `sampled-bidir`, `sampled-lsa` and their `-chain` forms | Variants of `sampled` that offer about one virtual rider per free vehicle (`--virtual-per-free`). They differ in how they choose the vehicle–rider pairs. |
| | `two-stage`, `two-stage-chain` | A two-stage stochastic program. It decides now and scores each decision against several sampled futures (`--scenarios`). |
| `rebalancing` (`--rebalancing`) | `reactive` | The paper's step: send idle vehicles to unassigned riders (minimum total travel time). |
| | `forecast-greedy` | Serve unassigned riders first, then spread idle vehicles over zones in proportion to the forecast demand. |
| | `forecast-optimal` | Serve only the riders a vehicle can still reach in time, then spread idle vehicles like the forecast with the fewest moves. |
| | `mpc` | Carry out the empty moves of the zone-level fleet plan. Use it with `--vehicle-value mpc`. |
| | `oracle` | Send idle vehicles to where requests will actually appear in the next minutes. This is a benchmark, not a method. |
| | `none` | No rebalancing step (for solvers that move vehicles themselves). |
| `vehicle_value` (`--vehicle-value`) | `zero` | No look-ahead. With `zero`, `--solver joint` gives the same result as the paper's ILP. |
| | `mpc` | Value of a free vehicle per zone and time, from the dual prices of a fleet linear program over zones. The LP is solved again every minute. |
| | `mpc_sampled` | As `mpc`, but averaged over several sampled demand futures (`--fleet-futures`), solved in parallel. |
| | `mpc_p1`, `mpc_half` | `mpc` with other pooling or weight settings. |
| | `mpc_flat`, `mpc_flat_plan`, `mpc_shuffled` | Control variants of `mpc`. They average the values over all zones or shuffle them between zones. Use them to check how much of the effect of `mpc` comes from location. |
| `demand_model` (`--demand-model`) | `historical`, `day-of-week` | Average per 15-minute slot and taxi zone over the other days of the month (weekday/weekend, or per day of the week). |
| | `past-weekday`, `past-day-of-week`, `gbm` | Use only history from before the simulated day (`--history-weeks`, or gradient boosting). These need the demand history. |
| | `nowcast`, `nowcast-past-day-of-week`, `nowcast-gbm` | Scale a base forecast by the demand seen so far today (`--nowcast-window`). |
| | `oracle` | The actual requests of the day (upper bound). |
| `objective` (`--objective`) | `delay`, `profit` | Minimise rider delay plus a penalty for unserved riders, or maximise fares minus cost per vehicle-km. |
| `rtv_builder` (`--rtv`) | `trip`, `vehicle` | Build the RTV graph trip by trip, or vehicle by vehicle as in the paper's Algorithm 1. |
| `fleet_replay` (`--fleet-replay`) | `arrays`, `paths` | How vehicles move between batches. Both give the same results. `arrays` is faster. |
| `request_sampler` (`--request-sampler`) | `historical` | Source of virtual future riders for the `sampled` and `two-stage` solvers. |
| `trip_solver`, `rv_builder`, `zoning`, `fare_model`, `controller` | one option each | `astar` trip routing, the standard RV graph, NYC taxi zones, a linear fare model, the batch controller. |

To add a variant, implement the interface in a new file and register it in
`src/experiment/registry.py`. Then `ExperimentConfig` can select it. Components
use only the interfaces and the data models in `src/interfaces/`.

## Output

Each run writes a folder `reports/<timestamp>_<name>/` (set `<name>` with
`--name`):

- `simulation_stats.csv`: one row per measured batch: requests, riders served,
  wait and delay, idle vehicles, kilometres, revenue and solver times.
- `metadata.json`: the constants from `config/config.py` and the full
  `ExperimentConfig`.
- `timing.csv`: where the run spent its time.
- PNG plots: delay over time, fleet use, throughput, request volume, graph
  size and run time.
- `profile.prof` with `--profile`.

To summarise or compare runs:

```sh
uv run python scripts/analysis/summarize_runs.py reports/*<name>*
uv run python scripts/analysis/timing_report.py reports/<run_a> reports/<run_b>
```

## Tests

The tests use small synthetic networks. They do not need the trip data or a
Gurobi licence.

```sh
uv run python -m pytest -q tests
```

## Project layout

```
config/                 constants (config.py), paths, logging
data/                   street graph and taxi zones; trip data and caches are git-ignored
src/
  interfaces/           interfaces and data models that all components use
  components/           implementations, one subpackage per role
    simulation/           batch controller, fleet replay, data provider
    network/              street network service, trip routing, zones
    graph_builder/        RV and RTV graphs
    assignment/           assignment ILP, joint, sampled and two-stage solvers
    rebalancing/          reactive, forecast, fleet-plan and oracle rebalancing
    fleet_plan/           zone-level fleet plan and vehicle values
    demand/               demand forecasts and sampled future requests
    economics/            objectives and fares
  experiment/           ExperimentConfig, the component registry, running a simulation
  inputs/               loading the street graph, placing the starting fleet
  database/             travel-time cache (DuckDB)
  simulator.py          the batch loop
  utils/                plots, run output, statistics, timing
scripts/
  data_prep/            download TLC data, snap trips to street nodes, build demand history
  analysis/             summaries and comparisons of runs, model calibration
  compute_all_sp_fast.py  all-pairs travel times
tests/                  tests on small synthetic networks
reports/                run output (git-ignored)
main.py                 command line
```

## Reference

J. Alonso-Mora, S. Samaranayake, A. Wallar, E. Frazzoli and D. Rus,
"On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment",
*PNAS* 114(3), 2017.

## Licence

MIT. See `LICENSE`.
