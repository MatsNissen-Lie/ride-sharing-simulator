# Ride-sharing code and master's thesis

The new master's thesis starts in [`thesis/main.tex`](thesis/main.tex).
It reuses the previous project's page layout and typography, with a new title
page and empty sections. Research content will be written after the literature
review establishes the direction. The source is self-contained for the editor.

Keep code and thesis changes together on each research branch. The earlier
TIØ4500 specialization project is preserved in
[`archive/previous-project-2025/`](archive/previous-project-2025/README.md).
That archive includes its LaTeX source, figures, compiled PDF, feedback, and a
file inventory. The original December 2025 simulation code is merged into this
repository, including the local configuration and plotting edits.

The historical simulation results and plots are in [`reports/`](reports/).
Exact vehicle samples are in [`data/vehicles/`](data/vehicles/).
See the archive README for the data requirements and validation results.

## Running the simulation

The baseline is the Alonso-Mora et al. (2017) batch assignment: RV graph, RTV
graph, assignment ILP and rebalancing, replayed on the Manhattan street graph.

One-time data preparation (gitignored outputs):

```sh
# NYC TLC trips for a month, e.g. January 2025, into data/riders/
curl -o data/riders/fhvhv_tripdata_2025-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2025-01.parquet
curl -o data/riders/yellow_tripdata_2025-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
uv run python scripts/data_prep/generate_rider_locations_multi.py  # snap trips to street nodes
uv run python scripts/compute_all_sp_fast.py                             # all-pairs travel times
# a year of TLC history for the default demand forecast (nowcast-past-day-of-week)
scripts/data_prep/download_tlc.sh 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01
uv run python scripts/data_prep/build_demand_history.py
```

Run (defaults in `config/config.py`; results go to `reports/<timestamp>_sim_standard/`):

```sh
uv run python main.py                    # trip-centric RTV builder, Gurobi
uv run python main.py --rtv vehicle      # vehicle-centric RTV builder (paper's Algorithm 1)
uv run python main.py --solver highs     # no Gurobi licence needed
uv run python main.py --hours 1 --fleet 1000
```

From Python (e.g. for experiment grids):

```python
from src.experiment import ExperimentConfig, run_experiment
run_experiment(ExperimentConfig(fleet=750, rebalancing="oracle", hours=1))
```

Each pipeline step is a swappable component behind an interface in
`src/interfaces/`, registered by name in `src/experiment/registry.py` and chosen by a
field of `ExperimentConfig`:

| Config field | Interface | Registry | Options |
|---|---|---|---|
| `controller` | `IBatchController` | `CONTROLLERS` | batch |
| `fleet_replay` | `IFleetReplay` | `FLEET_REPLAYS` | arrays, paths (same results; arrays faster) |
| `zoning` | `IZoneIndex` | `ZONINGS` | taxi |
| `demand_model` | `IDemandModel` | `DEMAND_MODELS` | historical |
| `fare_model` | `IFareModel` | `FARE_MODELS` | linear |
| `objective` | `IObjective` | `OBJECTIVES` | delay, profit |
| `trip_solver` | `ITripSolver` | `TRIP_SOLVERS` | astar |
| `rv_builder` | `IRVGraphBuilder` | `RV_BUILDERS` | standard |
| `rtv_builder` | `IRTVGraphBuilder` | `RTV_BUILDERS` | trip, vehicle |
| `assignment_solver` | `IAssignmentSolver` | `ASSIGNMENT_SOLVERS` | gurobi, highs, joint, sampled-* and two-stage, two-stage-chain (stage C design C2; two-stage is a documented negative result) |
| `vehicle_value` | `IVehicleValue` / `IFleetPlan` | `VEHICLE_VALUES` | zero, mpc, mpc_sampled (C3 over sampled demand futures), mpc_p1, mpc_half, mpc_flat, mpc_shuffled and mpc_flat_plan (placebos) (stage C design C3) |
| `request_sampler` | `IRequestSampler` | `REQUEST_SAMPLERS` | historical |
| `rebalancing` | `IRebalancingSolver` | `REBALANCERS` | reactive, forecast-greedy, forecast-optimal, mpc, oracle, none (old names demand, transport still work) |

To try a new variant, implement the interface in a new file and register it;
it is then selectable from `ExperimentConfig` (and the command line for the
main choices). Components talk to each other only through these interfaces
and the data models; see the ownership rules in `AGENTS.md`.

Tests compare each component with brute force on a small synthetic network:

```sh
uv run python -m pytest -q tests
```

Open `thesis/main.tex` in the Codex LaTeX editor to edit and preview the document.

For VS Code, open this repository folder and use the LaTeX Workshop extension.
The workspace settings build on save with two `pdflatex` passes and format with
`latexindent` using `thesis/.latexindent.yaml`. Build output goes to
`thesis/build/`. This recipe does not require `latexmk`.

The local TeX installation needs the packages used in `thesis/main.tex`. On the
current BasicTeX setup, these additional packages were installed in the user
TeX tree:

```sh
tlmgr --usermode install enumitem csquotes siunitx algorithms algorithmicx titlesec
```

# important docs
https://osmnx.readthedocs.io/en/stable/
https://networkx.org/documentation/stable/developer/index.html



# Project layout

```
config/                 settings (config.py), paths, logging
data/                   inputs: street graph, riders, zones, caches (large files git-ignored)
src/
  interfaces/           interfaces and data models every component depends on
  components/           implementations, one subpackage per role:
    simulation/           batch controller, fleet replay, data provider
    network/              street network service, trip solver, zones
    graph_builder/        RV and RTV graphs
    assignment/           paper's ILP and the stage C joint, sampled and two-stage solvers
    rebalancing/          reactive, forecast, oracle and fleet-plan rebalancers
    fleet_plan/           vehicle values and fleet plans (C3), placebos
    demand/               demand model, sampled future requests
    economics/            objectives, fares
  experiment/           ExperimentConfig, the registry of components by name, running a simulation
  inputs/               loading the Manhattan graph, sampling the starting fleet
  database/             all-pairs travel time cache (DuckDB)
  simulator.py          the batch loop
  utils/                plots, run output, statistics, timing
scripts/
  analysis/             analyses behind the numbers in the notes (summaries, Pareto, timing, ...)
  data_prep/            snapping trips to street nodes, plotting pickups
  compute_all_sp_fast.py, verify_db.py, with_sim_slot.sh
tests/                  brute-force checks on a small synthetic network
research/               ROADMAP (work log, decisions, experiments), stage C design notes, literature
thesis/                 working notes and figures
archive/                the 2025 project, restored for reference
reports/                output of runs (git-ignored; data of all runs on branch runs-archive)
main.py                 command line
```
