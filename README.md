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
uv run python -m src.state_representation.generate_rider_locations_multi  # snap trips to street nodes
uv run python scripts/compute_all_sp_fast.py                             # all-pairs travel times
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
`src/interfaces/`, registered by name in `src/experiment.py` and chosen by a
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
| `assignment_solver` | `IAssignmentSolver` | `ASSIGNMENT_SOLVERS` | gurobi, highs, joint, sampled-* (stage C option B) |
| `vehicle_value` | `IVehicleValue` / `IFleetPlan` | `VEHICLE_VALUES` | zero, mpc, mpc_p1, mpc_half, mpc_flat (stage C option C) |
| `request_sampler` | `IRequestSampler` | `REQUEST_SAMPLERS` | historical |
| `rebalancing` | `IRebalancingSolver` | `REBALANCERS` | reactive, demand, transport, mpc, oracle, none |

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



# potential structure for the project
├── data/
│   ├── riders/
│   │   └── nyc_taxi_trips_2013.parquet
│   └── maps/
│       └── manhattan_graph.graphml
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_results_analysis.ipynb
│
├── src/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── data_models.py          # <-- Core Data classes
│   │   └── services.py             # <-- Routing Logic and Service Layer
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── network_service.py
│   │   ├── route_planner.py
│   │   ├── graph_builders.py
│   │   └── assignment_solvers.py
│   │
│   ├── data_loader/
│   │   ├── __init__.py
│   │   └── request_generator.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   └── state.py
│   │
│   └── main.py
│
├── tests/
│   ├── test_route_planner.py
│   └── test_request_generator.py
│
└── pyproject.toml
