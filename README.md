# Ride-sharing code and master's thesis

The new master's thesis starts in [`thesis/main.tex`](thesis/main.tex).
It reuses the previous project's page layout and typography, with a new title
page and empty sections. Research content will be written after the literature
review establishes the direction. The source is self-contained for the editor.

Keep code and thesis changes together on each research branch. The earlier
TIØ4500 specialization project remains in the separate `latex-ride-sharing`
repository as reference material.

Open `thesis/main.tex` in the Codex LaTeX editor to edit and preview the document.

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
