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
