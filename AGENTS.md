# project spesfication

## USER PREFERENCES (IMPORTANT):
- If new code is added, assume it is written by THE MASTER DEVELOPER. ALWAYS ASK BEFORE DELETING NEW CODE.
- MINIMAL CODE IS GOOD CODE.
- FOCUS ON A NARROW SCOPE AND DO THAT WELL. AVOID BIG CHANGES THAT TOUCH MANY PARTS OF THE CODEBASE.
- PRESENT THE MASTER DEVELOPER WITH A PLAN IF THE SCOPE OF WORK IS MEDIUM OR LARGER. READ FILES CAREFULLY BEFORE SUGGESTING A PLAN.
- KEEP THE CODE MODULAR AND CONFIGURABLE (most important): every pipeline component (controller, zoning, demand model, fare model, objective, trip solver, RV/RTV builder, assignment solver, vehicle value, rebalancing) sits behind an interface, is registered by name in `src/experiment/registry.py` and is chosen in `ExperimentConfig`, so it can be swapped or removed to run experiments. New behaviour goes in a new implementation of an interface, not in flags inside an existing one.
- OWNERSHIP AND ISOLATION: a component never calls another component's private members (`_name`) or reads its constants; it depends on interfaces, not concrete classes, and receives its collaborators from the registry (no constructing another component inside). Helpers shared by several components go in a shared module; the meaning of data (e.g. what kind of stop a stop is) belongs to the data model in `src/interfaces/`. Do not edit a component you are not working on; if you need a hook in it, add the smallest one, test it, and say so.
- Document as you go, in the same commit as the work:
  - `research/ROADMAP.md`: add a dated line to the Work log for every piece of work, record decisions in the Decisions table, and experiment results under Experiments.
  - `thesis/working_notes.tex` (temporary section of the thesis): key words, findings, numbers and figures. Every data analysis or modelling choice gets a method entry: data, measure (formula), result, how it is used in the simulator, limitations, and the script that reproduces it.
  - Numbers in the notes must come from a script in the repo (e.g. `scripts/analysis/`), not from one-off commands.

## Core Directive: 
This project exclusively uses uv for all Python package management and script execution.

## Runtime Assumptions:
Target runtime is Python 3.12, so take advantage of NATIVE TYPING features. DO NOT USE `from __future__ import annotations` or backports.

## Terminal Context:
Assume the active terminal is already positioned at the project root; avoid running extra `cd` commands before executing uv operations.

## Dependency Management: 
To add new packages, you must use the uv add <package-name> command from the project root. You do not need to cd into any subdirectories.

## Code Execution: 
All scripts and commands must be executed using uv run path/to/python_file.py.

## Logging
Modules call `config.get_logger(__name__)` to get INFO-level, module-scoped stream handlers while the root stays at WARNING; `PolygonApiFetcher` still uses stdlib `logging.basicConfig` with the shared level constant.

## Paths & Imports
Use `config.BASE_DIR` and `pathlib.Path` chaining for filesystem work, prefer absolute `src.*` imports, and keep packages importable via `__init__.py` stubs.


## Context7 Documentation Lookup
- Resolve with `context7.resolve-library-id` then fetch via `context7.get-library-docs`; use `/websites/osmnx_readthedocs_io_en_stable` for OSMnx, `/websites/networkx_stable` for NetworkX.
- For Polars use `/websites/pola_rs_api_python_stable_reference` after resolving.

## POLARS is the default dataframe library (DO NOT USE PANDAS)

