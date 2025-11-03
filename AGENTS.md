# project spesfication

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
