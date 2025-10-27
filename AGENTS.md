# project spesfication

## Core Directive: 
This project exclusively uses uv for all Python package management and script execution.

## Runtime Assumptions:
Target runtime is Python 3.12, so take advantage of native typing features without requiring `from __future__ import annotations` or backports.

## Terminal Context:
Assume the active terminal is already positioned at the project root; avoid running extra `cd` commands before executing uv operations.

## Dependency Management: 
To add new packages, you must use the uv add <package-name> command from the project root. You do not need to cd into any subdirectories.

## Code Execution: 
All scripts and commands must be executed using uv run path/to/python_file.py.

## Testing Guidance:
When a file has already had its relevant tests executed during this session, you don't need to rerun the same tests again after making additional edits to that file unless the change affects test coverage or functionality.