![Python version](https://img.shields.io/badge/python-3.10–3.11-blue.svg)

# boat

A toolkit for one-shot (low-iteration) sequence optimization workflows in biologics R&D.
It combines:
- Sequence encodings (PLMs, bio-specific embeddings)
- Bayesian & evolutionary optimization loops
- Liability filtering
- Scoring interfaces to internal / external models (e.g. PLMs)
- Modular acquisition & model abstractions for rapid experimentation

The goal: enable fast design-iteration cycles with pluggable scoring functions and flexible optimization strategies.

## Key Features (overview)
- Bayesian optimization (single & multi-objective) with BoTorch / GPyTorch
- Genetic algorithm framework for sequence-level search
- Encodings: physicochemical, PLM-based, antibody numbering, etc.
- Liability and developability scoring utilities
- Pluggable scoring interfaces (fake, PLM, Oasis, liabilities)

## Installation

Requires: Python 3.10 or 3.11 (see pyproject). Poetry is used for dependency management.

1. Install Poetry (if needed).

2. Standard install (core only):
   poetry install

3. Install with selected extras, e.g.:
```bash
poetry install --extras "bayesopt"
```
4. Activate virtual environment:
   poetry shell
   or run commands with:
   poetry run python ...

## Optional Extras (summary)

- bayesopt: Bayesian optimization stack (ablang2, blosum, botorch, gpytorch, scikit-learn)

Install any combination via:
```bash
poetry install --extras "<space separated extras>"
```
## Quick Start

Example (pseudo) usage sketch:

```python
from boat.bayesopt.mo_loop import MultiObjectiveLoop
from boat.scoring_function.fake import FakeScoringFunction

loop = MultiObjectiveLoop(
    scoring_functions=[FakeScoringFunction()],
    n_init=8,
    n_iter=5,
)
loop.run()
```

Replace FakeScoringFunction with real interfaces (PLM, etc.) as configured.

## Project Organization

```
├── .github/workflows        # CI workflows (lint, test, build, publish, docs)
├── Makefile                 # Common developer shortcuts
├── Dockerfile               # Base container recipe
├── README.md
├── data/                    # (Git-ignored) local or mounted datasets
├── docs/                    # MkDocs documentation project
├── models/                  # Serialized models / artifacts (Git LFS / ignored as needed)
├── notebooks/               # Exploratory / analysis notebooks
├── pyproject.toml           # Poetry configuration & extras
└── boat/
    ├── data_utils.py        # Generic data helpers
    ├── bayesopt/            # Bayesian optimization components
    │   ├── mo_loop.py       # Multi-objective loop orchestration
    │   ├── acquisition/     # Acquisition strategies & utilities
    │   ├── encodings/       # Feature encodings for sequences
    │   ├── loop/            # Core loop utilities
    │   └── models/          # GP models, kernels, wrappers
    ├── biologics/           # Domain-specific sequence & liability helpers
    ├── genetic_algorithm/   # GA operators, optimizers, vocabularies
    ├── scoring_function/    # Unified scoring interfaces (fake, PLM, Oasis, liabilities)
    └── __init__.py
```

### Subpackage Highlights
- bayesopt: Acquisition functions, GP kernels, loops for sequential / multi-objective optimization.
- genetic_algorithm: Mutation / crossover / population management for sequence search.
- scoring_function: Abstraction layer to plug different scoring backends uniformly.
- biologics: Sequence manipulation, liabilities and developability heuristics.


## Troubleshooting

- Missing optional features: confirm you installed correct extras.
