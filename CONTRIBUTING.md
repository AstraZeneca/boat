# Contributing

Thanks for your interest in contributing to `mlab-oneshot-active-learning`.

This guide explains how to contribute through GitHub Issues and Pull Requests (PRs).

## Ways to Contribute

- Report bugs
- Propose improvements
- Improve documentation
- Add tests
- Submit code fixes or new features

## Before You Start

1. Search existing GitHub Issues and PRs to avoid duplicates.
2. Prefer small, focused changes over large mixed PRs.
3. For behavior changes, open an issue first so scope and approach can be aligned.

## Creating a GitHub Issue

Open an issue when you find a bug, want a feature, or need design discussion.

Use a clear title and include:

- Problem summary
- Current behavior
- Expected behavior
- Reproduction steps (for bugs)
- Relevant logs, traceback, screenshots, or sample inputs
- Environment details (OS, Python version, package versions)

### Bug Report Checklist

- Minimal reproducible example
- Full error message and traceback
- What you already tried
- Whether this is a regression (worked before, broken now)

### Feature Request Checklist

- User/problem statement
- Proposed solution
- Alternatives considered
- Potential impact on APIs, configs, or workflows

## Pull Request Workflow

### 1. Set Up Locally

From the repository root:

```bash
poetry install
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

Or use the helper target:

```bash
make init
```

### 2. Create a Branch

Use a descriptive branch name:

```bash
git checkout -b <type>/<short-description>
```

Examples: `fix/encoder-shape-check`, `feat/new-acquisition-heuristic`, `docs/contributing-guide`.

### 3. Make Your Changes

- Keep changes scoped to one topic.
- Add or update tests when behavior changes.
- Update docs when user-facing behavior changes.

### 4. Run Checks Before Opening a PR

Run formatting, linting, and tests locally:

```bash
poetry run ruff check .
poetry run black --check .
poetry run pytest
```

If relevant, include coverage-sensitive tests for changed modules.

### 5. Commit Clearly

Write commit messages in the imperative mood with clear intent.

Good examples:

- `fix: handle empty sequence batch in encoder`
- `test: add regression test for GA mutation edge case`
- `docs: clarify bayesopt extra installation`

### 6. Open the Pull Request

In your PR description, include:

- What changed
- Why it changed
- Linked issue(s) (for example: `Closes #123`)
- Test evidence (commands run and outcomes)
- Any breaking changes or migration notes

Keep PRs reviewable. Smaller PRs are merged faster and with fewer regressions.

## PR Review Expectations

- Maintainers may request changes before merge.
- Please respond to review feedback and keep discussion in the PR thread.
- Force-push is acceptable on your branch; avoid rewriting shared branches.

## Definition of Done

A contribution is generally ready to merge when:

- CI passes
- Relevant tests exist and pass
- Docs are updated if needed
- Reviewer feedback is addressed

## Security and Sensitive Data

- Do not commit secrets, credentials, internal tokens, or private datasets.
- If you discover a security issue, report it privately to maintainers instead of opening a public issue.

## Code Style and Standards

- Python target: 3.10+ (see `pyproject.toml`)
- Formatting: `black`
- Linting: `ruff`
- Testing: `pytest`

## Questions

If you are unsure whether to start with an issue or a PR, start with an issue for alignment.

Thanks for helping improve this project.
