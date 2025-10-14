# Contributing

Thank you for considering contributing!

- Use Python 3.12+
- Create a virtualenv:
  - `python -m venv .venv && . .venv/bin/activate`
- Install dev tools:
  - `pip install -U pip setuptools wheel cython build pytest ruff black isort tox`
- Lint and test:
  - `tox`
- Build package:
  - `python -m build`

Before submitting PRs, please run linters and tests locally.
