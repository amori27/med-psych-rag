# Contributing to Med-Psych RAG

Thank you for your interest in contributing! We welcome contributions from the community.

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/amori27/med-psych-rag.git
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Copy the environment file and configure:
   ```bash
   cp .env.example .env
   ```

## Code Style

- This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Run linting before committing:
  ```bash
  ruff check src/ tests/
  ```

## Testing

- Run tests with:
  ```bash
  pytest --cov=src --cov-report=term
  ```
- All new features should include tests.
- Maintain or improve the current coverage level.

## Pull Request Process

1. Ensure your branch is up to date with `main`.
2. Run the full test suite and lint checks locally.
3. Update documentation if your changes introduce new behavior or modify existing APIs.
4. Open a pull request against the `main` branch.
5. Ensure CI passes (lint, test, and docker jobs).

## Commit Messages

Use clear, concise commit messages. Follow conventional commits style:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `refactor:` code restructuring
- `test:` test additions or changes
- `chore:` maintenance tasks

## Code of Conduct

All contributors must adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).
