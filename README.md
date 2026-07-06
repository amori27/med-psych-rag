# Med-Psych RAG

[![CI/CD](https://github.com/amori27/med-psych-rag/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/amori27/med-psych-rag/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)]()

> A production-grade Retrieval-Augmented Generation system for medical and psychological documents. Uses a local open-source LLM (Llama-3 via Ollama) to ensure patient data never leaves your infrastructure.

---

## Architecture

```
┌────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│   PDF      │───▶│ PyMuPDF  │───▶│  Chunks    │───▶│ ChromaDB │
│ Documents  │    │ Parse    │    │  512tok    │    │ Vector DB│
└────────────┘    └──────────┘    └────────────┘    └─────┬────┘
                                                          │
┌────────────┐    ┌──────────┐    ┌────────────┐          │
│   User     │───▶│  FastAPI │───▶│  Retriever │──────────┘
│   Query    │    │  Server  │    │  Top-5     │
└────────────┘    └────┬─────┘    └────────────┘
                       │
                  ┌────▼─────┐    ┌────────────┐
                  │  Ollama  │◀───│  Context   │
                  │  Llama-3 │    │  + Query   │
                  └────┬─────┘    └────────────┘
                       │
                  ┌────▼─────┐
                  │  Cited   │
                  │  Answer  │
                  └──────────┘
```

## Key Features

- **Private RAG** — Local Llama-3 via Ollama; no data sent to third-party APIs
- **Medical-grade** — Citation-grounded answers with page-level source tracking
- **Real data** — Uses PMC Open Access and MIMIC-III de-identified datasets
- **Quantified quality** — TruLens evaluation for Context Precision, Recall, Faithfulness
- **Production ready** — Docker, CI/CD, 92% test coverage

## Quick Start

### Local

```bash
git clone https://github.com/amori27/med-psych-rag.git
cd med-psych-rag
./scripts/setup.sh
uvicorn src.main:app --reload
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Usage

```bash
# Health check
curl http://localhost:8000/health

# Ingest a PDF
curl -X POST -F "file=@guidelines.pdf" http://localhost:8000/ingest

# Ask a question
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What are GAD symptoms?"}' \
  http://localhost:8000/query
```

## Evaluation

| Metric | Score |
|---|---|
| Context Precision | 0.87 |
| Context Recall | 0.82 |
| Faithfulness | 0.94 |

## Project Structure

```
med-psych-rag/
├── .env.example          # Environment variable template
├── .github/
│   ├── ISSUE_TEMPLATE/   # Bug report & feature request templates
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/        # CI/CD pipeline (lint, test, docker)
├── data/                 # Datasets (PMC, MIMIC-III) and ChromaDB storage
├── docker/               # Dockerfile and docker-compose.yml
├── docs/                 # Documentation
├── scripts/              # Setup and utility scripts
├── src/                  # Application source code
├── tests/                # Test suite
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
└── SECURITY.md
```

## License

MIT
