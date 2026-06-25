# Med-Psych RAG — Architecture

## Overview

A Retrieval-Augmented Generation system for medical and psychological documents.
Uses a local open-source LLM (Llama-3 via Ollama) to ensure data privacy.

## Stack

| Layer | Technology |
|---|---|
| LLM | Llama-3 8B (via Ollama) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | ChromaDB |
| PDF Parsing | PyMuPDF |
| Framework | LangChain |
| API | FastAPI |
| Container | Docker + docker-compose |

## Data Flow

```
PDF Docs ──> PyMuPDF ──> Text Chunks ──> Embeddings ──> ChromaDB
                                                          │
User Query ───────────────────────────────────────────────┤
                                                          │
                                                    Semantic Search
                                                          │
                                                   Context + Query
                                                          │
                                                     Ollama LLM
                                                          │
                                                   Cited Answer
```

## Directory Layout

```
src/
├── core/          # RAG pipeline orchestration
├── models/        # Pydantic schemas
└── utils/         # Helpers (PDF, chunking)
```

## Evaluation

Uses TruLens to measure Context Precision, Context Recall, and Faithfulness.
