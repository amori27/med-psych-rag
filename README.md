# Med-Psych RAG System

Citation-based Retrieval-Augmented Generation system for medical and psychological documents.

## Features

- **PDF ingestion** — Extract and chunk PDF documents with metadata
- **Semantic retrieval** — Find relevant passages using sentence embeddings
- **Citation-grounded generation** — LLM answers with source page references, reducing hallucination
- **Multiple LLM backends** — OpenAI, Anthropic, or local Ollama models
- **REST API** — FastAPI endpoint for querying
- **Configurable** — Chunk size, overlap, top-k retrieval, similarity thresholds

## Project Structure

```
med-psych-rag/
├── src/
│   ├── ingestion/          # PDF processing and text splitting
│   │   ├── pdf_processor.py
│   │   ├── text_splitter.py
│   │   └── pipeline.py
│   ├── embeddings/         # Embedding model
│   │   └── embedder.py
│   ├── retrieval/          # Vector store and retrieval
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── generation/         # LLM prompt and answer generation
│   │   ├── prompt_templates.py
│   │   └── generator.py
│   ├── config.py
│   └── app.py              # FastAPI server
├── data/                   # Place PDFs here
├── notebooks/demo.ipynb
├── tests/
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Ingest PDFs
python -m src.ingestion.pipeline data/your-document.pdf

# Run API server
uvicorn src.app:app --reload

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of anxiety?"}'
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` or `faiss` |

## API

- `GET /health` — Health check
- `POST /query` — Ask a question with citation-grounded answer

```json
{
  "question": "How is PTSD diagnosed?",
  "top_k": 5
}
```

Response includes `answer`, `sources` (with page numbers), and `source_count`.
