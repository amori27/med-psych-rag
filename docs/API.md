# API Reference

## `GET /health`

Health check.

```json
{"status": "ok", "llm": "ollama/llama3", "docs_indexed": 142}
```

## `POST /query`

Ask a question.

**Request:**
```json
{"question": "What are GAD symptoms?", "top_k": 5}
```

**Response:**
```json
{
  "answer": "GAD symptoms include restlessness, fatigue... [PMID: 12345]",
  "sources": [
    {"file": "gad_guidelines.pdf", "page": 3, "score": 0.91}
  ],
  "metrics": {"context_precision": 0.87, "faithfulness": 0.94}
}
```

## `POST /ingest`

Upload and index a PDF.

**Request:** multipart/form-data with `file` field.

**Response:**
```json
{"chunks": 47, "document": "gad_guidelines.pdf"}
```
