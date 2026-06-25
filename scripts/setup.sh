#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Med-Psych RAG ==="

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling Llama-3..."
ollama pull llama3

echo ""
echo "Setup complete! Run:"
echo "  source .venv/bin/activate"
echo "  uvicorn src.main:app --reload"
