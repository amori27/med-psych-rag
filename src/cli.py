import argparse
from pathlib import Path
from src.ingestion.pipeline import process_pdf
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Med-Psych RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a PDF into the vector store")
    ingest.add_argument("file", type=str, help="Path to PDF file")

    query = sub.add_parser("query", help="Ask a question")
    query.add_argument("question", type=str, help="Your question")

    args = parser.parse_args()

    if args.command == "ingest":
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            return
        count = process_pdf(path)
        print(f"Ingested {count} chunks from {path.name}")

    elif args.command == "query":
        docs = retrieve(args.question)
        if not docs:
            print("No relevant documents found.")
            return
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, "
            f"Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        )
        answer = generate_answer(args.question, context)
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {len(docs)} chunk(s) retrieved")


if __name__ == "__main__":
    main()
