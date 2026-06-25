import argparse
from pathlib import Path
from src.config import Config
from src.ingestion.pdf_processor import PDFProcessor
from langchain_core.documents import Document
from src.ingestion.text_splitter import split_documents


def main():
    parser = argparse.ArgumentParser(description="Med-Psych RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a PDF into the vector store")
    ingest.add_argument("file", type=str, help="Path to PDF file")

    query_cmd = sub.add_parser("query", help="Ask a question")
    query_cmd.add_argument("question", type=str, help="Your question")
    query_cmd.add_argument("--top-k", type=int, default=None)
    query_cmd.add_argument("--demo", action="store_true", help="Use demo/local mode")

    args = parser.parse_args()

    if args.command == "ingest":
        _ingest(args.file)
    elif args.command == "query":
        _query(args.question, args.top_k)


def _ingest(file_path: str):
    is_demo = Config.demo_mode or Config.llm_provider == "demo"

    if is_demo:
        from src.retrieval.local_retriever import LocalRetriever
        retriever = LocalRetriever()
        retriever.load()

    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    processor = PDFProcessor(path)
    pages = processor.extract_text_by_page()
    docs = [
        Document(page_content=p["text"], metadata={"source": str(path), "page": p["page"]})
        for p in pages
    ]
    chunks = split_documents(docs)

    if is_demo:
        retriever.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunks into local index from {path.name}")
    else:
        from src.ingestion.pipeline import process_pdf as remote_process
        count = remote_process(path)
        print(f"Ingested {count} chunks from {path.name}")


def _query(question: str, top_k: int | None = None):
    is_demo = Config.demo_mode or Config.llm_provider == "demo"

    if is_demo:
        from src.retrieval.local_retriever import LocalRetriever
        retriever = LocalRetriever()
        if not retriever.load():
            print("No local index found. Run 'ingest' with DEMO=true first.")
            return
        docs = retriever.retrieve(question, top_k)
    else:
        from src.retrieval.retriever import retrieve as _retrieve
        docs = _retrieve(question)

    if not docs:
        print("No relevant documents found.")
        return

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, "
        f"Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

    from src.generation.generator import generate_answer
    answer = generate_answer(question, context)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources: {len(docs)} chunk(s) retrieved")
    for d in docs:
        print(f"  - {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', 'N/A')}")


if __name__ == "__main__":
    main()
