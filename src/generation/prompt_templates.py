from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise medical and psychological assistant. Answer based solely on "
        "the provided context. If the context lacks sufficient information, state that "
        "clearly. Cite the source page numbers when possible.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

CONCISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer concisely using only the context below. Include page citations.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])
