from src.config import Config
from src.generation.prompt_templates import RAG_PROMPT


def get_llm():
    match Config.llm_provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=Config.openai_api_key,
                temperature=0,
            )
        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-3-haiku-20240307",
                api_key=Config.anthropic_api_key,
                temperature=0,
            )
        case "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=Config.ollama_model,
                base_url=Config.ollama_base_url,
                temperature=0,
            )
        case _:
            raise ValueError(f"Unsupported provider: {Config.llm_provider}")


def generate_answer(question: str, context: str) -> str:
    if Config.llm_provider == "demo" or Config.demo_mode:
        sources = []
        for line in context.split("\n"):
            if line.startswith("[Source:"):
                sources.append(line.strip())
        source_str = "\n".join(sources[:3]) if sources else "local index"
        return (
            f"[Demo Mode] Retrieved answer for: {question}\n\n"
            f"Based on the following sources:\n{source_str}\n\n"
            f"(Set LLM_PROVIDER to openai/anthropic/ollama and configure API keys for LLM-powered answers)"
        )

    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
