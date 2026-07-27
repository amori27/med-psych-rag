"""Evaluates RAG quality using TruLens.

Provides feedback functions for context relevance and groundedness,
and a convenience wrapper to run evaluation over a set of queries.
"""

from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback.provider.openai import OpenAI


def create_feedback() -> list[Feedback]:
    """Create TruLens feedback functions for RAG evaluation.

    Returns:
        List containing context-precision and faithfulness feedback objects.
    """
    provider = OpenAI()

    context_precision = Feedback(
        provider.context_relevance_with_cot_reasons
    ).on_input_output()

    faithfulness = Feedback(
        provider.groundedness_measure_with_cot_reasons
    ).on_output()

    return [context_precision, faithfulness]


def evaluate_rag(rag_chain, queries: list[str]):
    """Run a set of evaluation queries through the RAG chain.

    Args:
        rag_chain: A TruLlama-wrapped RAG chain.
        queries: List of evaluation questions.

    Returns:
        TruLens leaderboard DataFrame.
    """
    from trulens_eval import Tru

    tru = Tru()
    feedbacks = create_feedback()
    tru_recorder = TruLlama(rag_chain, app_id="med-psych-rag", feedbacks=feedbacks)

    for q in queries:
        with tru_recorder:
            rag_chain.query(q)

    return tru.get_leaderboard(app_ids=["med-psych-rag"])
