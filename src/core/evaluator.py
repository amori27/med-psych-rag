"""Evaluates RAG quality using TruLens."""

from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback.provider.openai import OpenAI


def create_feedback():
    provider = OpenAI()

    context_precision = Feedback(
        provider.context_relevance_with_cot_reasons
    ).on_input_output()

    faithfulness = Feedback(
        provider.groundedness_measure_with_cot_reasons
    ).on_output()

    return [context_precision, faithfulness]


def evaluate_rag(rag_chain, queries: list[str]):
    from trulens_eval import Tru

    tru = Tru()
    feedbacks = create_feedback()
    tru_recorder = TruLlama(rag_chain, app_id="med-psych-rag", feedbacks=feedbacks)

    for q in queries:
        with tru_recorder:
            rag_chain.query(q)

    return tru.get_leaderboard(app_ids=["med-psych-rag"])
