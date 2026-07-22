"""
Agent under test: agents/rag_agent.py
"""
import asyncio
from openai import AsyncOpenAI
from ragas.metrics import DiscreteMetric  # AspectCritic's v0.4 replacement
from ragas.llms import llm_factory

from agents.rag_agent import ask

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)

invented_figures = DiscreteMetric(
    name="invented_figures_detected",
    allowed_values=["clean", "invented"],
    prompt=(
        "You are checking whether a response invented any numbers."
        "Look at every specific figure in the RESPONSE -- fees, prices, "
        "percentages, weights, hour/day counts, dates."
        "For each one, check whether that exact figure appears in the "
        "RETRIEVED CONTEXTS below."
        "Answer 'invented' if the response states ANY figure that is not "
        "present in the retrieved contexts."
        "Answer 'clean' if every figure in the response is supported by the "
        "retrieved contexts (or the response states no figures at all)."
        "RESPONSE: {response}"
        "RETRIEVED CONTEXTS: {retrieved_contexts}"
        "Answer with only one word: 'clean' or 'invented'."
    ),
)

user_question = "What's the baggage overnight fee?"

def test_no_invented_figures():
    # Real agent answer, grounded in retrieved context -- should be "clean".
    response = ask(user_question)
    result = asyncio.run(invented_figures.ascore(
        llm=judge_llm,
        response=response["answer"],
        retrieved_contexts = response["retrieval_context"]
    ))

    print(result.reason)
    assert result.value == "clean"

def test_invented_figures():
    # Fabricated response with a figure the knowledge base never mentions
    # -- should be flagged "invented". Without a case like this, a
    # lenient/broken metric would look identical to a strict one.
    response = ask(user_question)
    result = asyncio.run(invented_figures.ascore(
        llm=judge_llm,
        response="There's a $1000 baggage overnight fee",
        retrieved_contexts = response["retrieval_context"]
    ))

    print(result.reason)
    assert result.value == "invented"
