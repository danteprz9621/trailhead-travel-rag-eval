"""
Agent under test: agents/rag_agent.py

Measures how much a metric's score wobbles between identical runs on an
unchanged system -- the judge LLM is non-deterministic, so re-scoring the
same question/answer/context twice can give two different numbers even
though nothing about the RAG pipeline changed. That wobble is the
eval-noise floor: how much a score can move before it's a real regression
rather than judge noise (e.g. "faithfulness dropped from 0.86 to 0.83, is
that real or noise?"). Run manually -- this isn't a pytest file and isn't
part of the automated suite:

    python -m scripts.measure_noise

Judge construction matches test_dataset_eval.py: AsyncOpenAI + llm_factory,
not LangchainLLMWrapper/ChatOllama. Faithfulness/ContextRecall are the
ragas.metrics.collections classes scored directly via ascore() -- see
test_dataset_eval.py's docstring for why ResponseRelevancy specifically
doesn't work with embedding_factory()'s embeddings.
"""

import statistics
import asyncio
import sys

# Windows defaults to WindowsProactorEventLoopPolicy, and AsyncOpenAI
# clients built on it don't survive being reused across separate
# asyncio.run() calls -- this script makes 20 of them (2 metrics x 10
# runs) against the same client, which reliably triggers
# "RuntimeError: Event loop is closed" without this. Unlike the tests/
# files, this script isn't run through pytest, so tests/conftest.py's copy
# of this same fix doesn't apply here -- it needs its own.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.metrics.collections import Faithfulness, ContextRecall
from ragas.llms import llm_factory

from agents.rag_agent import ask

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
faithfulness = Faithfulness(judge_llm)
context_recall = ContextRecall(judge_llm)

# One fixed question/answer/context, held still across all 10 runs below --
# this measures judge variance, not agent variance, so ask() only runs once.
user_question = "What's the refund policy?"
context_reference = "There's a refund policy and flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method"
result = ask(user_question)
sample = SingleTurnSample(
    user_input=user_question,
    response=result["answer"],
    retrieved_contexts=result["retrieval_context"],
    reference=context_reference
)

# Score the same fixed sample N times per metric; the spread across runs
# is the noise floor.
runs_faithfulness = []
runs_context_recall = []
for _ in range(10):
    res_f = asyncio.run(faithfulness.ascore(
        user_input=user_question,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts))
    res_c = asyncio.run(context_recall.ascore(
        user_input=user_question,
        retrieved_contexts=sample.retrieved_contexts,
        reference=sample.reference))
    runs_faithfulness.append(res_f.value)
    runs_context_recall.append(res_c.value)

mean = statistics.mean(runs_faithfulness)
noise = statistics.pstdev(runs_faithfulness)
suggested_floor = mean - 2 * noise
print(f"Faithfulness baseline: {mean:.3f} ± {noise:.3f}")
print(f"Faithfulness suggested floor: {suggested_floor:.3f}")

mean = statistics.mean(runs_context_recall)
noise = statistics.pstdev(runs_context_recall)
suggested_floor = mean - 2 * noise
print(f"Context Recall baseline: {mean:.3f} ± {noise:.3f}")
print(f"Context Recall suggested floor: {suggested_floor:.3f}")
