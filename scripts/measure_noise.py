"""
Ragas skeleton: Eval-noise floor (course Module 6)
Agent under test: agents/rag_agent.py

Run this manually (it's not a pytest file, and not part of the automated
suite) to measure how much a metric's score wobbles between identical runs
on an unchanged system. That wobble is your "eval-noise floor" -- the judge
LLM is itself non-deterministic, so re-scoring the exact same
question/answer/context twice can give two different numbers even though
nothing about your RAG pipeline changed.

This is useful on its own, independent of CI: it tells you how much to
trust a single score before you act on it (e.g. "faithfulness dropped from
0.86 to 0.83, is that a real regression or just noise?"). It also happens
to be the input a CI gate would need if you build one later -- a threshold
tighter than this noise would flap red/green on identical input, which is
indistinguishable from a flaky test to whoever's watching the build -- but
building that gate isn't part of this project yet.

Follow the numbered comments top to bottom, then run:

    python scripts/measure_noise.py

Real API note (verified against ragas 0.4.3, the version this project
installs): EvaluationResult's __getitem__ (result["faithfulness"]) returns
a LIST of per-sample scores, not a single aggregate number -- even though
printing the result object itself (print(result)) shows a nicely-formatted
mean. Every run's score list needs to be reduced to a single number
(statistics.mean(...)) before you can compare runs to each other.

Judge construction matches test_dataset_eval.py: an AsyncOpenAI client
pointed at Ollama's OpenAI-compatible endpoint + llm_factory(), not
LangchainLLMWrapper/ChatOllama. Faithfulness is the legacy ragas.metrics
class (not ragas.metrics.collections) because evaluate()/EvaluationDataset
only accepts that family -- see test_dataset_eval.py's docstring for the
full story on why, and why ResponseRelevancy specifically isn't used
anywhere with the newer embedding_factory().
"""

import statistics
import asyncio
from openai import AsyncOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, ContextRecall
from ragas.llms import llm_factory

from agents.rag_agent import ask


# 1. Build an AsyncOpenAI client pointed at Ollama's OpenAI-compatible
#    endpoint, then judge_llm via llm_factory("llama3.1:8b",
#    provider="openai", client=client) -- same pattern as
#    test_rag_agent.py/test_dataset_eval.py step 1. Then a Faithfulness
#    instance (no embeddings needed).
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
faithfulness = Faithfulness(judge_llm)
context_recall = ContextRecall(judge_llm)

# 2. Pick ONE fixed question the knowledge base can answer, call ask() on
#    it ONCE, and build a single SingleTurnSample from the result (see
#    test_rag_agent.py's earlier steps for the user_input/response/
#    retrieved_contexts shape). The point is to hold the system's output
#    completely still -- you're measuring judge variance here, not agent
#    variance, so don't call ask() again inside the loop in step 3.
user_question = "What's the refund policy?"
context_reference="There's a refund policy and flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method"
result = ask(user_question)
sample = SingleTurnSample(
    user_input=user_question,
    response=result["answer"],
    retrieved_contexts=result["retrieved_contexts"],
    reference=context_reference
)


# 3. Write a loop that scores that ONE fixed sample N times (10 is a
#    reasonable start) by calling faithfulness.ascore() directly. In v0.4
#    there's no evaluate() batch call and no EvaluationDataset result dict
#    (both deprecated) -- you score per-sample and read the number off the
#    returned MetricResult. Each iteration:
#    - res = asyncio.run(faithfulness.ascore(
#          response=sample.response,
#          retrieved_contexts=sample.retrieved_contexts))
#      (ascore() is async: wrap it in asyncio.run in a sync test, or make
#      the test async under pytest-asyncio and await it)
#    - Read the score off res.value (a float; res.reason holds the judge's
#      explanation if you want it). There's no per-run list to unwrap now --
#      one sample gives one MetricResult. If you later score several samples
#      per run, you'd asyncio.gather the ascore() calls and
#      statistics.mean their .value's; write that reduction generically now
#      so it carries over to bigger datasets.
#    - Append that number to a running list of per-run scores
runs_faithfulness = []
runs_context_recall = []
for _ in range(10):
    res_f = asyncio.run(faithfulness.ascore(
        user_input=user_question,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts))
    res_c = asyncio.run(context_recall.ascore(
        user_input=user_question,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
        reference=sample.reference))
    runs_faithfulness.append(res_f.value)
    runs_context_recall.append(res_c.value)

# 4. Once the loop finishes, compute and print:
#    - mean = statistics.mean(per_run_scores)
#    - noise = statistics.pstdev(per_run_scores)
#    - suggested floor = mean - 2 * noise
#    That (mean, noise) pair is your answer to "how much can this score
#    move before I should treat it as a real change instead of judge
#    noise?" -- keep it around for whenever you're ready to act on it
#    (e.g. wiring an actual CI gate later).
mean = statistics.mean(runs_faithfulness)
noise = statistics.pstdev(runs_faithfulness)
suggested_floor = mean - 2*noise
print("Faithfulness baseline: {mean:.3f} ± {noise:.3f}")
print("Faithfulness suggested floor: {suggested_floor:.3f}")

mean = statistics.mean(runs_context_recall)
noise = statistics.pstdev(runs_context_recall)
suggested_floor = mean - 2*noise
print("Context Recall baseline: {mean:.3f} ± {noise:.3f}")
print("Context Recall suggested floor: {suggested_floor:.3f}")

# 5. (Optional, once step 4 works) Repeat steps 2-4 for a second metric --
#    e.g. ContextRecall from ragas.metrics.collections (the v0.4 name; the
#    old LLMContextRecall is legacy). It needs a `reference` on the sample,
#    so build the sample with reference= and pass reference= (alongside the
#    context fields) to its ascore(). Comparing its noise to Faithfulness's
#    tells you how much the metric choice itself affects stability.
