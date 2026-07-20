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

from openai import AsyncOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import llm_factory

from agents.rag_agent import ask


# 1. Build an AsyncOpenAI client pointed at Ollama's OpenAI-compatible
#    endpoint, then judge_llm via llm_factory("llama3.1:8b",
#    provider="openai", client=client) -- same pattern as
#    test_rag_agent.py/test_dataset_eval.py step 1. Then a Faithfulness
#    instance (no embeddings needed).


# 2. Pick ONE fixed question the knowledge base can answer, call ask() on
#    it ONCE, and build a single SingleTurnSample from the result (see
#    test_rag_agent.py's earlier steps for the user_input/response/
#    retrieved_contexts shape). The point is to hold the system's output
#    completely still -- you're measuring judge variance here, not agent
#    variance, so don't call ask() again inside the loop in step 3.


# 3. Write a loop that runs evaluate() on a fresh EvaluationDataset built
#    from that one sample, N times (10 is a reasonable start). Each
#    iteration:
#    - Pull the score list back out with result["faithfulness"]
#    - Reduce it to a single number with statistics.mean(...) (there's only
#      one sample per run, so this just unwraps a 1-element list -- write
#      it generically anyway, it generalizes to bigger datasets later)
#    - Append that number to a running list of per-run scores


# 4. Once the loop finishes, compute and print:
#    - mean = statistics.mean(per_run_scores)
#    - noise = statistics.pstdev(per_run_scores)
#    - suggested floor = mean - 2 * noise
#    That (mean, noise) pair is your answer to "how much can this score
#    move before I should treat it as a real change instead of judge
#    noise?" -- keep it around for whenever you're ready to act on it
#    (e.g. wiring an actual CI gate later).


# 5. (Optional, once step 4 works) Repeat steps 2-4 for a second metric --
#    e.g. LLMContextRecall, which needs a `reference` on the sample too --
#    so you can compare how noisy different metrics are relative to each
#    other.
