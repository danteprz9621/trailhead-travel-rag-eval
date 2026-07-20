"""
Ragas skeleton: Custom metric with AspectCritic (course Module 8)
Agent under test: agents/rag_agent.py

The four core metrics (test_rag_agent.py) are generic. Real systems have
domain rules the standard metrics don't capture -- for Trailhead Travel,
something like "never state a specific fee, percentage, or day/hour count
that isn't actually backed by the retrieved policy text." AspectCritic lets
you define a binary (pass/fail) LLM-judged metric from a plain-English
definition, closer to writing a custom assertion than using a pre-built
metric.

Follow the numbered comments top to bottom.
Docs: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/general_purpose/

Note: this uses the legacy ragas.metrics + evaluate()/EvaluationDataset API
(same as test_dataset_eval.py), not the newer ragas.metrics.collections API
that test_rag_agent.py uses -- as of this Ragas version, AspectCritic
doesn't have a collections equivalent to migrate to (a real gap, not a
choice -- see the README). Judge construction still follows
test_rag_agent.py's pattern (AsyncOpenAI + llm_factory, not
LangchainLLMWrapper) -- confirmed working: legacy AspectCritic accepts the
llm_factory-built judge fine via duck-typing, it's ResponseRelevancy
specifically (needs embeddings, see test_dataset_eval.py's docstring) that
doesn't.

Real finding worth knowing before you tune step 2's wording: two different
phrasings of an "invented figures" definition were tried against
llama3.1:8b while building this project. One over-triggered (flagged a
correct, context-backed $75 answer as inventing a figure). The other
under-triggered (missed a fabricated "15% early cancellation bonus"
entirely). Neither direction is a bug in your test -- it's the judge model
itself being unreliable on this class of check, which is exactly the eval
noise Module 6 is about. Don't assume you have a broken definition just
because a rerun disagrees with a previous run; that's the actual
motivation for measuring a noise floor instead of trusting a single score.
"""

from openai import AsyncOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import AspectCritic
from ragas.llms import llm_factory

from agents.rag_agent import ask


# 1. Build an AsyncOpenAI client pointed at Ollama's OpenAI-compatible
#    endpoint and judge_llm via llm_factory("llama3.1:8b",
#    provider="openai", client=client) -- same pattern as
#    test_rag_agent.py step 1. No embeddings needed here; AspectCritic
#    doesn't use them.


# 2. Create an AspectCritic instance named something like
#    "invented_figures_detected". Write the definition as an unambiguous
#    pass/fail rule, stating both the pass and the fail condition
#    explicitly -- don't make the judge infer the negative (this is the
#    course's own design tip for AspectCritic, and matches what this
#    project's other AspectCritic-adjacent metrics found in practice: small
#    local judge models are noticeably less reliable on vague or
#    compound-clause definitions than on a single direct one). Something
#    like: "Does the response state a specific fee, percentage, or day/hour
#    figure that is not present anywhere in the retrieved_contexts?"


# 3. Write test_no_invented_figures():
#    - Pick a question where the real answer involves a specific number
#      (e.g. the baggage overweight fee, or the refund percentage tiers --
#      check data/knowledge_base/ for the real figures)
#    - Call ask(question) to run the RAG agent
#    - Build a SingleTurnSample with user_input, response, and
#      retrieved_contexts from the result
#    - evaluate() with the metric from step 2, wrapped in an
#      EvaluationDataset
#    - Assert no issue was found (score == 0 for a "detected"-style
#      definition -- unlike a 0-1 continuous metric, AspectCritic's
#      per-sample score is exactly 0 or 1, so this is an equality check,
#      not a threshold)


# 4. For comparison, write a second test that deliberately checks a
#    fabricated bad response instead of a real agent call -- something like
#    building a SingleTurnSample by hand with a response inventing a "15%
#    early cancellation bonus" that data/knowledge_base/ never mentions,
#    empty or unrelated retrieved_contexts, and asserting the metric DOES
#    flag it (score == 1). This is the same discipline as writing a
#    negative test case for a validator: if you only ever test the metric
#    against good behavior, you can't tell a lenient/broken metric from a
#    strict one -- you need at least one case you know should fail.
