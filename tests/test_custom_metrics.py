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
import asyncio
from openai import AsyncOpenAI
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import DiscreteMetric  # AspectCritic was removed in v0.4;
                                          # DiscreteMetric is its replacement
from ragas.llms import llm_factory

from agents.rag_agent import ask


# 1. Build an AsyncOpenAI client pointed at Ollama's OpenAI-compatible
#    endpoint and judge_llm via llm_factory("llama3.1:8b",
#    provider="openai", client=client) -- same pattern as
#    test_rag_agent.py step 1. No embeddings needed here. The judge_llm IS
#    used: DiscreteMetric is LLM-backed (the model produces the verdict),
#    so it needs judge_llm even though embeddings aren't involved.
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)

# 2. Create a DiscreteMetric named something like
#    "invented_figures_detected". Its verdict comes from the judge LLM, so
#    the criterion lives in the `prompt` (with a {response} and
#    {retrieved_contexts} placeholder) and the outcomes live in
#    `allowed_values` -- e.g. ["clean", "invented"]. Write the prompt as an
#    unambiguous pass/fail rule, stating both conditions explicitly --
#    don't make the judge infer the negative (course design tip for
#    aspect-style metrics, and it matches what this project found in
#    practice: small local judge models are noticeably less reliable on
#    vague or compound-clause definitions than on a single direct one).
#    Something like: "Does the response state a specific fee, percentage,
#    or day/hour figure that is NOT present anywhere in retrieved_contexts?
#    Answer 'invented' if it does, 'clean' if every figure is supported."
#    Pass llm=judge_llm (at construction or at ascore() time -- both work).
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
    llm=judge_llm,
)

user_question = "What's the baggage overnight fee?"

# 3. Write test_no_invented_figures():
#    - Pick a question where the real answer involves a specific number
#      (e.g. the baggage overweight fee, or the refund percentage tiers --
#      check data/knowledge_base/ for the real figures)
#    - Call ask(question) to run the RAG agent
#    - Score directly with the metric from step 2: await metric.ascore(
#      response=result["answer"],
#      retrieved_contexts=result["retrieved_contexts"]) -- no evaluate() and
#      no EvaluationDataset needed; ascore() is the current path (both are
#      deprecated / evaluate() is). Because ascore() is async, wrap the call
#      in asyncio.run() (or make the test async under pytest-asyncio).
#    - Assert on result.value. DiscreteMetric returns one of your
#      allowed_values as a STRING (not 0/1), so this is an equality check
#      against your label, e.g. assert result.value == "clean". result.reason
#      holds the judge's explanation if you want it in the failure message.
def test_no_invented_figures():
    response = ask(user_question)
    result = asyncio.run(invented_figures.ascore(
        response=response["answer"],
        retrieved_contexts = response["retrieved_contexts"]
    ))

    print(result.reason)
    assert result.value == "clean"

# 4. For comparison, write a second test that deliberately checks a
#    fabricated bad response instead of a real agent call -- build a
#    SingleTurnSample (or just pass the fields to ascore()) by hand with a
#    response inventing a "15% early cancellation bonus" that
#    data/knowledge_base/ never mentions, empty or unrelated
#    retrieved_contexts, and assert the metric DOES flag it, e.g.
#    assert result.value == "invented". This is the same discipline as a
#    negative test case for a validator: if you only ever test the metric
#    against good behavior, you can't tell a lenient/broken metric from a
#    strict one -- you need at least one case you know should fail. (If the
#    small local judge is flaky on this, that's a signal to tighten the
#    step-2 prompt, not to drop the test.)

def test_invented_figures():
    response = ask(user_question)
    result = asyncio.run(invented_figures.ascore(
        response="There's a $1000 baggage overnight fee",
        retrieved_contexts = response["retrieved_contexts"]
    ))

    print(result.reason)
    assert result.value == "invented"
