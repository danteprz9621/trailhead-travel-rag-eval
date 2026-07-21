"""
Ragas test skeleton: Golden Dataset & Synthetic Data
Agent under test: agents/rag_agent.py

Follow the numbered comments top to bottom. Each comment describes ONE
thing to implement below it -- fill in the code yourself.
Docs: https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/

TestsetGenerator is Ragas's analog to DeepEval's Synthesizer: it builds a
knowledge graph from local documents and generates synthetic goldens
(question + reference answer pairs) from it. It's the slowest test in this
suite -- expect well over a minute on local 7B/8B models.

Everything in this file runs through agents/rag_agent.py's ask() -- both the
hand-written and the synthesized goldens are questions about
data/knowledge_base/ content, so Faithfulness against the RAG agent's own
retrieved_contexts means something for both.

Judge construction here matches test_rag_agent.py: an AsyncOpenAI client
pointed at Ollama's OpenAI-compatible endpoint, llm_factory(), and
embedding_factory() -- not LangchainLLMWrapper/LangchainEmbeddingsWrapper.
This file uses evaluate()/EvaluationDataset (not the collections API's
per-sample ascore()) because that's what TestsetGenerator and this file's
metrics need -- see the note below on why only Faithfulness is used, not
ResponseRelevancy.

Real finding worth knowing before you start: legacy ragas.metrics.
ResponseRelevancy does NOT work with embedding_factory()'s embeddings
object -- it silently returns NaN instead of raising, because it expects an
object with an .embed_query() method (what LangchainEmbeddingsWrapper
produces) and embedding_factory()'s modern-interface object doesn't have
one. evaluate() also flatly refuses to mix ragas.metrics.collections
metrics into the same call (raises TypeError: "All metrics must be
initialised metric objects"), so collections.AnswerRelevancy isn't a drop-in
substitute here either. Net result: this file only uses Faithfulness, which
doesn't need embeddings and works fine with llm_factory's judge --
AnswerRelevancy/ResponseRelevancy coverage lives in test_rag_agent.py
instead.
"""
import asyncio
from glob import glob

from openai import AsyncOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import TextLoader

from agents.rag_agent import ask


# 1. Build an AsyncOpenAI client pointed at Ollama's OpenAI-compatible
#    endpoint (same as test_rag_agent.py step 1), then:
#    - judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
#      (local models still need provider="openai" -- the model name won't
#      auto-resolve to a provider on its own)
#    - judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)
#      (needed for TestsetGenerator in step 4, not for the Faithfulness
#      metric itself)
#    - a Faithfulness instance from ragas.metrics.collections (NOT the legacy
#      ragas.metrics one). evaluate() is deprecated in v0.4; the collections
#      metrics are the current path -- each exposes an async ascore(**kwargs)
#      that returns a MetricResult (.value float, optional .reason)
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)
faithfulness = Faithfulness(judge_llm)

# 2. Write test_evaluation_dataset():
#    - Hand-write a small list of (question, reference) goldens using real
#      facts from data/knowledge_base/ (e.g. baggage/refund/loyalty policy
#      numbers). retrieved_contexts is a list, not "" -- empty here because
#      ask() fills it in below
#    - Run each question through ask() (the RAG agent) to get a response and
#      its retrieved_contexts, and build a SingleTurnSample per golden with
#      user_input, response, retrieved_contexts, and reference
#    - Score each sample with faithfulness.ascore(response=...,
#      retrieved_contexts=...) -- the v0.4 replacement for the single
#      evaluate() call. ascore() is async and returns a MetricResult, so
#      gather the calls (asyncio.gather) and read .value off each. The
#      "batch" is now one gather over N ascore() calls, not one evaluate()
#      (note: Faithfulness scores response-vs-context and ignores reference;
#      it's on the sample for the dataset, and for reference-based metrics
#      like AnswerCorrectness/FactualCorrectness if you add them later)
#    - Aggregate (e.g. mean of the .value scores) and assert on it

def mean_score(scores) -> float:
    return sum(scores)/len(scores)

goldens = [
    SingleTurnSample(
        user_input="What's the refund policy?",
        retrieved_contexts="",
        response="",
        reference="There's a refund policy and flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method"
    ),
    SingleTurnSample(
        user_input="What's the baggage policy?",
        retrieved_contexts="",
        response="",
        reference="Economy passengers may check one bag up to 23kg free of charge, additional bags can be added by paying an additional cost"
    )
]

def test_evaluation_dataset():
    scores = list()
    for golden in goldens():
        result = ask(golden.user_input)
        res = asyncio.run(faithfulness.ascore(
            response=result["answer"],
            retrieved_contexts=result["retrieved_contexts"],
        ))
        scores.append(res.value)
    
    assert mean_score(scores) >= 0.7
    

# 3. Write load_knowledge_base_docs():
#    - Use glob() to find every .txt file under data/knowledge_base/
#    - Load each with langchain_community's TextLoader (not DirectoryLoader
#      -- its default loader needs the heavyweight `unstructured` package,
#      which this project doesn't install)
#    - Return the combined list of loaded documents

def load_knowledge_base_docs():
    docs = []
    for path in glob("data/knowledge_base/**/*.txt", recursive=True):
        docs.extend(TextLoader(path).load())
    return docs


# 4. Build a TestsetGenerator(llm=judge_llm, embedding_model=judge_embeddings)
#    using the judge models from step 1. Needs the `rapidfuzz` package
#    installed (it's in requirements.txt) -- without it, generation fails
#    partway through with ImportError: rapidfuzz is required for string
#    distance.
generator = TestsetGenerator(llm=judge_llm, embedding_model=judge_embeddings)

# 5. Write test_synth_evaluation_dataset():
#    - Call generator.generate_with_langchain_docs(docs, testset_size=6)
#      (docs from step 3) to get a Testset
#    - For each row in testset.samples, pull the generated question off
#      row.eval_sample.user_input and the generated ground truth off
#      row.eval_sample.reference
#    - Run the question through ask() the same way as step 2, and build a
#      SingleTurnSample per row
#    - Score the samples the same way as step 2 (gather ascore() calls, mean
#      the .value scores) -- no evaluate()
#    - Assert on the result, but use a looser threshold than step 2's test:
#      synthetic questions/references are generated by the same small local
#      model that also judges them, so they're noisier than hand-written
#      goldens
def test_synth_evaluation_dataset():
    scores = list()
    testset = generator.generate_with_langchain_docs(load_knowledge_base_docs, testset_size=6)
    for row in testset.samples():
        response = ask(row.eval_sample.user_input)
        res = asyncio.run(faithfulness.ascore(
            user_input=row.eval_sample.user_input,
            response=response["answer"],
            retrieved_contexts=response["retrieved_contexts"],
        ))
        scores.append(res.value)
    assert mean_score(scores) >= 0.7

# 6. Write test_evaluation_dataset_from_list(): the same idea as step 2, but
#    built the way you'd actually do it against a real app -- from a list
#    of plain dicts (e.g. pulled from your app's own logs/traces), not
#    hand-built SingleTurnSample objects.
#    - Build a list of dicts, one per golden, each with keys "user_input",
#      "retrieved_contexts", "response", and "reference" -- run ask() for
#      each question first to fill in "response"/"retrieved_contexts", same
#      as step 2
#    - Call EvaluationDataset.from_list(rows) to get the samples container
#      (EvaluationDataset itself is NOT deprecated -- it's just a sample
#      holder; only evaluate() is), then score dataset.samples the same way
#      as step 2
#    - Aggregate and assert the same way as step 2
#    The keys must match the field names exactly ("user_input",
#    "retrieved_contexts", "response", "reference"). from_list() itself won't
#    complain about a typo'd key (it just quietly drops it, leaving that
#    field None) -- the failure surfaces later, when ascore() gets None for a
#    field the metric needs and errors out. Worth deliberately typo-ing one
#    yourself once, just to see that failure and recognize it later.

goldens = [
    {
        "user_input": "What's the refund policy?",
        "retrieved_contexts": [],
        "response": "",            
        "reference": "Flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method.",
    },
    {
        "user_input": "What's the baggage policy?",
        "retrieved_contexts": [],
        "response": "",
        "reference": "Economy passengers may check one bag up to 23kg free of charge; additional bags can be added for an extra fee.",
    },
    {
        "user_input": "How do I reach the next loyalty tier?",
        "retrieved_contexts": [],
        "response": "",
        "reference": "Qualifying miles flown per calendar year: Blue (0 miles, default), Silver (25,000 miles), Gold (50,000 miles), and Platinum (100,000 miles)."
    },
]

def test_evaluation_dataset_from_list():
    scores = list()
    samples = EvaluationDataset.from_list(goldens)
    for sample in samples:
        response = ask(sample.user_input)
        res = asyncio.run(faithfulness.ascore(
            user_input=sample.user_input,
            response = response["answer"],
            retrieved_contexts=response["retrieved_contexts"]
        ))
        scores.append(res)
    assert mean_score(scores) >= 0.7