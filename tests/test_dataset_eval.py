"""
Agent under test: agents/rag_agent.py
"""
import asyncio
from glob import glob

from openai import AsyncOpenAI
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import TextLoader

from agents.rag_agent import ask


def build_judges():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client, max_tokens=4096)
    judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)
    return judge_llm, judge_embeddings


def mean_score(scores) -> float:
    return sum(scores) / len(scores)


goldens = [
    SingleTurnSample(
        user_input="What's the refund policy?",
        retrieved_contexts=[],
        response="",
        reference="There's a refund policy and flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method"
    ),
    SingleTurnSample(
        user_input="What's the baggage policy?",
        retrieved_contexts=[],
        response="",
        reference="Economy passengers may check one bag up to 23kg free of charge, additional bags can be added by paying an additional cost"
    )
]

def test_evaluation_dataset():
    # Hand-written goldens, each scored against a fresh judge.
    judge_llm, _ = build_judges()
    faithfulness = Faithfulness(judge_llm)
    scores = list()
    for golden in goldens:
        result = ask(golden["user_input"])
        res = asyncio.run(faithfulness.ascore(
            user_input=golden["user_input"],
            response=result["answer"],
            retrieved_contexts=result["retrieval_context"],
        ))
        scores.append(res.value)

    assert mean_score(scores) >= 0.7


def load_knowledge_base_docs():
    """Load every .txt file under data/knowledge_base/ as LangChain docs."""
    docs = []
    for path in glob("data/knowledge_base/**/*.txt", recursive=True):
        docs.extend(TextLoader(path).load())
    return docs


def test_synth_evaluation_dataset():
    # Same idea as test_evaluation_dataset(), but the goldens are generated
    # from data/knowledge_base/ by TestsetGenerator instead of hand-written
    # -- noisier, hence the looser threshold below.
    judge_llm, judge_embeddings = build_judges()
    generator = TestsetGenerator(llm=judge_llm, embedding_model=judge_embeddings)
    faithfulness = Faithfulness(judge_llm)

    scores = list()
    testset = generator.generate_with_langchain_docs(load_knowledge_base_docs(), testset_size=6)
    for row in testset.samples:
        response = ask(row.eval_sample.user_input)
        res = asyncio.run(faithfulness.ascore(
            user_input=row.eval_sample.user_input,
            response=response["answer"],
            retrieved_contexts=response["retrieval_context"],
        ))
        scores.append(res.value)
    assert mean_score(scores) >= 0.7


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
    # Same as test_evaluation_dataset(), but built from plain dicts via
    # EvaluationDataset.from_list() -- the shape a real app's logs/traces
    # would already be in.
    judge_llm, _ = build_judges()
    faithfulness = Faithfulness(judge_llm)
    scores = list()
    samples = EvaluationDataset.from_list(goldens)
    for sample in samples:
        response = ask(sample.user_input)
        res = asyncio.run(faithfulness.ascore(
            user_input=sample.user_input,
            response = response["answer"],
            retrieved_contexts=response["retrieval_context"]
        ))
        scores.append(res)
    assert mean_score(scores) >= 0.7
