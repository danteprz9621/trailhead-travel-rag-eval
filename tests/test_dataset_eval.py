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

from glob import glob

from openai import AsyncOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import TextLoader

from agents.rag_agent import ask


# 1. Build an AsyncOpenAI client pointed at Ollama's OpenAI-compatible
#    endpoint (same as test_rag_agent.py step 1), then:
#    - judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
#    - judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)
#      (needed for TestsetGenerator in step 4, not for the Faithfulness
#      metric itself)
#    - a Faithfulness instance, using ragas.metrics (not
#      ragas.metrics.collections) -- this is the version that works with
#      evaluate()/EvaluationDataset
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)

# 2. Write test_evaluation_dataset():
#    - Hand-write a small list of (question, reference) goldens using real
#      facts from data/knowledge_base/ (e.g. baggage/refund/loyalty policy
#      numbers)
#    - Run each question through ask() (the RAG agent) to get a response and
#      its retrieved_contexts, and build a SingleTurnSample per golden with
#      user_input, response, retrieved_contexts, and reference
#    - Wrap them all in one EvaluationDataset and call evaluate() once with
#      the Faithfulness metric from step 1 -- this is the "batch" pattern:
#      one evaluate() call scores every sample in the dataset
#    - Assert on the result


# 3. Write load_knowledge_base_docs():
#    - Use glob() to find every .txt file under data/knowledge_base/
#    - Load each with langchain_community's TextLoader (not DirectoryLoader
#      -- its default loader needs the heavyweight `unstructured` package,
#      which this project doesn't install)
#    - Return the combined list of loaded documents


# 4. Build a TestsetGenerator(llm=judge_llm, embedding_model=judge_embeddings)
#    using the judge models from step 1. Needs the `rapidfuzz` package
#    installed (it's in requirements.txt) -- without it, generation fails
#    partway through with ImportError: rapidfuzz is required for string
#    distance.


# 5. Write test_synth_evaluation_dataset():
#    - Call generator.generate_with_langchain_docs(docs, testset_size=6)
#      (docs from step 3) to get a Testset
#    - For each row in testset.samples, pull the generated question off
#      row.eval_sample.user_input and the generated ground truth off
#      row.eval_sample.reference
#    - Run the question through ask() the same way as step 2, and build a
#      SingleTurnSample per row
#    - evaluate() the whole batch with the Faithfulness metric from step 1
#    - Assert on the result, but use a looser threshold than step 2's test:
#      synthetic questions/references are generated by the same small local
#      model that also judges them, so they're noisier than hand-written
#      goldens


# 6. Write test_evaluation_dataset_from_list(): the same idea as step 2, but
#    built the way you'd actually do it against a real app -- from a list
#    of plain dicts (e.g. pulled from your app's own logs/traces), not
#    hand-built SingleTurnSample objects.
#    - Build a list of dicts, one per golden, each with keys "user_input",
#      "retrieved_contexts", "response", and "reference" -- run ask() for
#      each question first to fill in "response"/"retrieved_contexts", same
#      as step 2
#    - Call EvaluationDataset.from_list(rows) instead of constructing
#      SingleTurnSample objects directly
#    - evaluate() and assert the same way as step 2
#    The keys must match the field names exactly ("user_input",
#    "retrieved_contexts", "response", "reference"). Verified directly
#    against this Ragas version: from_list() itself won't complain about a
#    typo'd key (it just quietly drops it, leaving that field None), but
#    evaluate() does catch it -- it raises a clear ValueError naming the
#    missing column as soon as a metric needs a field that came back None.
#    Worth deliberately typo-ing one yourself once, just to see that error
#    message and recognize it later if it shows up for real.
