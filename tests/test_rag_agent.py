"""
Agent under test: agents/rag_agent.py
"""

import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecisionWithReference,
    ContextRecall,
    ResponseGroundedness,
    AnswerRelevancy
)

from agents.rag_agent import ask

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)


faithfulness = Faithfulness(judge_llm)
context_precision = ContextPrecisionWithReference(judge_llm)
context_recall = ContextRecall(judge_llm)
response_groundedness = ResponseGroundedness(judge_llm)
answer_relevancy = AnswerRelevancy(judge_llm, judge_embeddings)

user_question = "What's the refund policy?"
user_unanswerable_question = "Can I change then name on my ticket for the kanji equivalent?"
context_reference = "There's a refund policy and flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method"


def test_rag_answer_is_faithful_to_context():
   # Answer shouldn't contradict what was actually retrieved.
   result = ask(user_question)
   score = asyncio.run(
      faithfulness.ascore(
         user_input=user_question,
         response=result["answer"],
         retrieved_contexts=result["retrieval_context"]))
   assert score.value >= 0.7

def test_rag_retrieval_is_precise_and_complete():
   # Retriever should rank relevant chunks near the top (precision) and
   # not miss anything needed to answer (recall).
   result = ask(user_question)
   precision_score = asyncio.run(
      context_precision.ascore(user_input=user_question,
                               retrieved_contexts=result["retrieval_context"],
                               reference=context_reference)
   )
   recall_score = asyncio.run(
      context_recall.ascore(user_input=user_question,
                            retrieved_contexts=result["retrieval_context"],
                            reference=context_reference)
   )
   assert precision_score.value >= 0.7 and recall_score.value >= 0.7

def test_rag_no_hallucination_on_unanswerable_question():
   # Agent shouldn't invent an answer when the knowledge base doesn't
   # actually cover the question.
   result = ask(user_unanswerable_question)
   score = asyncio.run(
      response_groundedness.ascore(
         response=result["answer"],
         retrieved_contexts=result["retrieval_context"]
      )
   )
   assert score.value >= 0.5

def test_rag_answer_is_relevant():
   # Answer should stay grounded in retrieved context for an in-scope
   # question too.
   result = ask(user_question)
   score = asyncio.run(
      response_groundedness.ascore(
         response=result["answer"],
         retrieved_contexts=result["retrieval_context"]
      )
   )
   assert score.value >= 0.5
