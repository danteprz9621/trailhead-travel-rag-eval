# Ragas Practice Project: Testing a RAG Agent

A practice project for evaluating a RAG (retrieval-augmented generation)
pipeline with [Ragas](https://docs.ragas.io) — deliberately scoped to just
that, since RAG evaluation is what Ragas is actually built and battle-tested
for. It reuses the "Trailhead Travel" RAG agent and knowledge base from
[`../deepeval-practice`](../deepeval-practice), a separate practice project
that also covered a single-turn agent, a chatbot, and safety/red-teaming
with DeepEval. This project isn't a 1:1 port of that scope: DeepEval is a
general-purpose eval framework that covers all of those domains natively and
well, but Ragas's non-RAG metrics (safety-style criteria via the generic
`AspectCritic` escape hatch, or the newer `TopicAdherenceScore` for
multi-turn role adherence) turned out to be noticeably less mature and less
reliable than its core RAG metrics when tried against a small local judge
model.

Runs fully local and free from the start: the agent and every judge/embedder
model is served by [Ollama](https://ollama.com) — no API keys, no rate
limits, no cost. See [Running fully local with Ollama](#running-fully-local-with-ollama)
below.

## Project structure

```
ragas-capstone/
├── agents/
│   └── rag_agent.py           # TF-IDF retrieval over data/knowledge_base/ + LLM answer
├── data/
│   └── knowledge_base/        # same 7 policy docs as deepeval-practice, reused as-is
├── scripts/
│   └── measure_noise.py            # eval-noise floor calibration (run manually, not part of pytest)
├── tests/
│   ├── conftest.py                 # Windows event-loop-policy fix, applies to the whole suite
│   ├── test_rag_agent.py           # Faithfulness, ContextPrecision/Recall, AnswerRelevancy, ResponseGroundedness (collections API + ascore())
│   ├── test_dataset_eval.py        # EvaluationDataset + TestsetGenerator, scored via ascore() (collections API)
│   ├── test_custom_metrics.py      # domain-specific DiscreteMetric (legacy metrics API)
├── requirements.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

This structure follows `ragas-course.md` (one level above `files/`, i.e.
`../../ragas-course.md` from here) module by module.

`agents/rag_agent.py` and `data/knowledge_base/` are reused unchanged from
`deepeval-practice` — the agent never depended on DeepEval, just `ollama`
and `scikit-learn`.

## Setup

```bash
cd ragas-practice
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

No API key is required anywhere in this project — the agent and every
judge/embedder model runs locally via Ollama (see below).

Try the agent standalone before you start testing it:

```bash
python agents/rag_agent.py
```

## Running fully local with Ollama

Same setup as `deepeval-practice` ended up with, applied from the start this
time instead of two rounds of corrections (OpenAI → Gemini for free-tier
availability, then Gemini → Ollama once its 20-requests/day quota and
same-model self-evaluation bias became a problem — see that project's git
history and README):

- **The agent** generates with `qwen2.5-coder:7b`.
- **Judges** (every `tests/*.py` file) score with a *different* model,
  `llama3.1:8b`, built the same way in all three files: `llm_factory`
  wanting an OpenAI-SDK-style client rather than a LangChain one. Ollama
  exposes an OpenAI-compatible endpoint, so:

  ```python
  from openai import AsyncOpenAI
  from ragas.llms import llm_factory

  client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
  judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
  ```

  Metrics/subsystems that also need an **embedding** model
  (`AnswerRelevancy` in `test_rag_agent.py`, `TestsetGenerator` in
  `test_dataset_eval.py`) use `nomic-embed-text` via `embedding_factory`,
  built from the same client:

  ```python
  from ragas.embeddings import embedding_factory

  judge_embeddings = embedding_factory("openai", "nomic-embed-text", client=client)
  ```

  Every file in this project — `tests/*.py` and `scripts/measure_noise.py`
  alike — builds its judge this way now. None of them use
  `LangchainLLMWrapper`/`ChatOllama`/`LangchainEmbeddingsWrapper`/
  `OllamaEmbeddings` any more.

  One real API split remains, independent of how the judge is built: every
  file scores samples directly with `await metric.ascore(...)`, but the
  metric classes come from two different places. `test_rag_agent.py`,
  `test_dataset_eval.py`, and `scripts/measure_noise.py` use
  `ragas.metrics.collections` (`Faithfulness`, `ContextRecall`,
  `AnswerRelevancy`, etc.). `test_custom_metrics.py` is the one exception —
  `DiscreteMetric` (the current replacement for the removed `AspectCritic`)
  has no `ragas.metrics.collections` equivalent yet, so it stays on the
  legacy `ragas.metrics` import. Neither `evaluate()` nor `EvaluationDataset`'s
  batch scoring is actually used anywhere in this project any more. See the
  cheat sheet below for which file uses which.

Install [Ollama](https://ollama.com), then pull all three models and make
sure the Ollama server is running (it typically runs as a background
service after install):

```bash
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

`qwen2.5-coder:7b` and `llama3.1:8b` are roughly the same size (~5GB each);
on an 8GB GPU, expect Ollama to swap models in/out of VRAM between an agent
call and a judge call.

## Running tests

```bash
pytest tests/test_rag_agent.py -v   # one file at a time
pytest tests/ -v                    # the whole suite
```

Every metric here uses an LLM as judge, and `test_dataset_eval.py`'s
`TestsetGenerator` additionally builds a small knowledge graph from
`data/knowledge_base/` before generating goldens — expect that one to be the
slowest in the suite (well over a minute on local 7B/8B models), same as the
Synthesizer-based test was in `deepeval-practice`.

