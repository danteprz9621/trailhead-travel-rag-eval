# Ragas Practice Project: Testing a RAG Agent

A practice project for evaluating a RAG (retrieval-augmented generation)
pipeline with [Ragas](https://docs.ragas.io) — deliberately scoped to just
that, since RAG evaluation is what Ragas is actually built and battle-tested
for. It reuses the "Trailhead Travel" RAG agent and knowledge base from
[`../deepeval-capstone`](../deepeval-capstone), a broader capstone that also
covered a single-turn agent, a chatbot, and safety/red-teaming with
DeepEval. This project isn't a 1:1 port of that scope: DeepEval is a
general-purpose eval framework that covers all of those domains natively and
well, but Ragas's non-RAG metrics (safety-style criteria via the generic
`AspectCritic` escape hatch, or the newer `TopicAdherenceScore` for
multi-turn role adherence) turned out to be noticeably less mature and less
reliable than its core RAG metrics when tried against a small local judge
model — see [Why not the other agents too?](#why-not-the-other-agents-too)
below for what that looked like in practice.

The `tests/` files here are **skeletons, not finished tests** — each one has
a docstring plus numbered comments describing what to build step by step;
you write the actual code. This mirrors how the DeepEval project's `tests/`
files started out (see that project's initial git commit) before being
filled in and corrected over several commits.

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
│   └── knowledge_base/        # same 7 policy docs as deepeval-capstone, reused as-is
├── scripts/
│   └── measure_noise.py            # SKELETON: eval-noise floor calibration (run manually, not part of pytest)
├── tests/
│   ├── test_rag_agent.py           # SKELETON: Faithfulness, ContextPrecision/Recall, AnswerRelevancy, ResponseGroundedness (collections API + ascore())
│   ├── test_dataset_eval.py        # SKELETON: EvaluationDataset + TestsetGenerator + evaluate() (legacy metrics API)
│   ├── test_custom_metrics.py      # SKELETON: domain-specific AspectCritic (legacy metrics API)
├── requirements.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

This structure follows `ragas-course.md` (one level above `files/`, i.e.
`../../ragas-course.md` from here) module by module — see
[Course coverage](#course-coverage) below for the exact mapping, and a
couple of places where the course's sample code doesn't match what this
Ragas version actually does.

`agents/rag_agent.py` and `data/knowledge_base/` are reused unchanged from
`deepeval-capstone` — the agent never depended on DeepEval, just `ollama`
and `scikit-learn`.

## Setup

```bash
cd ragas-capstone
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

Same setup as `deepeval-capstone` ended up with, applied from the start this
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

  One real API split remains, independent of how the judge is built:
  `test_rag_agent.py` uses `ragas.metrics.collections` metrics scored
  directly with `await metric.ascore(...)`, while `test_dataset_eval.py`,
  `test_custom_metrics.py`, and `scripts/measure_noise.py` use the older
  `ragas.metrics` classes scored via `evaluate()`/`EvaluationDataset` —
  `evaluate()` flatly rejects `collections` metric objects
  (`TypeError: All metrics must be initialised metric objects`), so the two
  metric families can't be mixed in one `evaluate()` call. See the cheat
  sheet below for which file uses which.

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

Once you've filled in a skeleton file:

```bash
pytest tests/test_rag_agent.py -v   # one file at a time as you go
pytest tests/ -v                    # everything, once it's all filled in
```

Every metric here uses an LLM as judge, and `test_dataset_eval.py`'s
`TestsetGenerator` additionally builds a small knowledge graph from
`data/knowledge_base/` before generating goldens — expect that one to be the
slowest in the suite (well over a minute on local 7B/8B models), same as the
Synthesizer-based test was in `deepeval-capstone`.

## Gotchas to get right while implementing

These came up while validating that this project's approach actually works
end-to-end against the local models above, before the test files were
turned back into skeletons for you to fill in. Keep them in mind — they're
easy ways to end up with a test that looks right but never actually checks
anything:

- **Two Ragas metric families, and `evaluate()` won't mix them — but this
  isn't a judge-construction limitation.** An earlier version of this README
  claimed `TestsetGenerator` *required* the legacy `LangchainLLMWrapper`/
  `LangchainEmbeddingsWrapper` types. That was wrong — tested directly:
  `TestsetGenerator` works fine with `llm_factory`/`embedding_factory`
  judges (once `rapidfuzz` is installed; see below). The real, still-true
  constraint is narrower: `evaluate()` only accepts `ragas.metrics.*`
  (legacy) metric objects — passing a `ragas.metrics.collections.*` object
  raises `TypeError: All metrics must be initialised metric objects`. So
  `test_rag_agent.py` (collections + `ascore()`) and `test_dataset_eval.py`/
  `test_custom_metrics.py`/`scripts/measure_noise.py` (legacy + `evaluate()`)
  are on different metric families for that reason, not because of how the
  judge itself was built.
- **Legacy `ResponseRelevancy` silently breaks with `embedding_factory`'s
  embeddings — confirmed, not documented anywhere.** It needs an object
  with an `.embed_query()` method (what `LangchainEmbeddingsWrapper`
  produces); `embedding_factory("openai", ..., client=client)`'s modern-
  interface object doesn't have one. It doesn't raise — `evaluate()` just
  returns `{'answer_relevancy': nan}`, the exact silent-failure mode
  `ragas-course.md` warns about for a different reason (a typo'd dataset
  column). This is why `test_dataset_eval.py` only uses `Faithfulness` (no
  embeddings needed) rather than also covering `ResponseRelevancy` —
  `AnswerRelevancy` coverage lives in `test_rag_agent.py`'s collections API
  instead, where `embedding_factory` works fine.
- **`TestsetGenerator` needs `rapidfuzz` installed, unrelated to any of the
  above.** Without it, `generate_with_langchain_docs(...)` fails partway
  through with `ImportError: rapidfuzz is required for string distance`.
  It's in `requirements.txt`; if you're seeing this error, your environment
  is stale — reinstall.
- **`llm_factory()` needs an OpenAI-SDK-style client, not a LangChain one —
  and it needs the async variant.** `AsyncOpenAI`, not `OpenAI`: the
  collections metrics' `.ascore()` is async internally
  (`llm.agenerate(...)`), and it raises a clear `TypeError` if you hand it a
  sync client. Point that client's `base_url` at Ollama's OpenAI-compatible
  endpoint (`http://localhost:11434/v1`) and pass any placeholder string as
  `api_key` — Ollama doesn't check it, but the OpenAI SDK requires the field
  to be non-empty.
- **`evaluate()` never fails your test for you** (`test_dataset_eval.py`,
  `test_custom_metrics.py`). Unlike DeepEval, a low Ragas score doesn't raise — you always need an
  explicit `assert` on the returned result. Figure out what
  `EvaluationResult` actually looks like (what you can index into it with)
  before you decide how to assert on it. Collections metrics
  (`test_rag_agent.py`) have the same issue in a different shape: `ascore()`
  returns a `MetricResult`, and the numeric score is on its `.value` — an
  un-asserted `MetricResult` object is truthy, so a bare `assert result`
  would always pass no matter the score.
- **Wire a real reference into any correctness-style check.** A metric like
  `ContextPrecisionWithReference`/`ContextRecall` (or their legacy
  `LLMContextPrecisionWithReference`/`LLMContextRecall` equivalents) is only
  meaningful if you actually pass a real `reference` — an empty or made-up
  one means the judge has nothing real to compare against.
- **`ResponseGroundedness` can be strict about extrapolation.** It's scoring
  whether the response is grounded in `retrieved_contexts`, not whether the
  response is *correct* — a reasonable inference that goes slightly beyond
  what the context literally says (e.g. extending an explicit "Latin-alphabet
  corrections only" rule to cover a script the context never mentions) can
  score low even though the inference itself is defensible. Worth deciding
  for yourself whether that's a real failure before picking a threshold.
- **Watch for accidental tuples.** In `test_dataset_eval.py`, it's an easy
  typo to write `sample = SingleTurnSample(...),` with a trailing comma,
  silently making `sample` a 1-tuple instead of a `SingleTurnSample`.
- **Answer every golden with the RAG agent, hand-written or synthetic.**
  Both halves of `test_dataset_eval.py` are questions about
  `data/knowledge_base/` content, so both need to go through
  `agents/rag_agent.py`'s `ask()` (which returns `retrieved_contexts`) for
  `Faithfulness` to have something real to check the answer against.
- **`result["metric_name"]` is a list, not a scalar — confirmed against
  this Ragas version, contradicting `ragas-course.md`'s own Module 9 sample
  code.** The course writes `score = ragas_result[metric]; assert score >=
  floor(metric)`, treating the subscript as a single aggregate number.
  Tested directly against this installed version: it's a **list of
  per-sample scores**. `print(result)` shows a nicely-formatted mean, but
  indexing into the object itself does not give you that mean — comparing
  the list to a float raises `TypeError: '>=' not supported between
  instances of 'list' and 'float'`. Reduce it yourself with
  `statistics.mean(result[metric])` before comparing to anything. This
  matters most in `scripts/measure_noise.py` (and would matter in any CI
  gate built on top of it later).
- **`llm_factory("gpt-4o-mini")` (the course's zero-argument shortcut) also
  doesn't work as written against this version.** It now requires an
  explicit `client=`, and raises `ValueError: llm_factory() requires a
  client instance` without one — matching what `test_rag_agent.py` already
  does (build an `AsyncOpenAI` client explicitly, don't rely on env-var
  auto-detection).

## Ragas metric → DeepEval metric cheat sheet

**`test_rag_agent.py`** (current `ragas.metrics.collections` API):

| DeepEval | Ragas | Notes |
|---|---|---|
| `FaithfulnessMetric` | `Faithfulness` | Decomposes the response into claims, checks each against `retrieved_contexts` |
| `AnswerRelevancyMetric` | `AnswerRelevancy` | The collections-API name for what the course and the legacy API both call `ResponseRelevancy`/`answer_relevancy` — needs `embeddings` too |
| `ContextualPrecisionMetric` | `ContextPrecisionWithReference` | Needs `reference` (ground truth), not `expected_output`; `ContextPrecisionWithoutReference` is the reference-free variant the course recommends for production |
| `ContextualRecallMetric` | `ContextRecall` | Same |
| `HallucinationMetric` | `ResponseGroundedness` | Checks the response is grounded in `retrieved_contexts` — see the strictness gotcha above |

**`test_dataset_eval.py`** and **`test_custom_metrics.py`** (legacy
`ragas.metrics` + `evaluate()` API — not because of the judge, which is
`llm_factory` here too, but because `evaluate()` requires this metric
family, and `TestsetGenerator` needs its `llm`/`embedding_model`):

| DeepEval | Ragas | Notes |
|---|---|---|
| `FaithfulnessMetric` | `Faithfulness` | Legacy import path (`ragas.metrics`, not `ragas.metrics.collections`); no embeddings needed |
| `AnswerRelevancyMetric` | *(not used here)* | Legacy `ResponseRelevancy` is broken with `embedding_factory`'s embeddings (see gotchas) — this coverage lives in `test_rag_agent.py`'s `AnswerRelevancy` instead |
| `GEval` (custom criteria) | `AspectCritic` | `test_custom_metrics.py` — binary yes/no judge, no collections equivalent exists yet |
| `Synthesizer` | `TestsetGenerator` | Both build synthetic goldens from local docs; Ragas builds a knowledge graph first |

## Course coverage

Mapping from `ragas-course.md`'s modules to what's in this project:

| Module | Where |
|---|---|
| 1. Orientation | This README; [Why not the other agents too?](#why-not-the-other-agents-too) below |
| 2. Data model (`SingleTurnSample`/`EvaluationDataset`/`.from_list()`) | `test_dataset_eval.py` steps 2 and 6 |
| 3. Evaluator LLM | Every test file; see [Running fully local with Ollama](#running-fully-local-with-ollama) |
| 4. The four core metrics | `test_rag_agent.py` (all four, steps 3-6) |
| 5. Running a full evaluation, reading scores diagnostically | `test_dataset_eval.py` (uses `evaluate()`/batch scoring); diagnostic table below |
| 6. Non-determinism / eval-noise floor / `RunConfig` | `scripts/measure_noise.py` |
| 7. Test set generation | `test_dataset_eval.py` steps 3-5 (`TestsetGenerator`) |
| 8. Custom metrics (`AspectCritic`) | `test_custom_metrics.py` |
| 9. Wiring alongside DeepEval in CI | **Deliberately not built yet** — see note below |
| 10. Capstone brief | This whole project — see below for what maps to what |

Module 9 and the CI-gate half of the Module 10 capstone brief are
intentionally not in this project yet — `scripts/measure_noise.py` (Module
6) is here and stands on its own, but wiring its output into an actual
pytest gate (what Module 9 describes) is a deliberate next step, not
something to build until you're ready for it.

Capstone brief, mapped to this project's actual deliverables:

1. **Add retrieval** → `agents/rag_agent.py` (already done, reused from `deepeval-capstone`)
2. **Golden dataset, generated + hand-corrected** → `test_dataset_eval.py` steps 3-5; the "hand-correct before it becomes a gate" discipline is on you when you pick `testset_size` and actually read what comes back, not something code can enforce
3. **Measure eval-noise floor** → `scripts/measure_noise.py`
4. **RAGAS gate with noise-aware floors** → not built yet, by choice — see the Module 9 note above
5. **One custom `AspectCritic`, gated high** → `test_custom_metrics.py`
6. **DeepEval as the fast per-PR layer, RAGAS on schedule/release** → conceptual only for now, covered in [Why not the other agents too?](#why-not-the-other-agents-too); becomes concrete once item 4 exists
7. **Write up the flakiness section** → your job, once you've actually run `scripts/measure_noise.py` and have real numbers to write about — a writeup with placeholder numbers isn't the deliverable the course is asking for

### Reading scores diagnostically (Module 5)

Print this and pin it, same as the course says. Don't average the four
core metrics into one score — read them as a decision tree instead:

| Pattern | Diagnosis | First thing to check |
|---|---|---|
| Faithfulness low, precision/recall high | Retrieval's fine, the *generator* is ignoring context | Prompt, model — not a retriever problem |
| Context recall low | Never retrieved the needed docs | Chunking, embeddings, `top_k` — no prompt tweak fixes this |
| Context precision low, recall high | Retrieving the right stuff *plus* a lot of noise | Reranker, lower `top_k` |
| Everything high, but a real user would still be unhappy | Golden dataset doesn't reflect real queries | Regenerate it (`test_dataset_eval.py`'s `TestsetGenerator` step) |

`result.to_pandas()` (on an `evaluate()` result, i.e. in `test_dataset_eval.py`)
is where you'd actually go look — one row per sample, one column per
metric — to find which specific question tanked a score instead of only
seeing the aggregate.

## Why not the other agents too?

`deepeval-capstone` also tests a single-turn agent, a multi-turn chatbot,
and safety/red-teaming. Ragas *can* technically be pointed at those — it has
grown a broader metrics catalog over time beyond pure RAG, including
general-purpose criteria (`AspectCritic`, `RubricsScore`) and agent/
multi-turn metrics (`TopicAdherenceScore`, `AgentGoalAccuracyWithReference`)
— but while building this project, that non-RAG surface was noticeably
rougher than the RAG core:

- `TopicAdherenceScore` (Ragas's own dedicated multi-turn "stayed on topic"
  metric, the closest thing to DeepEval's `RoleAdherenceMetric`) reliably
  broke on `llama3.1:8b`: its judge prompt requires structured JSON output,
  and the local model kept failing to produce valid JSON for it
  (`OutputParserException`).
- Safety-style checks (bias, toxicity, PII leakage) have no dedicated Ragas
  metrics at all — the only option was `AspectCritic` with a hand-written
  definition, a generic fallback rather than a purpose-built, tuned detector
  the way DeepEval's `BiasMetric`/`ToxicityMetric`/`PIILeakageMetric` are.
  And even that fallback is on its way out without a replacement: `AspectCritic`
  itself is deprecated in favor of the newer `ragas.metrics.collections` API
  (see the RAG cheat sheet above), but as of this Ragas version there's no
  `AspectCritic` in `collections` at all to migrate to — a real gap, not
  just a rename.

None of that is a reason Ragas is "bad" — it's a reason to use the right
tool for each job. `Faithfulness` and the `ContextPrecision`/`ContextRecall`
family are genuinely Ragas's strongest, most reliable metrics, which is
exactly what this project focuses on. For single-turn correctness,
conversational quality, or safety/red-teaming, DeepEval (or a dedicated tool
like Guardrails AI / NeMo Guardrails for safety specifically) is the
better-fitting choice — that's what `deepeval-capstone` already covers.
