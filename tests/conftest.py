"""
pytest auto-loads this file before collecting/running any test in this
directory. It exists for one reason: Windows defaults to
WindowsProactorEventLoopPolicy, and AsyncOpenAI clients built on top of it
don't survive being reused across separate asyncio.run() calls -- the
connection pool ends up bound to whichever event loop was active last time
it made a request, and the next asyncio.run() call (which creates a new
loop) hits "RuntimeError: Event loop is closed" trying to clean up a
connection tied to the old one.

This bites even within a single test: TestsetGenerator's
generate_with_langchain_docs() (test_dataset_eval.py) internally makes
several separate asyncio.run()-style calls across its pipeline stages
(transforms/extractors, then personas, then scenarios, then samples) using
the same judge_llm passed in -- confirmed directly: it fails partway
through (during persona generation) without this fix, even when judge_llm
was freshly built right before the call.

WindowsSelectorEventLoopPolicy doesn't have this problem. Switching to it
is the standard fix for this exact class of error on Windows with async
HTTP clients (httpx/openai's AsyncClient) -- confirmed fixed here directly:
the same TestsetGenerator call that crashed under the Proactor policy
completes cleanly under Selector.
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
