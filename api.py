from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from scraper_agent.db import init_db, ensure_session, add_message, save_results, get_results, list_messages
from agent import run_agent
import os
import sys
import asyncio
import json
import re

from dotenv import load_dotenv

load_dotenv(override=True)

# On Windows: Use ProactorEventLoopPolicy to support asyncio subprocess APIs
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | api | %(message)s")
log = logging.getLogger("scraper_agent.api")

app = FastAPI(title="Scraper Agent API")
init_db()


class ChatRequest(BaseModel):
    url: str
    user_prompt: str
    max_pages: Optional[int] = None
    max_iterations: Optional[int] = None
    session_id: str


class ChatResponse(BaseModel):
    text: str
    results: list[Any]
    session_id: str
    schema: dict[str, Any] | None = None
    visited_count: int | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Handle a chat request by running the scraping agent, persisting state, and
    returning a summarized response with structured results.


    This endpoint:

    - Ensures a session exists and appends the user's prompt to the conversation.

    - Loads recent conversation memory (up to 20 messages) and any prior scrape results for the session.

    - Invokes the scraping agent with the provided URL, prompt, and limits.

    - Saves the agent's outputs (schema, items, visited count) for future reuse.

    - Adds a helpful assistant message to the conversation, preferring an LLM-provided summary when the agent reuses previous results.
    
    
    Args:

        req (ChatRequest):

            session_id (str): Identifier for the chat session; used to persist memory and results.

            user_prompt (str): The user's instruction or question guiding the agent.

            url (str): The starting URL for the scraper/agent to process.

            max_pages (int | None): Optional page-visit limit for the crawl.

            max_iterations (int | None): Optional iteration cap for the agent.


    Returns:

        ChatResponse:

            text (str): A human-readable summary of what was done (reuse summary or scrape summary).

            results (list[dict]): The structured items extracted by the agent.

            session_id (str): Echo of the provided session identifier.

            schema (dict | None): The inferred or provided schema describing item structure.

            visited_count (int | None): Number of pages the agent reports having visited.


    Side Effects:

    - Writes conversation messages and agent outputs to persistent storage for the session.

    - Emits log entries describing the run (session, URL, limits).

    - May reuse and refine previously saved results depending on the agent's chosen mode.


    Raises:
    
    - Propagates I/O, network, validation, or agent execution errors that occur during run or persistence.
    """

    # Ensure session and store user message
    ensure_session(req.session_id)
    add_message(req.session_id, "user", req.user_prompt)

    # Retrieve memory and last results for agent decision
    prior = get_results(req.session_id)
    memory = list(reversed(list_messages(req.session_id, limit=20)))

    # Run agent fresh with given parameters
    log.info("chat: running agent | session=%s url=%s pages=%s iters=%s", req.session_id, req.url, req.max_pages, req.max_iterations)
    outcome = await run_agent(
        req.url,
        req.user_prompt,
        max_pages=req.max_pages,
        max_iterations=req.max_iterations,
        memory=memory,
        prior_results=prior or {},
    )

    # Persist results for future follow-ups
    save_results(req.session_id, outcome.get("schema", {}), outcome.get("items", []), int(outcome.get("visited_count", 0)))

    mode = outcome.get("mode") or "crawl"
    if mode == "reuse":
        # Prefer a more helpful LLM-provided summary if available
        text = outcome.get("answer_text") or f"Reused previous results and applied your instruction. Returned {len(outcome.get('items', []))} items."
    else:
        text = f"Scraped {len(outcome.get('items', []))} items from {req.url}."
    add_message(req.session_id, "assistant", text)

    return ChatResponse(text=text, results=outcome.get("items", []), session_id=req.session_id, schema=outcome.get("schema"), visited_count=outcome.get("visited_count"))


async def run_in_thread(fn):
    import asyncio
    return await asyncio.to_thread(fn)
