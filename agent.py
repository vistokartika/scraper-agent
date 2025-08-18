from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from scraper_agent.web import fetch_markdown
from scraper_agent.schema import generate_schema
from scraper_agent.extract import extract_with_llm
from scraper_agent.crawl import guess_next_page_urls, gather_related_links

load_dotenv(override=True)


# Configure logging once for the agent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("scraper_agent.agent")


@dataclass
class AgentConfig:
    max_steps: int = 12
    max_links_per_page: int = 40
    paginate_cap: int = 10


def _parse_json_block(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


async def llm_decide_next(client: OpenAI, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the LLM to choose the next action based on current state.
    Returns a dict: {"action": "extract|paginate|follow|stop", "params": {...}}
    """
    provider = os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini")
    model = provider.split("/", 1)[1] if "/" in provider else provider

    system = (
        "You are an effective web scraping agent planner. You can call tools in sequence to accomplish the user's request.\n"
        "Available tools: \n"
        "- paginate(url, max_pages): returns page URLs if the site supports ?page=\n"
        "- follow(links): choose a small set of the most relevant links to follow next (subset of candidates)\n"
        "- extract(url): attempt data extraction using the already generated schema\n"
        "- stop(): when you believe the goal is achieved.\n\n"
        "Rules: Prefer extracting on each list page, use pagination when the query explicitly requests multiple pages.\n"
        "When looking for contact details, follow organization or contact page links.\n"
        "Return strict JSON with keys: action, params. No prose."
    )

    user = (
        f"User query: {query}\n\n"
        f"State: {json.dumps(state, ensure_ascii=False)}\n\n"
        "Return one of: \n"
        "- {\"action\": \"paginate\", \"params\": {\"max_pages\": <int>}}\n"
        "- {\"action\": \"follow\", \"params\": {\"urls\": [<url> ...]}}\n"
        "- {\"action\": \"extract\", \"params\": {}}\n"
        "- {\"action\": \"stop\", \"params\": {}}\n"
    )

    def _run():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=200,
        )

    logger.info("planner: calling LLM for next action | state_keys=%s", list(state.keys()))
    resp = await asyncio.to_thread(_run)
    content = (resp.choices[0].message.content or "").strip()
    logger.info("planner: LLM raw decision=%s", content.replace("\n", " ")[:500])
    obj = _parse_json_block(content)
    if not isinstance(obj, dict) or "action" not in obj:
        logger.warning("planner: unparsable decision, defaulting to extract")
        return {"action": "extract", "params": {}}
    # Sanitize
    act = obj.get("action")
    params = obj.get("params") or {}
    if act not in {"extract", "paginate", "follow", "stop"}:
        logger.warning("planner: unknown action=%s, defaulting to extract", act)
        act = "extract"
    if act == "follow":
        urls = params.get("urls") or []
        if not isinstance(urls, list):
            urls = []
        params = {"urls": urls[: state.get("max_links_per_decision", 10)]}
    if act == "paginate":
        mp_raw = params.get("max_pages", 1)
        mp = 1
        if isinstance(mp_raw, (int, float)) and not isinstance(mp_raw, bool):
            mp = int(mp_raw)
        else:
            s = str(mp_raw).strip()
            mp = int(s) if s.isdigit() else 1
        params = {"max_pages": max(1, min(mp, state.get("paginate_cap", 10)))}
    logger.info("planner: decision action=%s params=%s", act, params)
    return {"action": act, "params": params}


async def llm_decide_followup(client: OpenAI, query: str, memory: List[Dict[str, Any]] | None, prior_items_count: int) -> Dict[str, Any]:
    """Decide whether to reuse prior results or re-crawl, based on chat memory and the new prompt.
    Returns: {"mode": "reuse"|"crawl", "reason": str}
    """
    provider = os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini")
    model = provider.split("/", 1)[1] if "/" in provider else provider

    last_msgs = (memory or [])[:6]
    last_msgs = list(reversed(last_msgs))  # oldest first
    mem_snippets = [f"{m.get('role')}: {m.get('content')}" for m in last_msgs]
    mem_text = "\n".join(mem_snippets)

    system = (
        "You are an agent orchestrator. Given recent chat messages, the user's new prompt, and a count of previous scraped items, "
        "decide if this is a follow-up that can be answered by reusing/transforming the prior items (no re-crawl), or if we must crawl again.\n"
        "Return JSON: {\"mode\": \"reuse\"|\"crawl\", \"reason\": <short>}."
    )
    user = (
        f"Prior items count: {prior_items_count}\n\nLast messages (oldest first):\n{mem_text}\n\nCurrent prompt: {query}\n"
        "Reply with JSON only."
    )

    def _run():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=100,
        )

    logger.info("planner: followup decision | prior_items=%d", prior_items_count)
    resp = await asyncio.to_thread(_run)
    content = (resp.choices[0].message.content or "").strip()
    logger.info("planner: followup raw=%s", content.replace("\n", " ")[:300])
    obj = _parse_json_block(content)
    mode = (obj.get("mode") if isinstance(obj, dict) else None) or "crawl"
    reason = (obj.get("reason") if isinstance(obj, dict) else None) or "default"
    if mode not in {"reuse", "crawl"}:
        mode = "crawl"
    logger.info("planner: followup -> mode=%s reason=%s", mode, reason)
    return {"mode": mode, "reason": reason}


async def transform_prior_items_with_llm(client: OpenAI, items: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
    """Use LLM to transform/filter prior items according to prompt.
    Returns a dict: {"items": [...], "answer_text": <optional str>}.
    On parse failure, falls back to returning the original items.
    """
    provider = os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini")
    model = provider.split("/", 1)[1] if "/" in provider else provider
    system = (
        "You are a precise data post-processor. You are given a JSON array of items scraped from a website and a user instruction.\n"
        "- Think step-by-step about what the instruction requests (e.g., filter by pay per day > 2000, select fields, deduplicate).\n"
        "- Then produce a STRICT JSON response in one of these forms (prefer the first):\n"
        "  {\n    \"items\": [<filtered_or_transformed_items>],\n    \"answer_text\": \"a short, human-readable summary of what changed and how many items\"\n  }\n"
        "  OR just a JSON array if transformation is trivial.\n"
        "Rules:\n"
        "- Output must be valid JSON with no markdown fences.\n"
        "- Keep only fields relevant to the instruction and key identifiers (e.g., role/title, location, pay/salary/rate, link/url, name).\n"
        "- Normalize obvious numeric fields when helpful (e.g., parse pay per day into a number in the same currency if possible).\n"
        "- Do not invent data not present in the input."
    )
    user = (
        f"Instruction: {prompt}\n\nItems JSON:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        "Return JSON only as specified."
    )

    def _run():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=1500,
        )

    logger.info("tool: transform_prior_items_with_llm | items=%d", len(items))
    resp = await asyncio.to_thread(_run)
    content = (resp.choices[0].message.content or "[]").strip()
    try:
        out = json.loads(content)
        # Preferred structured output
        if isinstance(out, dict) and isinstance(out.get("items"), list):
            return {"items": out.get("items", []), "answer_text": out.get("answer_text")}
        # Back-compat: plain array
        if isinstance(out, list):
            return {"items": out}
    except json.JSONDecodeError:
        pass
    # Fallback
    return {"items": items}


async def run_agent(
    url: str,
    query: str,
    max_pages: int | None = None,
    max_iterations: int | None = None,
    memory: List[Dict[str, Any]] | None = None,
    prior_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """LLM-driven agent loop with planning and tool calls (no hardcoded patterns)."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    cfg = AgentConfig()
    if isinstance(max_iterations, int) and max_iterations > 0:
        cfg.max_steps = max_iterations

    logger.info("agent: start | url=%s | query=%s | max_pages=%s", url, query, max_pages)
    visited: set[str] = set()
    to_visit: List[str] = [url]
    items: List[Any] = []
    schema: Optional[Dict[str, str]] = None
    # first_page_done reserved for future use

    # Follow-up decision: reuse prior items or crawl anew
    prior_items = []
    prior_schema: Dict[str, Any] = {}
    if isinstance(prior_results, dict):
        prior_items = list(prior_results.get("items") or [])
        prior_schema = dict(prior_results.get("schema") or {})
    if prior_items:
        decision = await llm_decide_followup(client, query, memory or [], len(prior_items))
        if decision.get("mode") == "reuse":
            transformed = await transform_prior_items_with_llm(client, prior_items, query)
            # Clean transformed results
            def _is_empty_value(v: Any) -> bool:
                if v is None:
                    return True
                if isinstance(v, str) and v.strip() == "":
                    return True
                if isinstance(v, (list, dict)) and len(v) == 0:
                    return True
                return False

            raw_items = transformed.get("items") if isinstance(transformed, dict) else transformed
            if not isinstance(raw_items, list):
                raw_items = []
            cleaned: List[Dict[str, Any]] = []
            for it in raw_items:
                if isinstance(it, dict):
                    filtered = {k: v for k, v in it.items() if not _is_empty_value(v)}
                    if filtered:
                        cleaned.append(filtered)
                else:
                    cleaned.append(it)
            logger.info("agent: reuse mode | items_raw=%d items_clean=%d", len(raw_items), len(cleaned))
            answer_text = None
            if isinstance(transformed, dict):
                at = transformed.get("answer_text")
                if isinstance(at, str) and at.strip():
                    answer_text = at.strip()
            return {"schema": prior_schema, "items": cleaned, "visited_count": 0, "mode": "reuse", "answer_text": answer_text}

    steps = 0
    while steps < cfg.max_steps and to_visit:
        steps += 1
        current = to_visit.pop(0)
        logger.info("step %d: visiting %s | queue=%d visited=%d items=%d", steps, current, len(to_visit), len(visited), len(items))
        if current in visited:
            continue
        visited.add(current)

        # Fetch and maybe bootstrap schema
        logger.info("tool: fetch_markdown(url=%s)", current)
        md = await fetch_markdown(current)
        logger.info("tool: fetch_markdown -> markdown_chars=%d", len(md))
        if schema is None:
            logger.info("tool: generate_schema(query len=%d, markdown_chars=%d)", len(query), len(md))
            schema = await generate_schema(md, query)
            logger.info("tool: generate_schema -> keys=%s", list(schema.keys()))

        # Extract on this page
        logger.info("tool: extract_with_llm(url=%s)", current)
        res = await extract_with_llm(md, query, schema, context=f"URL: {current}")
        if res and isinstance(res.data, list) and res.data:
            items.extend(res.data)
            logger.info("tool: extract_with_llm -> appended %d items (total=%d)", len(res.data), len(items))
        else:
            logger.info("tool: extract_with_llm -> no items extracted")

        # Gather candidate links from this page
        logger.info("tool: gather_related_links(base=%s, limit=%d)", current, cfg.max_links_per_page)
        candidates = await gather_related_links(md, current, patterns=None, limit=cfg.max_links_per_page)
        logger.info("tool: gather_related_links -> %d candidates", len(candidates))

        # Let LLM decide what to do next
        state = {
            "current_url": current,
            "collected_items": len(items),
            "have_schema": schema is not None,
            "candidates": candidates[:20],  # keep short
            "suggested_max_pages": max_pages,
            "paginate_cap": cfg.paginate_cap,
            "max_links_per_decision": 10,
        }
        decision = await llm_decide_next(client, query, state)
        action = decision.get("action")
        params = decision.get("params", {})
        logger.info("agent: action=%s params=%s", action, params)

        if action == "stop":
            logger.info("agent: stopping per planner decision")
            break
        elif action == "paginate":
            mp = int(params.get("max_pages", max_pages or 1))
            # If user provided max_pages, cap by that
            if max_pages:
                mp = min(mp, max_pages)
            logger.info("tool: guess_next_page_urls(base=%s, max_pages=%d)", current, mp)
            urls = guess_next_page_urls(current, max_pages=mp)
            logger.info("tool: guess_next_page_urls -> %d urls", len(urls))
            # Append unseen pages
            for u in urls:
                if u not in visited and u not in to_visit:
                    to_visit.append(u)
            logger.info("agent: enqueued %d pagination urls | queue=%d", len(urls), len(to_visit))
        elif action == "follow":
            chosen: List[str] = [u for u in params.get("urls", []) if isinstance(u, str)]
            if not chosen:
                # fallback: pick a few candidates
                chosen = candidates[:5]
            for u in chosen:
                if u not in visited and u not in to_visit:
                    to_visit.append(u)
            logger.info("agent: enqueued %d follow urls | queue=%d", len(chosen), len(to_visit))
        else:  # extract or default
            # If extraction chosen explicitly, try next page or next candidate automatically
            # to make progress. We'll push first few candidates.
            for u in candidates[:5]:
                if u not in visited and u not in to_visit:
                    to_visit.append(u)
            logger.info("agent: default enqueue %d candidates | queue=%d", min(5, len(candidates)), len(to_visit))

    # Clean results: remove keys with empty string/None, and drop objects that become empty
    def _is_empty_value(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        if isinstance(v, (list, dict)) and len(v) == 0:
            return True
        return False

    cleaned: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            filtered = {k: v for k, v in it.items() if not _is_empty_value(v)}
            if filtered:
                cleaned.append(filtered)
        else:
            cleaned.append(it)

    logger.info("agent: done | visited=%d items_raw=%d items_clean=%d", len(visited), len(items), len(cleaned))
    return {"schema": schema or {}, "items": cleaned, "visited_count": len(visited)}


if __name__ == "__main__":
    # The demo runner has been removed. Use api.py or import run_agent() directly.
    pass
