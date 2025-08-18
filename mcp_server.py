from __future__ import annotations

from typing import Any, Dict
import logging

from fastmcp import FastMCP
from dotenv import load_dotenv

from scraper_agent.web import fetch_markdown
from scraper_agent.schema import generate_schema
from scraper_agent.extract import extract_with_llm
from scraper_agent.crawl import guess_next_page_urls, gather_related_links

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | mcp | %(message)s")
log = logging.getLogger("scraper_agent.mcp")

app = FastMCP(name="scraper-agent-mcp", version="0.1.0")


@app.tool()
async def tool_fetch_markdown(url: str, timeout: float = 60.0) -> Dict[str, Any]:
    """Fetch a URL and return markdown content."""
    log.info("tool_fetch_markdown | url=%s timeout=%s", url, timeout)
    md = await fetch_markdown(url, timeout=timeout)
    log.info("tool_fetch_markdown | markdown_chars=%d", len(md))
    return {"url": url, "markdown": md}


@app.tool()
async def tool_generate_schema(markdown: str, query: str) -> Dict[str, Any]:
    """Generate a minimal flat schema for extraction from markdown and user query."""
    log.info("tool_generate_schema | query_len=%d markdown_chars=%d", len(query), len(markdown))
    schema = await generate_schema(markdown, query)
    log.info("tool_generate_schema | keys=%s", list(schema.keys()))
    return {"schema": schema}


@app.tool()
async def tool_extract_with_llm(markdown: str, query: str, schema: Dict[str, str], context: str | None = None) -> Dict[str, Any]:
    """Extract structured data using Crawl4AI LLM extraction with a given schema."""
    log.info("tool_extract_with_llm | context=%s", (context or "")[:120])
    res = await extract_with_llm(markdown, query, schema, context=context)
    if res is None:
        log.info("tool_extract_with_llm | result=None")
        return {"ok": False, "error": "extraction_failed"}
    n = len(res.data) if isinstance(res.data, list) else 1
    log.info("tool_extract_with_llm | items=%d model=%s", n, res.model)
    return {"ok": True, "data": res.data, "raw": res.raw, "model": res.model, "usage": res.usage}


@app.tool()
async def tool_guess_pagination(url: str, max_pages: int) -> Dict[str, Any]:
    """Guess a list of page URLs starting from the given URL up to max_pages (inclusive)."""
    log.info("tool_guess_pagination | url=%s max_pages=%d", url, max_pages)
    urls = guess_next_page_urls(url, max_pages=max_pages)
    log.info("tool_guess_pagination | generated=%d", len(urls))
    return {"urls": urls}


@app.tool()
async def tool_gather_related_links(markdown: str, base_url: str, patterns: list[str] | None = None, limit: int = 50) -> Dict[str, Any]:
    """Extract and filter related links from a page markdown."""
    log.info("tool_gather_related_links | base=%s limit=%d patterns=%s", base_url, limit, patterns)
    hrefs = await gather_related_links(markdown, base_url, patterns=patterns, limit=limit)
    log.info("tool_gather_related_links | links=%d", len(hrefs))
    return {"links": hrefs}


if __name__ == "__main__":
    # When run as a script, serve the MCP server on stdio
    app.run()
