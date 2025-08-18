from __future__ import annotations

import asyncio
import json

from scraper_agent.web import fetch_markdown
from scraper_agent.schema import generate_schema
from scraper_agent.extract import extract_with_llm


async def demo():
    url = "https://medrecruit.medworld.com/jobs/list?location=New+South+Wales&page=1"
    query = "From the URL, get the role, location, and pay of jobs on page 1."

    markdown = await fetch_markdown(url)
    print("[markdown snippet]", markdown[:500], "...\n")

    schema = await generate_schema(markdown, query=query)
    print("[schema]", schema)

    result = await extract_with_llm(markdown, query=query, schema=schema, context=f"URL: {url}")

    payload = getattr(result, "data", None)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            pass
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(demo())
