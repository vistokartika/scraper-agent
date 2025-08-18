from __future__ import annotations

import json
import os
from typing import Any, Optional
import logging
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, LLMConfig
from crawl4ai import LLMExtractionStrategy
from scraper_agent.types import ExtractResult

load_dotenv(override=True)
logger = logging.getLogger("scraper_agent.extract")


def make_model(name: str, schema: dict[str, str], base=BaseModel) -> type[BaseModel]:
    TYPE_MAP = {"str": str, "int": int, "float": float, "bool": bool}
    fields = {}
    for field_name, type_name in schema.items():
        py_type = TYPE_MAP.get(type_name)
        if py_type is None:
            raise ValueError(f"Unknown type '{type_name}' for field '{field_name}'")
        fields[field_name] = (py_type, ...)
    return create_model(name, __base__=base, **fields)


async def extract_with_llm(markdown: str, query: str, schema: dict, context: Optional[str] = None) -> ExtractResult | None:
    provider = os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini")
    api_token = os.getenv("OPENAI_API_KEY")

    llm_cfg = LLMConfig(provider=provider, api_token=api_token)

    instruction = (
        "You are an expert web data modeler and extractor. You will be given a webpage content in a markdown format.\n\n"
        "Context:\n"
        f"{context or ''}\n\n"
        "Instruction:\n"
        "1) Analyze the webpage content and understand what the site/page is about by considering the given context as well."
        "2) You may need to prepare and design a Pydantic model that best captures the items needed to answer the user's request."
        "3) Extract data into an array of objects matching your model."
        "Return format: JSON array of objects.\n"
        "Rules: Output valid JSON only. Do not include markdown fences.\n"
        f"User request: {query}"
    )

    Schema = make_model("Schema", schema)
    strat = LLMExtractionStrategy(
        llm_config=llm_cfg,
        schema=Schema.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        input_format="markdown",
        apply_chunking=True,
        chunk_token_threshold=1200,
        overlap_rate=0.05,
        extra_args={"temperature": 0.0, "max_tokens": 1500},
        verbose=False,
    )

    config = CrawlerRunConfig(extraction_strategy=strat, cache_mode=CacheMode.BYPASS)
    try:
        # On Windows under some servers (e.g., Uvicorn), the default loop policy may be WindowsSelectorEventLoopPolicy,
        # which does not support asyncio subprocess used by Playwright. Run crawler in a separate thread with Proactor.
        if os.name == "nt":
            def _worker() -> Any:
                try:
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
                except Exception:
                    pass

                async def _inner() -> Any:
                    async with AsyncWebCrawler() as crawler:
                        return await crawler.arun(url=f"raw:{markdown}", config=config)

                return asyncio.run(_inner())

            result = await asyncio.to_thread(_worker)
        else:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=f"raw:{markdown}", config=config)

        if getattr(result, "success", False) and result.extracted_content:
            payload = result.extracted_content
            try:
                parsed: Any = json.loads(payload)
            except json.JSONDecodeError:
                parsed = payload
            usage = getattr(strat, "total_usage", None)
            return ExtractResult(data=parsed, raw=payload, model=provider, usage=usage)
    except Exception as e:
        logger.exception("extract_with_llm failed: %s", e)
    return None
