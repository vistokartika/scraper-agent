from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

# Map string names to Python types you want to allow
TYPE_MAP: Dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

load_dotenv(override=True)


async def generate_schema(markdown: str, query: str) -> dict[str, str]:
    """Use an LLM to propose a minimal flat schema mapping field->(str|int|float|bool)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    provider = os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini")
    model = provider.split("/", 1)[1] if "/" in provider else provider

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))

    truncated_md = markdown[:20000] if markdown else ""

    system_prompt = (
        "You are a senior data modeler. Given webpage markdown and a user request, "
        "design the smallest flat JSON schema that answers the request and enables consistent item matching. "
        "Include at least one identifier field if appropriate (e.g., name, title, label, company, organization), "
        "and keep keys concise. Return a single JSON object mapping field names to one of: 'str', 'int', 'float', 'bool'. "
        "No explanations; JSON only."
    )
    user_prompt = (
        f"User request:\n{query}\n\n"
        f"Webpage markdown:\n{truncated_md}\n\n"
        "Return JSON only, e.g. {\"role\": \"str\", \"location\": \"str\"}."
    )

    async def _call_openai() -> str:
        def _run():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
            )

        resp = await asyncio.to_thread(_run)
        return (resp.choices[0].message.content or "").strip()

    def _parse_schema(text: str) -> dict[str, str]:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        return {}

    content = await _call_openai()
    raw_schema = _parse_schema(content)

    allowed = set(TYPE_MAP.keys())
    schema: dict[str, str] = {}
    for k, v in (raw_schema or {}).items():
        if not isinstance(k, str):
            continue
        v_str = v if isinstance(v, str) else "str"
        schema[k.strip()] = v_str if v_str in allowed else "str"

    # Ensure we have at least one identifier-like field for linking items (general case)
    ident_candidates = ["name", "title", "label", "company", "organization"]
    if not any(ic in schema for ic in ident_candidates):
        schema["name"] = "str"

    if not schema:
        return {"value": "str"}

    return schema
