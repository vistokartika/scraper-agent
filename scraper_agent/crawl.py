from __future__ import annotations

import re
from typing import AsyncGenerator, Iterable, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from scraper_agent.web import fetch_markdown, extract_markdown_links


def guess_next_page_urls(url: str, max_pages: int | None = None) -> List[str]:
    """Heuristic pagination by detecting a 'page' query param or trailing page number.
    If max_pages is given and url has page=1, returns [page=1..max_pages].
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if "page" in qs and qs["page"]:
        try:
            start = int(qs["page"][0])
        except ValueError:
            start = 1
        pages = max_pages or start
        urls: List[str] = []
        for p in range(start, pages + 1):
            new_qs = dict(qs)
            new_qs["page"] = [str(p)]
            query = urlencode({k: v[0] if isinstance(v, list) else v for k, v in new_qs.items()})
            urls.append(urlunparse(parsed._replace(query=query)))
        return urls

    # Fallback: detect trailing /page/1 or ?p=1 patterns could be added here.
    m = re.search(r"(\bpage=)(\d+)", url)
    if m:
        start = int(m.group(2))
        pages = max_pages or start
        return [re.sub(r"(\bpage=)(\d+)", rf"\g<1>{p}", url) for p in range(start, pages + 1)]

    return [url]


async def crawl_pages(urls: Iterable[str], timeout: float = 60.0) -> AsyncGenerator[tuple[str, str], None]:
    """Yield (url, markdown) for each URL."""
    for u in urls:
        md = await fetch_markdown(u, timeout=timeout)
        yield (u, md)


async def gather_related_links(markdown: str, base_url: str, patterns: Optional[list[str]] = None, limit: int = 50) -> List[str]:
    """From page markdown, return candidate links to follow, filtered by optional regex patterns."""
    pairs = extract_markdown_links(markdown, base_url)
    hrefs = [h for _, h in pairs]
    if patterns:
        compiled = [re.compile(p, re.I) for p in patterns]
        hrefs = [h for h in hrefs if any(c.search(h) for c in compiled)]
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for h in hrefs:
        if h not in seen:
            seen.add(h)
            out.append(h)
        if len(out) >= limit:
            break
    return out
