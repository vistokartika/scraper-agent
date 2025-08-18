from __future__ import annotations

from typing import List, Tuple
from urllib.parse import urljoin

import zendriver as zd
from bs4 import BeautifulSoup
from markdownify import markdownify as md


async def fetch_markdown(url: str, timeout: float = 60.0, headless: bool = True) -> str:
    """Fetch a URL with zendriver and return markdown content.
    Removes script/style/noscript/template and JSON script tags.
    """
    browser = await zd.start(headless=headless)
    try:
        page = await browser.get(url)
        await page.wait_for_ready_state("complete", timeout=timeout)
        html = await page.get_content()

        soup = BeautifulSoup(html, "lxml")
        for t in soup.select('script, style, noscript, template, script[type="application/json"]'):
            t.decompose()

        main = soup.select_one("main") or soup.body

        markdown = md(
            str(main),
            heading_style="ATX",
            strip=["script", "style"],
        ).strip()
        return markdown
    finally:
        await browser.stop()


def extract_markdown_links(markdown: str, base_url: str | None = None) -> List[Tuple[str, str]]:
    """Extract links from markdown as (text, href). Optionally resolve relative links to base_url."""
    links: List[Tuple[str, str]] = []
    # Markdown links as [text](href)
    import re

    for m in re.finditer(r"\[([^\]]+)\]\(([^)\s]+)\)", markdown):
        text = m.group(1).strip()
        href = m.group(2).strip()
        if base_url:
            href = urljoin(base_url, href)
        links.append((text, href))
    return links
