"""Regulation lookup through Tavily, restricted to osha.gov. Never returns anything the search did not."""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlparse

DOMAIN = "osha.gov"
MAX_RESULTS = 3
SNIPPET_CHARS = 300


class RegulationLookup:
    """Looks up the regulation behind a zone rule.

    ``factory`` builds the Tavily client (injected in tests); by default it is
    ``tavily.TavilyClient(api_key=...)``.
    """

    def __init__(self, api_key: str | None = None, mock: bool = False, factory: Callable[[], Any] | None = None):
        self.api_key = api_key
        self.mock = mock
        self.factory = factory

    def lookup(self, rule: dict[str, Any]) -> dict[str, Any]:
        rule_id = rule["id"]
        if self.mock:
            return {"rule_id": rule_id, "results": [], "note": "mock mode: no regulation lookup"}
        if not self.api_key and self.factory is None:
            return {"error": "TAVILY_API_KEY not set", "results": []}
        query = rule.get("regulation_query")
        if not query:
            return {"error": f"rule {rule_id} has no regulation_query", "results": []}
        client = self.factory() if self.factory else self._default_client()
        raw = client.search(query, max_results=MAX_RESULTS, include_domains=[DOMAIN])
        results = []
        for r in (raw or {}).get("results") or []:
            url = str(r.get("url") or "")
            host = (urlparse(url).hostname or "").lower()
            if not (host == DOMAIN or host.endswith("." + DOMAIN)):
                continue  # whatever the API returned, only osha.gov links are passed on
            results.append(
                {"title": str(r.get("title") or url), "url": url, "snippet": str(r.get("content") or "")[:SNIPPET_CHARS]}
            )
        return {"rule_id": rule_id, "results": results[:MAX_RESULTS]}

    def _default_client(self):
        from tavily import TavilyClient

        return TavilyClient(api_key=self.api_key)
