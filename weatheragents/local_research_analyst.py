"""Agent 05 - Local Research Analyst.

Looks for the national met agency's own bulletin and reputable local news
about the station's city, in the local language, and summarizes what the
global models might miss.

Steps: the LLM writes a few search queries (local language + English), a
key-less web search (ddgs) runs them, and the LLM summarizes the snippets.
Prediction-market and betting domains are dropped from the results, and the
prompts carry semantics only, so no price can leak in.

Search uses Tavily when TAVILY_API_KEY is set (the artifact's choice), and falls
back to a key-less search library otherwise or when Tavily errors.
"""

import os
import re
from datetime import datetime
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict
from weatheragents.schemas import ContractSemantics, LocalContextReport

BLOCKED_DOMAINS = (
    "polymarket.com",
    "kalshi.com",
    "predictit.org",
    "manifold.markets",
    "metaculus.com",
    "betfair.com",
    "oddschecker.com",
    "marketmarket.io",
)

# Betting/market pages come in under domains no blocklist will ever cover, so
# results are also dropped on what they say.
MARKET_TEXT_RE = re.compile(
    r"polymarket|kalshi|prediction market|prediction page|betting|odds|wager|sportsbook|"
    r"market (?:resolves?|resolution|page)|will the highest temperature|will the lowest temperature|"
    r"buy (?:yes|no)|yes/no",
    re.I,
)

QUERY_SYSTEM = (
    "You write web-search queries to find a city's weather forecast from its national meteorological "
    "agency and local news. Write exactly 3 short queries: two in the country's own language (use the "
    "agency's real name, e.g. the local word for 'forecast' and the city's local name) and one in English. "
    "Do not mention betting, odds or markets."
)

SUMMARY_SYSTEM = (
    "You summarize web-search snippets about a city's weather forecast for a temperature forecaster. Use "
    "only what the snippets say; do not add facts. Say which snippets are about the right city and date and "
    "which are not. 'leans' is whether the snippets point to a warmer or cooler day than a typical model "
    "forecast ('neutral' if they match, 'no_information' if nothing is specific to that city and date). "
    "'sources' lists the page titles or domains you relied on."
)


class _QueriesOut(BaseModel):
    queries: list[str]


class _LocalOut(BaseModel):
    sources: list[str]
    summary: str
    leans: str  # warmer | cooler | neutral | no_information
    local_only_warnings: list[str]
    caveats: list[str]


def blocked(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


def looks_like_market(result: dict) -> bool:
    return blocked(result.get("href", "")) or bool(
        MARKET_TEXT_RE.search(f"{result.get('title', '')} {result.get('href', '')} {result.get('body', '')}")
    )


TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, *, api_key: str, max_results: int = 6) -> list[dict]:
    """Tavily basic search (1 credit). Betting domains are excluded server-side too."""
    resp = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "exclude_domains": list(BLOCKED_DOMAINS),
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [{"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")} for r in resp.json().get("results", [])]


def search(query: str, *, max_results: int = 6) -> list[dict]:
    """Tavily when TAVILY_API_KEY is set, else (or if Tavily fails) the key-less ddgs fallback.

    Returns raw [{title, href, body}]; research() filters market pages.
    """
    key = os.environ.get("TAVILY_API_KEY")
    if key:
        try:
            results = tavily_search(query, api_key=key, max_results=max_results)
            if results:
                return results
        except httpx.HTTPError:
            pass  # out of credits or a blip: fall back rather than lose the event
    return _ddgs_search(query, max_results=max_results)


def _ddgs_search(query: str, *, max_results: int = 6) -> list[dict]:
    from ddgs import DDGS

    return DDGS().text(query, max_results=max_results)


def query_prompt(s: ContractSemantics) -> str:
    what = "daily high" if s.kind == "highest" else "daily low"
    return (
        f"City: {s.city}. Station: {s.station_name or s.station_code} ({s.station_code}). "
        f"Date: {s.observation_date}. We want the forecast {what} for that date."
    )


def research(llm, s: ContractSemantics, *, search_fn=search) -> LocalContextReport:
    base = dict(event_id=s.event_id, station_code=s.station_code or "")

    def empty(reason: str) -> LocalContextReport:
        return LocalContextReport(
            **base, sources=[], summary="No local information found.", leans="no_information",
            local_only_warnings=[], caveats=[reason], generated_at=ict_now(),
        )

    queries = llm.structured(system=QUERY_SYSTEM, user=query_prompt(s), schema=_QueriesOut).queries[:3]
    hits, seen = [], set()
    for q in queries:
        try:
            for r in search_fn(q):
                if r["href"] not in seen and not looks_like_market(r):
                    seen.add(r["href"])
                    hits.append(r)
        except Exception:  # search is best-effort; the pipeline must not die on it
            continue
    if not hits:
        return empty("search returned nothing usable")

    snippets = "\n".join(f"- {r['title']} ({r['href']}): {r.get('body', '')}" for r in hits[:15])
    out = llm.structured(
        system=SUMMARY_SYSTEM,
        user=f"City: {s.city}, station {s.station_code}, date {s.observation_date}.\n\nSnippets:\n{snippets}",
        schema=_LocalOut,
    )
    leans = out.leans if out.leans in ("warmer", "cooler", "neutral", "no_information") else "no_information"
    return LocalContextReport(
        **base,
        sources=out.sources,
        summary=out.summary,
        leans=leans,
        local_only_warnings=out.local_only_warnings,
        caveats=out.caveats + ["based on search-result snippets, not full pages"],
        generated_at=ict_now(),
    )
