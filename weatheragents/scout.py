"""Agent 01 - Market Scout.

Pulls every open daily-temperature event from Polymarket's public Gamma API,
parses each into a Contract (semantics + pricing), and drops ones that are
illiquid or too close to resolution to act on. No LLM and no credentials.

    python -m weatheragents.scout --min-liquidity 20000
"""

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from weatheragents.schemas import (
    Bracket,
    BracketPrice,
    Contract,
    ContractPricing,
    ContractSemantics,
)

GAMMA_URL = "https://gamma-api.polymarket.com/events"
TAG = "daily-temperature"
PAGE_SIZE = 100

TITLE_RE = re.compile(r"^(Highest|Lowest) temperature in (.+?) on .+\?$")
OBS_DATE_RE = re.compile(r"on (\d{1,2} [A-Z][a-z]{2} '\d{2})")
BRACKET_RE = re.compile(r"^(-?\d+)(?:-(-?\d+))?°([CF])( or below| or higher)?$")
NOAA_SITE_RE = re.compile(r"site=(\w+)")
WUNDERGROUND_RE = re.compile(r"wunderground\.com/history/daily/\S*/([A-Z]{4})\b")
STATION_NAME_RE = re.compile(r"recorded (?:by NOAA )?at the (.+?) Station")
URL_RE = re.compile(r"https?://\S+")


class ParseError(ValueError):
    pass


def fetch_events(client: httpx.Client) -> list[dict]:
    events, offset = [], 0
    while True:
        resp = client.get(
            GAMMA_URL,
            params={"tag_slug": TAG, "closed": "false", "limit": PAGE_SIZE, "offset": offset},
        )
        resp.raise_for_status()
        page = resp.json()
        events.extend(page)
        if len(page) < PAGE_SIZE:
            return events
        offset += PAGE_SIZE


def parse_bracket(label: str) -> tuple[Bracket, str]:
    m = BRACKET_RE.match(label.strip())
    if not m:
        raise ParseError(f"unrecognised bracket {label!r}")
    first, second, unit, open_end = m.groups()
    low, high = int(first), int(second) if second else int(first)
    if open_end == " or below":
        low = None
    elif open_end == " or higher":
        high = None
    return Bracket(label=label, low=low, high=high), unit


def parse_station(event: dict) -> tuple[str | None, str | None]:
    text = f"{event.get('resolutionSource') or ''}\n{event.get('description') or ''}"
    if "Hong Kong Observatory" in text:
        return "Hong Kong Observatory", "HKO"
    code = None
    if m := NOAA_SITE_RE.search(text):
        code = m.group(1).upper()
    elif m := WUNDERGROUND_RE.search(text):
        code = m.group(1)
    name = m.group(1) if (m := STATION_NAME_RE.search(text)) else None
    return name, code


def parse_observation_date(event: dict, resolves_at: datetime) -> date:
    if m := OBS_DATE_RE.search(event.get("description") or ""):
        return datetime.strptime(m.group(1), "%d %b '%y").date()
    return resolves_at.date()


def _float(value) -> float | None:
    return None if value in (None, "") else float(value)


def parse_event(event: dict) -> Contract:
    title = event.get("title", "")
    m = TITLE_RE.match(title)
    if not m:
        raise ParseError(f"unrecognised title {title!r}")
    kind, city = m.group(1).lower(), m.group(2)

    resolves_at = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00"))
    description = event.get("description") or ""
    station_name, station_code = parse_station(event)

    brackets, prices, units = [], [], set()
    for market in event.get("markets", []):
        if market.get("closed") or not market.get("active", True):
            continue
        bracket, unit = parse_bracket(market.get("groupItemTitle", ""))
        units.add(unit)
        token_ids = json.loads(market.get("clobTokenIds") or "[]")
        outcomes = json.loads(market.get("outcomes") or "[]")
        if outcomes != ["Yes", "No"]:
            raise ParseError(f"unexpected outcomes {outcomes} in {title!r}")
        if not token_ids:
            raise ParseError(f"no order-book tokens (stale market) in {title!r}")
        brackets.append(bracket)
        prices.append(
            BracketPrice(
                label=bracket.label,
                yes_token_id=token_ids[0],
                best_bid=_float(market.get("bestBid")),
                best_ask=_float(market.get("bestAsk")),
                last_trade=_float(market.get("lastTradePrice")),
                liquidity=_float(market.get("liquidityNum")) or 0.0,
            )
        )
    if not brackets:
        raise ParseError(f"no open brackets in {title!r}")
    if len(units) != 1:
        raise ParseError(f"mixed units {units} in {title!r}")

    order = sorted(range(len(brackets)), key=lambda i: brackets[i].low if brackets[i].low is not None else -10**6)
    event_id = str(event["id"])
    resolution_source = event.get("resolutionSource") or next(iter(URL_RE.findall(description)), None)

    return Contract(
        semantics=ContractSemantics(
            event_id=event_id,
            city=city,
            kind=kind,
            observation_date=parse_observation_date(event, resolves_at),
            unit=units.pop(),
            station_name=station_name,
            station_code=station_code,
            resolution_source=resolution_source,
            brackets=[brackets[i] for i in order],
            resolves_at=resolves_at,
            rules=description,
        ),
        pricing=ContractPricing(
            event_id=event_id,
            total_liquidity=_float(event.get("liquidity")) or 0.0,
            volume_24h=_float(event.get("volume24hr")) or 0.0,
            brackets=[prices[i] for i in order],
        ),
    )


def scout(
    events: list[dict],
    *,
    min_liquidity: float,
    min_hours_left: float,
    now: datetime | None = None,
) -> tuple[list[Contract], list[str]]:
    """Parse and filter events. Returns (contracts, skipped-reasons)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=min_hours_left)
    contracts, skipped = [], []
    for event in events:
        try:
            contract = parse_event(event)
        except (ParseError, KeyError, ValueError) as exc:
            skipped.append(f"parse: {exc}")
            continue
        if contract.semantics.resolves_at <= cutoff:
            skipped.append(f"too close to resolution: {event.get('title')}")
        elif contract.pricing.total_liquidity < min_liquidity:
            skipped.append(f"illiquid: {event.get('title')}")
        elif contract.semantics.station_code is None:
            skipped.append(f"no station code: {event.get('title')}")
        else:
            contracts.append(contract)
    contracts.sort(key=lambda c: -c.pricing.total_liquidity)
    return contracts, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Market Scout: list tradable Polymarket temperature contracts")
    parser.add_argument("--min-liquidity", type=float, default=20_000, help="USD, whole event (default 20000)")
    parser.add_argument("--min-hours", type=float, default=6, help="skip contracts resolving sooner (default 6)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output folder (default data/)")
    args = parser.parse_args(argv)

    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as client:
        events = fetch_events(client)

    contracts, skipped = scout(events, min_liquidity=args.min_liquidity, min_hours_left=args.min_hours)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = args.out / f"contract_list_{stamp}.json"
    out_file.write_text(
        json.dumps([c.model_dump(mode="json") for c in contracts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"{len(events)} open events -> {len(contracts)} tradable, {len(skipped)} skipped")
    parse_failures = [s for s in skipped if s.startswith("parse:")]
    for reason in parse_failures:
        print(f"  ! {reason}", file=sys.stderr)
    print(f"\n{'city':<18}{'kind':<9}{'date':<12}{'station':<9}{'brackets':>9}{'liquidity':>12}  resolves (UTC)")
    for c in contracts:
        s, p = c.semantics, c.pricing
        print(
            f"{s.city:<18}{s.kind:<9}{s.observation_date!s:<12}{s.station_code:<9}"
            f"{len(s.brackets):>9}{p.total_liquidity:>12,.0f}  {s.resolves_at:%m-%d %H:%M}"
        )
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
