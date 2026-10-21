"""Fetches outcomes for logged events and appends them to data/log/outcomes.jsonl.

    python -m weatheragents.resolver

Ground truth is Polymarket's own resolution, since that is what pays. A
winning bracket's YES price snaps to ~1.0 and the rest to ~0.0.

  final        event is closed and exactly one bracket is >= 0.99
  provisional  past its end time, exactly one bracket >= 0.99, but not closed
               yet (Polymarket's closed flag lags its prices by hours)

Provisional outcomes are re-checked on the next run and upgraded to final.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from weatheragents import prediction_log as plog
from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict

GAMMA_EVENT_URL = "https://gamma-api.polymarket.com/events/{event_id}"
WIN = 0.99
LOSE = 0.01


def winning_bracket(event: dict) -> str | None:
    """Label of the single resolved-YES bracket, or None if the event isn't decided."""
    winners, losers, total = [], 0, 0
    for m in event.get("markets", []):
        total += 1
        try:
            yes = float(json.loads(m.get("outcomePrices") or "[]")[0])
        except (ValueError, IndexError, TypeError):
            return None
        if yes >= WIN:
            winners.append(m.get("groupItemTitle"))
        elif yes <= LOSE:
            losers += 1
    if len(winners) == 1 and losers == total - 1:
        return winners[0]
    return None


def classify(event: dict, now: datetime) -> tuple[str, str | None]:
    label = winning_bracket(event)
    if label is None:
        return "open", None
    if event.get("closed"):
        return "final", label
    end = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00"))
    return ("provisional", label) if now >= end else ("open", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record outcomes for logged events")
    parser.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    args = parser.parse_args(argv)

    preds = plog.read(args.log_dir / plog.PREDICTIONS)
    outcomes_path = args.log_dir / plog.OUTCOMES
    have = {}
    for o in plog.read(outcomes_path):
        have[o["event_id"]] = o  # later lines win, so an upgrade overrides a provisional
    now = datetime.now(timezone.utc)  # a real clock for comparisons; the stored stamp below is ICT

    todo = {}
    for p in preds:
        if have.get(p["event_id"], {}).get("status") != "final" and datetime.fromisoformat(p["resolves_at"]) <= now:
            todo[p["event_id"]] = p
    new = skipped = 0
    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for eid, p in todo.items():
            try:
                resp = client.get(GAMMA_EVENT_URL.format(event_id=eid))
                resp.raise_for_status()
                status, label = classify(resp.json(), now)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                print(f"  ! {p['city']} {p['observation_date']}: {exc}", file=sys.stderr)
                continue
            if status == "open":
                skipped += 1
                continue
            if label not in p["brackets"]:
                print(f"  ! {p['city']} {p['observation_date']}: winner {label!r} not in logged brackets", file=sys.stderr)
                continue
            if have.get(eid, {}).get("status") == status and have[eid]["resolved_label"] == label:
                continue
            plog.append(
                outcomes_path,
                {
                    "event_id": eid,
                    "city": p["city"],
                    "observation_date": p["observation_date"],
                    "resolved_label": label,
                    "status": status,
                    "source": "polymarket_gamma",
                    "fetched_at": ict_now().isoformat(),
                },
            )
            new += 1
            print(f"  {p['city']:<18}{p['observation_date']}  -> {label}  ({status})")
    print(f"{len(preds)} predictions logged; {len(todo)} past resolution time; {new} outcomes recorded, {skipped} not decided yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
