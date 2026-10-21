"""One command to refresh everything and read how the design is doing.

    python -m weatheragents.daily                 # new predictions, outcomes, then the full report
    python -m weatheragents.daily --report-only   # skip the live pass; just fetch outcomes and report

Steps, each failing on its own without stopping the rest:
  1  Scout             a fresh list of tradable events
  2  numeric pass      predictions from the model, the yardsticks and the other sources (logged)
  3  resolver          Polymarket's result for every event that has finished
  4  scoreboard        every source against the outcomes and against the market, plus the blend test
  5  market moves      when and how Polymarket prices move, and whether they move toward us
  6  agency bulletins  whether the national agencies know something our forecasts and the market do not
  7  hard days         which warning signs predict a big miss
  8  fresh runs        whether the newest model runs beat the one the ML models on

The whole report is also saved to data/reports/ (Thailand time).
"""

import argparse
import contextlib
import io
import sys
import traceback
from pathlib import Path

from weatheragents import agency_report, evaluate, fresh_run, hard_day, market_moves, paper_trader, pipeline, resolver, scout, trader
from weatheragents.timeutil import now as ict_now, stamp as ict_stamp

REPORT_DIR = Path("data") / "reports"


class _Capture(io.StringIO):
    def reconfigure(self, **kwargs) -> None:  # the tools set their own UTF-8 console mode; a buffer needs no such thing
        pass


def run_step(title: str, fn, argv: list[str]) -> str:
    buf = _Capture()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(argv)
    except SystemExit:
        pass
    except Exception:  # one step failing must not hide the others
        buf.write("\n! step failed:\n" + traceback.format_exc(limit=3))
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{buf.getvalue().rstrip()}\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Refresh predictions and outcomes, then print the full report")
    ap.add_argument("--report-only", action="store_true", help="skip Scout and the live pass")
    ap.add_argument("--limit", type=int, default=500, help="events in the live pass (default: all)")
    args = ap.parse_args(argv)

    parts = [f"WeatherAgents report, {ict_now():%Y-%m-%d %H:%M} Thailand time"]
    if not args.report_only:
        parts.append(run_step("1  SCOUT", scout.main, []).split("wrote")[0] + "(contract list saved)\n")
        parts.append(run_step("2  LIVE PASS (numeric: model, yardsticks, sources)", pipeline.main, ["--limit", str(args.limit)]))
    parts.append(run_step("3  OUTCOMES", resolver.main, []))
    parts.append(run_step("4  SCOREBOARD (lower is better; the market is the bar)", evaluate.main, ["--blend", "--by", "version"]))
    parts.append(run_step("5  MARKET MOVES", market_moves.main, []))
    parts.append(run_step("6  AGENCY BULLETINS vs OUR FORECASTS AND THE MARKET", agency_report.main, []))
    parts.append(run_step("7  HARD DAYS: WHICH WARNING SIGNS PREDICT A BIG MISS", hard_day.main, []))
    parts.append(run_step("8  FRESH FORECAST RUNS vs THE ONE WE MODEL ON", fresh_run.main, []))
    parts.append(run_step("9  PAPER TRADING (pretend dollars)", trader.main, ["--report"]))
    parts.append(run_step("10  PRE-REGISTERED TEST: SELL 12 HOURS AFTER ENTRY", paper_trader.main, []))

    report = "".join(parts)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"report_{ict_stamp()}.txt"
    out.write_text(report, encoding="utf-8")
    # the live pass prints one table per event; keep the terminal readable and leave the full text in the file
    shown = report
    if not args.report_only and "2  LIVE PASS" in report:
        a = report.index("2  LIVE PASS")
        b = report.index("3  OUTCOMES")
        live = report[a:b].splitlines()
        shown = report[:a] + "\n".join(live[:4] + [f"  ... {len(live)} lines of per-event tables in {out}"]) + "\n" + report[b - 80:]
    print(shown)
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
