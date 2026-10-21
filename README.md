# WeatherAgents

A forecasting and evaluation study on Polymarket's daily-temperature markets: can a learned weather forecast find anything the market's prices do not already contain, and can a rule turn that into money after the spread and the fees?

The point of the repo is the measurement machinery: every component has to earn its place on events it has never seen, and the pipeline was changed only when the measurements said to. It is research in progress.

No real order is ever placed. Everything is paper trading in pretend dollars.

## The evaluation, in one page

Most forecasting side projects stop at "the model is accurate." This one asks "accurate compared to what, on which events, and could that gap be luck?"

| Question | How it is answered |
|---|---|
| Is the forecast any good? | Whole-degree log loss, ranked probability score and Brier score on every resolved event, plus calibration tables. CRPS is used when testing the model on held-out dates. |
| Good relative to the bar that matters? | The market's own mid prices are scored as a source. Every comparison is on the **same events**, as a paired difference with a 95% range, so a gap is never an artifact of different samples. |
| Does the forecast hold information the market lacks? | A blend test: mix our curve into the market's at every weight. If no weight below 1 beats the market alone, there is nothing to add. |
| Did a change help, or did it just look better? | Training is split by time (train early, test on later dates). Input groups are added one at a time and kept only if they improve scores on dates the model never saw. Ideas that failed were dropped: dust and haze, neighbouring-station errors, run-to-run drift, the LLM layer. |
| Are the targets right? | Settlement was reproduced: building the day's high from every station report in the local calendar day matched Polymarket's paid bracket in 75 of 75 Celsius events. |
| Is there leakage? | A price wall in the code: no analyst or model ever sees a price. The evaluator uses only the latest prediction logged before resolution. Training data end dates are pinned. |
| Do versions regress? | Every pipeline change gets a version stamp. Versions are scored apart and compared on shared events. |
| Could a trading rule be a fluke? | Rules were written down before the data (`PREREGISTERED.md`), with a pass condition (at least 100 trades, return positive with the low end of its range above zero) and a fail condition (dropped, not re-tuned). |
| Is the money result real? | The paper Trader pays the ask, sells at the bid, pays Polymarket's actual taker fee (read from each market's fee schedule) and runs under a plain-code risk gate. Six accounts trade the same quotes, so each rule is compared against a control on the same events. A verdict line says "too early", "cannot tell from luck" or "losing beyond chance" from the range, not the dollars. |
| Where does the model fail? | A hard-day detector ranks warning signs against big misses (the model's own stated spread predicts them; most other signs are near chance), and a post-mortem tool explains single misses. |
| Does the market move toward us? | The market monitor snapshots prices every 30 minutes (every 5 for new markets) and asks whether prices later drift toward our forecast. |

## Why the LLM agents are switched off

The first design was a committee: a local-research agent that searched the web for news, a warmer-case and a cooler-case researcher that debated each event, and a Research Manager that adjusted the statistical forecast. It was built, run live, and scored with the same evaluator as everything else.

It did not earn its place:

- **Letting the LLM write the probabilities (version 1)** scored no better than a plain statistical baseline.
- **Letting it only nudge the baseline (version 2)** scored worse than the baseline it was adjusting, by about 0.05 log loss on 67 events. The extra reasoning added noise, not information.
- **The local-news signal** pointed the wrong way on 91 events: the researchers' "warmer" and "cooler" leans did not predict the actual miss.
- **It cost real calls.** Every event needed several model calls and web searches, for no measurable gain.

So the layer is off by default and the code stays, for two reasons. The measurements are part of the record, and the rule is simple: a component stays only if it improves scores on events it has never seen. To run it again, set the flags in `pipeline.py` (`SAMEDAY_ON`) and `open_watch.py` (`DEFAULT_LLM_BUDGET`), or use `--mode full` on `weatheragents.pipeline`. The same-day model went the same way: it beat a hand-built projector offline, but nothing live used it, so it was switched off rather than carried.

What replaced it is a pipeline of small, testable numeric parts. The only learned component in the live path is the gradient-boosted correction in `ml_analyst`, and the structured agency bulletins are collected as a cleaner way to test whether outside human forecasts add anything.

## How it is built

```
scout -> analysts (ensemble, nowcast, climatology, fresh run) -> ML analyst + baseline curves
                                   | price wall
                    live quotes -> Trader (6 paper accounts + risk gate)
      prediction log -> resolver -> evaluator -> scorecard
```

- `weatheragents/` is the package. The main parts are `scout`, the analyst modules, `ml_data` / `ml_train` / `ml_analyst` (a gradient-boosted correction to a four-model forecast, with a learned spread), `baseline` (the statistical yardstick), `evaluate` (scoring), `market_monitor` and `open_watch` (price collection), `trader` and `trader_report` (paper trading), and `paper_trader` (replay of logged predictions).
- Retired but kept: the LLM research layer and the same-day models, switched off by flags with the reasons in the code.
- Tests are in `tests/` (127 passing).

## Running it

```
uv sync
python -m weatheragents.daily               # scout, live pass, results, scoreboard, reports
python -m weatheragents.daily --report-only # just the reports
python -m weatheragents.evaluate --blend    # scoreboard with the blend test
python -m weatheragents.trader --watch 5    # paper accounts
python -m weatheragents.trader --report     # dollar scorecard
```

Needs `DEEPSEEK_API_KEY` and optionally `TAVILY_API_KEY` in the environment only if the LLM layer is turned back on. All times are Thailand time (UTC+7).

## What I would do next

Collect more events caught at the moment a market opens, since that is where the forecast looked least bad, and let the two written-down trading hypotheses run to their sample sizes before concluding anything.
