"""Markdown tables for a finished (or partial) experiment directory, read from its JSONL traces.

    uv run python scripts/pilot_report.py runs/pilot_deepseek [--config configs/experiments/pilot_deepseek.yaml]
        [--headline-config configs/experiments/headline_deepseek.yaml]

Definitions used in the tables:
- graded: run_end with ``passed`` not null. ``passed`` already counts contaminated and cost-ceiling runs as failures.
- silent failure: the main agent called ``claim_done`` (reported the task done) and the run did not pass.
- a rejection is a ``check`` failure (L0-L3) on a sub-agent claim; with no injected faults every rejected
  claim is a natural claim, so rejections are an upper bound on false rejects until the claims are hand-labelled.
- spend: ``event_spend`` (the larger of price-table cost and OpenRouter's bill) over every billed event,
  in finished traces and in traces set aside after a failed attempt (``*.jsonl.aborted-<n>``).
- provider mix: OpenRouter picks the endpoint per call from the pinned set, so the served provider is a
  logged covariate. An arm is flagged unbalanced when its share of calls served by a provider differs
  from the pooled share by more than ``IMBALANCE_PP`` percentage points (arms with at least 20 calls).
  "$ at reference rates" re-prices every call's tokens at the first endpoint of the set, so arms can be
  compared on cost whatever the mix (cache lost when a call moves provider still shows in the tokens).
- worst-case projection: every call's tokens re-priced at the larger rate of each component over the set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcarry.providers.usage import Rates, TokenUsage  # noqa: E402
from evcarry.runs.budget import event_spend  # noqa: E402

ARM_ORDER = ["sa", "vanilla_mad", "self_verify", "typed", "evidence", "evidence_random", "evidence_risk"]
IMBALANCE_PP = 10.0
_BILLED = ("llm_call", "judge_call", "llm_error")


def _tokens(event: dict) -> TokenUsage:
    raw = event.get("tokens") or {}
    return TokenUsage(
        int(raw.get("input_tokens") or 0), int(raw.get("cache_hit_tokens") or 0),
        int(raw.get("cache_miss_tokens") or 0), int(raw.get("output_tokens") or 0),
    )


def _provider(event: dict) -> str:
    return str(event.get("response_provider") or event.get("served_provider") or "unobserved")


def _seconds(trace: list[dict]) -> float | None:
    from datetime import datetime

    start = next((e.get("ts") for e in trace if e.get("type") == "run_start"), None)
    end = next((e.get("ts") for e in trace if e.get("type") == "run_end"), None)
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _route_rates(attempts: dict[str, list[list[dict]]]) -> tuple[Rates, Rates, str] | None:
    """(reference rates: the set's first endpoint, worst-case rates, reference tag) for the runs' model entry."""
    from evcarry.providers.registry import load_models

    name = next((e.get("model") for traces in attempts.values() for t in traces for e in t if e.get("type") == "run_start"), None)
    config = load_models().get(str(name)) if name else None
    if config is None:
        return None
    price = config.price_table()
    tags = list((config.routing or {}).get("order") or (config.routing or {}).get("only") or [])
    reference, label = price.rates_for(tags[0].split("/")[0]) if tags else (price.worst_rates(), "table")
    return reference, price.worst_rates(), label


def _events(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _arm_of(events: list[dict], path: Path) -> str:
    start = next((event for event in events if event.get("type") == "run_start"), None)
    return str(start.get("arm")) if start else path.name.split("__")[1]


def _table(header: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
        return str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(part: float, whole: float) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else f"{part}/0"


def load(run_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[list[dict]]]]:
    finished: dict[str, list[dict]] = defaultdict(list)  # arm -> list of traces (event lists) with run_end
    attempts: dict[str, list[list[dict]]] = defaultdict(list)  # arm -> every trace, finished or not
    for path in sorted([*run_dir.glob("*.jsonl"), *run_dir.glob("*.jsonl.aborted-*")]):
        events = _events(path)
        if not events:
            continue
        arm = _arm_of(events, path)
        attempts[arm].append(events)
        if path.suffix == ".jsonl" and any(event.get("type") == "run_end" for event in events):
            finished[arm].append(events)
    return finished, attempts


def report(run_dir: Path, planned: dict[str, int] | None, headline_runs_per_arm: int | None, headline_cap: float | None) -> str:
    finished, attempts = load(run_dir)
    arms = [arm for arm in ARM_ORDER if arm in attempts or (planned and arm in planned)]
    out: list[str] = []

    # Runs.
    rows = []
    for arm in arms:
        ends = [trace[-1] if trace[-1].get("type") == "run_end" else next(e for e in trace if e.get("type") == "run_end") for trace in finished[arm]]
        statuses = Counter(str(end.get("status")) for end in ends)
        agent = Counter(str(end.get("agent_status")) for end in ends)
        unfinished = sum(1 for trace in attempts[arm] if not any(e.get("type") == "run_end" for e in trace))
        rows.append([
            arm, (planned or {}).get(arm, "?"), len(ends), unfinished,
            ", ".join(f"{key} {value}" for key, value in sorted(statuses.items())),
            ", ".join(f"{key} {value}" for key, value in sorted(agent.items())),
        ])
    out.append("### Runs\n\n" + _table(["arm", "planned", "finished (run_end)", "failed attempts (no run_end)", "status", "agent_status"], rows))

    # Success.
    rows = []
    for arm in arms:
        ends = [next(e for e in trace if e.get("type") == "run_end") for trace in finished[arm]]
        graded = [end for end in ends if end.get("passed") is not None]
        passed = sum(1 for end in graded if end.get("passed"))
        eval_passed = sum(1 for end in graded if end.get("eval_passed", end.get("passed")))
        claimed = [end for end in graded if end.get("claimed_done")]
        silent = [end for end in claimed if not end.get("passed")]
        failed = [end for end in graded if not end.get("passed")]
        rows.append([
            arm, _pct(passed, len(graded)), eval_passed, len(claimed), _pct(len(silent), len(graded)),
            _pct(len(silent), len(failed)),
        ])
    out.append(
        "### Task success and silent failures\n\n"
        + _table(["arm", "passed / graded", "grader passed", "claimed done", "silent failures / graded", "silent / failed"], rows)
    )

    # Per task.
    tasks = sorted({str(next(e for e in trace if e.get("type") == "run_end").get("task")) for arm in arms for trace in finished[arm]})
    grid: dict[tuple[str, str], str] = {}
    for arm in arms:
        for trace in finished[arm]:
            end = next(e for e in trace if e.get("type") == "run_end")
            mark = {True: "pass", False: "fail", None: "n/g"}[end.get("passed")]
            if end.get("status") == "cost_ceiling":
                mark = "ceiling"
            elif end.get("agent_status") == "max_turns_reached":
                mark += " (turns)"
            grid[(str(end.get("task")), arm)] = mark
    out.append("### Per task\n\n" + _table(["task", *arms], [[task, *(grid.get((task, arm), "-") for arm in arms)] for task in tasks]))

    # Audit and faults.
    rows, level_rows, injected_rows = [], [], []
    for arm in arms:
        returns = claims = sampled = 0
        actions: Counter = Counter()
        levels: Counter = Counter()
        injected: Counter = Counter()
        caught: Counter = Counter()
        for trace in attempts[arm]:
            injected_ids: dict[int, dict] = {}
            for event in trace:
                kind = event.get("type")
                if kind == "subagent_return":
                    returns += 1
                    claims += len(event.get("claim_ids") or [])
                elif kind == "audit":
                    sampled += len(event.get("sampled") or [])
                elif kind == "policy":
                    actions[str(event.get("action"))] += 1
                elif kind == "injection" and event.get("applied"):
                    injected[str(event.get("kind"))] += 1
                    injected_ids[int(event.get("return_index", -1))] = event
                elif kind == "check":
                    hit = injected_ids.get(int(event.get("return_index", -2)))
                    for failure in event.get("failures") or []:
                        level = str(failure.get("reason", "")).split(":", 1)[0]
                        if hit and failure.get("claim_id") == hit.get("claim_id"):
                            caught[(str(hit.get("kind")), level)] += 1
                        else:
                            levels[level] += 1
        if returns or arm.startswith("evidence") or arm == "typed":
            rows.append([arm, returns, claims, sampled, ", ".join(f"{k} {v}" for k, v in sorted(actions.items())) or "-"])
            level_rows.append([arm, *(levels.get(level, 0) for level in ("L0", "L1", "L2", "L3"))])
        for fault, count in sorted(injected.items()):
            injected_rows.append([arm, fault, count, *(caught.get((fault, level), 0) for level in ("L0", "L1", "L2", "L3"))])
    out.append(
        "### Sub-agent returns and audits\n\n"
        + _table(["arm", "returns", "claims", "claims audited (L2/L3)", "policy actions"], rows)
        + "\n\nRejections of natural (not injected) claims, by the level that failed (one claim can fail at several levels):\n\n"
        + _table(["arm", "L0", "L1", "L2", "L3"], level_rows)
        + "\n\nInjected faults and the level that caught them:\n\n"
        + (_table(["arm", "fault", "injected", "L0", "L1", "L2", "L3"], injected_rows) if injected_rows else "none injected (the experiment has no `inject` block)")
    )

    # Judge.
    rows = []
    for arm in arms:
        calls = [event for trace in attempts[arm] for event in trace if event.get("type") == "judge_call"]
        if not calls:
            continue
        verdicts = Counter(str(call.get("verdict") or call.get("source")) for call in calls)
        fallbacks = sum(1 for call in calls if call.get("source") == "lexical_fallback")
        cost = sum(event_spend(call) for call in calls)
        rows.append([arm, len(calls), fallbacks, ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())), cost])
    out.append("### L3 judge\n\n" + (_table(["arm", "judge calls", "lexical fallbacks", "verdicts", "judge $"], rows) if rows else "no judge calls"))

    # Tokens and dollars.
    rows = []
    totals = Counter()
    per_run: dict[str, list[float]] = {}
    for arm in arms:
        agg = Counter()
        spends = []
        for trace in attempts[arm]:
            run_spend = 0.0
            for event in trace:
                run_spend += event_spend(event)
                if event.get("type") in ("llm_call", "judge_call", "llm_error"):
                    tokens = event.get("tokens") or {}
                    role = "judge" if event.get("type") == "judge_call" or event.get("role") == "judge" else "agent"
                    agg[f"{role}_calls"] += 1
                    agg["input"] += int(tokens.get("input_tokens") or 0)
                    agg["hit"] += int(tokens.get("cache_hit_tokens") or 0)
                    agg["output"] += int(tokens.get("output_tokens") or 0)
                    agg[f"{role}_usd"] += event_spend(event)
            if any(e.get("type") == "run_end" for e in trace):
                spends.append(run_spend)
            agg["spend"] += run_spend
        per_run[arm] = spends
        totals.update(agg)
        mean = statistics.mean(spends) if spends else 0.0
        rows.append([
            arm, agg["agent_calls"], agg["judge_calls"], agg["input"], f"{100 * agg['hit'] / agg['input']:.0f}%" if agg["input"] else "-",
            agg["output"], agg["agent_usd"], agg["judge_usd"], agg["spend"], mean, max(spends) if spends else 0.0,
        ])
    rows.append([
        "**total**", totals["agent_calls"], totals["judge_calls"], totals["input"],
        f"{100 * totals['hit'] / totals['input']:.0f}%" if totals["input"] else "-", totals["output"], totals["agent_usd"],
        totals["judge_usd"], totals["spend"], "", "",
    ])
    out.append(
        "### Tokens and dollars (every attempt, finished or not)\n\n"
        + _table(["arm", "agent calls", "judge calls", "input tokens", "cache hits", "output tokens", "agent $", "judge $", "spend $", "mean $/finished run", "max $/run"], rows)
    )

    # Provider pin and temperature.
    checks: Counter = Counter()
    providers: Counter = Counter()
    temperatures: Counter = Counter()
    for arm in arms:
        for trace in attempts[arm]:
            for event in trace:
                if event.get("type") not in ("llm_call", "judge_call"):
                    continue
                check = event.get("provider_check") or {}
                checks[str(check.get("ok"))] += 1
                providers[str(event.get("response_provider") or event.get("served_provider"))] += 1
                temperatures[str(event.get("temperature"))] += 1
    mismatches = sum(1 for arm in arms for trace in attempts[arm] for e in trace if e.get("type") == "provider_mismatch")
    out.append(
        "### Served provider and temperature (llm_call + judge_call)\n\n"
        + f"- provider_check.ok: {dict(checks)}\n- provider named in the response: {dict(providers)}\n"
        + f"- provider_mismatch events: {mismatches}\n- temperature sent: {dict(temperatures)}"
    )

    # Provider mix per arm: every answered (billed) call, finished runs and failed attempts.
    rates = _route_rates(attempts)
    mix: dict[str, Counter] = {arm: Counter() for arm in arms}
    reference_usd: Counter = Counter()
    for arm in arms:
        for trace in attempts[arm]:
            for event in trace:
                if event.get("type") in _BILLED and event.get("tokens"):
                    mix[arm][_provider(event)] += 1
                    if rates:
                        reference_usd[arm] += rates[0].usd(_tokens(event))
    pooled: Counter = sum(mix.values(), Counter())
    names = sorted(pooled)
    flagged: list[str] = []
    rows = []
    for arm in arms:
        total = sum(mix[arm].values())
        shares = {name: 100 * mix[arm][name] / total for name in names} if total else {}
        gaps = [abs(shares[name] - 100 * pooled[name] / sum(pooled.values())) for name in names] if total else []
        unbalanced = total >= 20 and max(gaps, default=0) > IMBALANCE_PP
        if unbalanced:
            flagged.append(arm)
        rows.append([
            arm, total, *(f"{mix[arm][name]} ({shares.get(name, 0):.0f}%)" for name in names),
            reference_usd[arm] if rates else "-", "**unbalanced**" if unbalanced else ("ok" if total >= 20 else "too few calls"),
        ])
    pooled_total = sum(pooled.values())
    rows.append(["**pooled**", pooled_total, *(f"{pooled[name]} ({100 * pooled[name] / pooled_total:.0f}%)" for name in names), sum(reference_usd.values()) if rates else "-", ""])
    reference_label = f"$ at {rates[2]} rates" if rates else "$ at reference rates"
    out.append(
        "### Provider mix per arm (answered calls, every attempt)\n\n"
        + _table(["arm", "calls", *names, reference_label, f"balance (>{IMBALANCE_PP:.0f} pp from pooled)"], rows)
        + ("\n\n**Unbalanced:** " + ", ".join(flagged) + ". Compare these arms' cost at the reference rates, and check success by provider." if flagged else "")
    )

    # Wall-clock per finished run.
    rows = []
    for arm in arms:
        seconds = [s for trace in finished[arm] if (s := _seconds(trace)) is not None]
        retried = sum(1 for trace in attempts[arm] for e in trace if e.get("type") in _BILLED and int(e.get("attempts") or 1) > 1)
        rows.append([
            arm, len(seconds), statistics.median(seconds) / 60 if seconds else "-", statistics.mean(seconds) / 60 if seconds else "-",
            max(seconds) / 60 if seconds else "-", retried,
        ])
    out.append(
        "### Wall-clock per finished run (minutes, run_start to run_end)\n\n"
        + _table(["arm", "runs", "median", "mean", "max", "calls retried after 429/5xx"], rows)
    )

    # Contaminated, ceiling, errors.
    contaminated, ceilings = [], Counter()
    errors: Counter = Counter()
    for arm in arms:
        for trace in attempts[arm]:
            for event in trace:
                kind = event.get("type")
                if kind == "run_end":
                    if event.get("contaminated"):
                        hits = "; ".join(f"{h.get('tool')} ({h.get('markers')})" for h in event.get("ground_truth_suspects") or [])
                        hits += "".join(f"; {h.get('tool')} (obtained)" for h in event.get("ground_truth_obtained") or [])
                        # "attempted" or "obtained" from 2026-10-25 (env.contamination); older traces have no tier.
                        contaminated.append([event.get("task"), arm, event.get("seed"), event.get("eval_passed"),
                                             event.get("contamination_tier") or "-", hits])
                    if event.get("cost_ceiling"):
                        ceilings[arm] += 1
                elif kind == "llm_error":
                    errors[(arm, f"llm_error {event.get('role')} {event.get('error_type')}")] += 1
                elif kind == "subagent_error":
                    errors[(arm, f"subagent_error {event.get('subagent')} {event.get('error_type')}")] += 1
                elif kind == "run_error":
                    errors[(arm, f"run_error {event.get('error_type')}")] += 1
                elif kind == "environment" and event.get("error"):
                    errors[(arm, "env_error")] += 1
    out.append(
        "### Contaminated runs (counted as failures)\n\n"
        + (_table(["task", "arm", "seed", "grader passed", "tier", "flagged calls"], contaminated) if contaminated else "none")
        + "\n\n### Cost-ceiling aborts\n\n"
        + (_table(["arm", "runs stopped at $ ceiling"], [[arm, ceilings[arm]] for arm in arms]) if ceilings else "none")
        + "\n\n### Errors\n\n"
        + (_table(["arm", "error", "count"], [[arm, kind, count] for (arm, kind), count in sorted(errors.items())]) if errors else "none")
    )

    # Projection.
    if headline_runs_per_arm:
        rows = []
        projected = worst_total = 0.0
        for arm in arms:
            mean = statistics.mean(per_run[arm]) if per_run[arm] else 0.0
            # Worst case: every call of a finished run re-priced at the set's larger rate of each component.
            worst_runs = [
                sum(max(rates[1].usd(_tokens(e)), event_spend(e)) for e in trace if e.get("type") in _BILLED)
                for trace in finished[arm]
            ] if rates else []
            worst = statistics.mean(worst_runs) if worst_runs else mean
            projected += mean * headline_runs_per_arm
            worst_total += worst * headline_runs_per_arm
            rows.append([arm, mean, worst, headline_runs_per_arm, mean * headline_runs_per_arm, worst * headline_runs_per_arm])
        rows.append(["**total**", "", "", headline_runs_per_arm * len(arms), projected, worst_total])
        cap = f" against the ${headline_cap:.0f} cap" if headline_cap else ""
        out.append(
            f"### Projected headline cost{cap}\n\n"
            + _table(["arm", "mean $/run (pilot)", "worst-case $/run", "headline runs", "projected $", "worst-case $"], rows)
            + ("\n\nWorst case: every call priced at " + f"${rates[1].input_cache_miss}/${rates[1].input_cache_hit}/${rates[1].output} per M "
               "(miss/hit/output), the larger rate of each component over the endpoint set." if rates else "")
        )
    return "\n\n".join(out)


def _planned(config_path: str | None) -> dict[str, int] | None:
    if not config_path:
        return None
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import grid

    counts: Counter = Counter(arm for _task, arm, _seed in grid(load_experiment(config_path)))
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--headline-config")
    args = parser.parse_args()
    runs_per_arm = cap = None
    if args.headline_config:
        from evcarry.runs.config import load_experiment

        headline = _planned(args.headline_config) or {}
        runs_per_arm = max(headline.values()) if headline else None
        cap = load_experiment(args.headline_config).budget.cap_usd
    print(report(args.run_dir, _planned(args.config), runs_per_arm, cap))


if __name__ == "__main__":
    main()
