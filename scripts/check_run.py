"""Health check of a run from its telemetry: the gate between the 50-image probe and the full dev run.

    python scripts/check_run.py <outputs>/<run>          # exit code 1 if a threshold fails

Not checked here: the Colab compute units left (read them in the Colab "Resources" panel and compare
with the projected hours x the GPU's hourly rate, keeping a 15 % margin).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_READER_INVALID = 0.05   # first pass, per reader
MAX_CRITIC_FAIL = 0.10      # images without an LLM review
MAX_UNANSWERED = 0.02       # questions that end with no vote


def report(events: list[dict], project: int = 835, expert: str = "dinov3") -> dict:
    records: dict[tuple[str, str], dict] = {}  # latest record per (agent, image): resumed runs append
    done: dict[str, bool] = {}
    timed: dict[str, dict] = {}  # last interval of each pass that did real work (cache-only reruns are skipped)
    open_pass: dict | None = None
    final = None
    for e in events:
        kind = e.get("event")
        if kind == "pass_start":
            open_pass = {"start": e, "records": []}
            done[e["step"]] = False
        elif kind == "pass_end":
            done[e["step"]] = True
            if open_pass and open_pass["records"]:
                timed[e["step"]] = {**open_pass, "end": e}
            open_pass = None
        elif kind == "final_metrics":
            final = e
        elif "agent" in e:
            records[e["agent"], e["image_id"]] = e
            if open_pass:
                open_pass["records"].append(e)
    # hours for `project` images = model start-up of each pass + work per image of the split
    seconds, durations = 0.0, {}
    n_split = max([e["to_run"] + e["cached"] for e in events if e.get("event") == "pass_start"] or [1]) or 1
    for step, p in timed.items():
        rs, start = p["records"], p["start"]
        dur = p["end"]["ts"] - start["ts"]
        work = rs[-1]["ts"] - (rs[0]["ts"] - rs[0].get("latency_ms", 0) / 1000)
        applies = (start["to_run"] + start["cached"]) / n_split
        seconds += (dur - work) + project * applies * work / len(rs)
        durations[step] = dur / 60

    of = lambda agent: [r for (a, _), r in records.items() if a == agent]  # noqa: E731
    n_images = len(of("aggregator"))
    failures, stats = [], {"images": n_images}
    failures += [f"pass {step} not finished" for step, ok in done.items() if not ok]
    if final is None:
        failures.append("no final_metrics event")
    for reader in ("reader_a", "reader_b"):
        calls = [c for r in of(reader) for c in r["calls"]]
        bad = sum("error" in c for c in calls)
        stats[reader] = f"{bad}/{len(calls)} invalid"
        if not calls or bad / len(calls) >= MAX_READER_INVALID:
            failures.append(f"{reader}: {bad}/{len(calls)} invalid answers")
    reviewed = sum("llm" in r for r in of("critic"))
    stats["critic"] = f"{reviewed}/{len(of('critic'))} images reviewed by the LLM"
    if not n_images or 1 - reviewed / n_images >= MAX_CRITIC_FAIL:
        failures.append(f"critic: LLM review on only {reviewed}/{n_images} images")
    judged = sum(v is not None for r in of("judge") for v in r.get("answers", {}).values())
    stats["judge"] = f"{judged} answers"
    if not judged:
        failures.append("judge: no answer")
    finals = [v for r in of("aggregator") for v in r["final"].values()]
    unanswered = sum(v is None for v in finals)
    stats["unanswered"] = f"{unanswered}/{len(finals)}"
    if not finals or unanswered / len(finals) > MAX_UNANSWERED:
        failures.append(f"unanswered questions: {unanswered}/{len(finals)}")
    experts = sorted({r.get("model", "?") for r in of("visual_expert")})
    stats["expert"] = ", ".join(experts)
    if not experts or any(expert not in m for m in experts):
        failures.append(f"expert is not {expert} ({stats['expert']}): the configured calibration does not apply")
    return {"failures": failures, "stats": stats, "minutes": durations, "hours": seconds / 3600,
            "mean": final and final["mean"], "project": project}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run folder (or its telemetry.jsonl)")
    ap.add_argument("--project", type=int, default=835, help="images of the next run (dev = 835)")
    ap.add_argument("--expert", default="dinov3", help="expert model the anomaly thresholds were calibrated for")
    args = ap.parse_args()
    path = args.run / "telemetry.jsonl" if args.run.is_dir() else args.run
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    r = report(events, args.project, args.expert)
    for k, v in r["stats"].items():
        print(f"{k:12s} {v}")
    for k, v in r["minutes"].items():
        print(f"{k:20s} {v:6.1f} min")
    if r["mean"] is not None:
        print(f"MEAN (7 subtasks) {r['mean']:.1%}")
    print(f"projected for {r['project']} images: {r['hours']:.1f} h (check the compute units left)")
    print("VERDICT: " + ("FAIL\n  - " + "\n  - ".join(r["failures"]) if r["failures"] else "OK"))
    return 1 if r["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
