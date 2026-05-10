"""
eval_suite.py
=============
Walk-forward evaluation suite for the NIFTY signal pod.

COMMIT THIS FILE TO THE REPOSITORY BEFORE THE FIRST KAGGLE TRAINING RUN.
Section 1 thresholds are hard-coded below and must not be changed after training.

Evaluation design:
  - Walk-forward only. Days 31-60 (2024-11-12 to 2024-12-23).
  - 6 non-overlapping 5-day rolling blocks.
  - k-fold cross-validation on a time series is DISQUALIFYING — not used.
  - All metrics reported with 95% confidence intervals.

Pre-committed pass/fail thresholds (Section 1):
  ┌──────────────────────────────────────────┬──────────┬──────────┐
  │ Metric                                   │  PASS    │  FAIL    │
  ├──────────────────────────────────────────┼──────────┼──────────┤
  │ Directional accuracy (overall)           │  ≥ 0.52  │  < 0.48  │
  │ Directional accuracy (worst 5-day block) │  ≥ 0.45  │  < 0.40  │
  │ Output schema pass rate                  │  1.00    │  < 0.99  │
  │ Conviction mean (non-NEUTRAL signals)    │ 0.45–0.75│  outside │
  │ Orchestrator suppress rate (ADX<20)      │  ≤ 0.35  │  > 0.50  │
  │ Low-conviction downgrade rate            │  ≤ 0.40  │  > 0.55  │
  │ Parse fail rate                          │  < 0.05  │  ≥ 0.10  │
  │ High-VIX directional accuracy (VIX≥20)  │  ≥ 0.45  │  < 0.40  │
  └──────────────────────────────────────────┴──────────┴──────────┘

Rationale for 52% overall accuracy floor:
  The majority-class baseline on the eval window is 38.7% (NEUTRAL = 151/390).
  A 52% threshold represents a ~13pp lift over baseline — meaningful but
  conservative given the small signal-to-noise ratio in 30-min NIFTY data.

Conviction validity (addressed as a design problem, not just a metric):
  The conviction field is a learned float generated as text by the model.
  It is NOT a softmax probability. We validate it via a reliability diagram:
  for each conviction bin [0.3-0.4, 0.4-0.5, ... 0.7-0.8], we report
  empirical directional accuracy. A trustworthy pod should show monotonically
  increasing accuracy with conviction — higher stated conviction should
  correspond to higher hit rate. Flat or inverted reliability indicates the
  conviction field carries no calibrated information.

Usage:
  python eval_suite.py \\
    --decisions orchestrator_decisions.ndjson \\
    --market_states market_states.parquet \\
    --output eval_report.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Pre-committed thresholds ────────────────────────────────────────────────
T = {
    "accuracy_overall_pass":       0.52,
    "accuracy_overall_fail":       0.48,
    "accuracy_block_worst_pass":   0.45,
    "accuracy_block_worst_fail":   0.40,
    "schema_pass_rate_pass":       1.00,
    "schema_pass_rate_fail":       0.99,
    "conviction_mean_min":         0.45,
    "conviction_mean_max":         0.75,
    "suppress_rate_pass":          0.35,
    "suppress_rate_fail":          0.50,
    "downgrade_rate_pass":         0.40,
    "downgrade_rate_fail":         0.55,
    "parse_fail_rate_pass":        0.05,
    "parse_fail_rate_fail":        0.10,
    "high_vix_accuracy_pass":      0.45,
    "high_vix_accuracy_fail":      0.40,
    "high_vix_threshold":          20.0,
}

REQUIRED_FIELDS = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}


# ─── Stat helpers ─────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
    return (round(max(0.0, c-m), 4), round(min(1.0, c+m), 4))


def mean_ci(vals: list[float], z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95% CI for the mean of vals."""
    n = len(vals)
    if n < 2:
        return (round(float(vals[0]), 4), round(float(vals[0]), 4)) if vals else (0.0, 0.0)
    mu = float(np.mean(vals))
    se = float(np.std(vals, ddof=1)) / math.sqrt(n)
    return (round(mu - z*se, 4), round(mu + z*se, 4))


# ─── Core metrics per set of decisions ───────────────────────────────────────

def compute_metrics(decisions: list[dict],
                    market_df: pd.DataFrame) -> dict:
    """Compute all metrics for a subset of decisions against ground truth."""

    ts_to_label = {}
    ts_to_vix   = {}
    for _, row in market_df.iterrows():
        key = pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%dT%H:%M")
        ts_to_label[key] = str(row["label"])
        ts_to_vix[key]   = float(row["vix_india"])

    total = len(decisions)
    if total == 0:
        return {}

    suppress_n    = sum(1 for d in decisions if d["reason_code"] == "REGIME_SUPPRESS")
    parse_fail_n  = sum(1 for d in decisions if d["reason_code"] == "PARSE_FAIL")
    downgrade_n   = sum(1 for d in decisions if d["reason_code"] == "LOW_CONVICTION")
    schema_ok_n   = sum(1 for d in decisions
                        if REQUIRED_FIELDS.issubset(set(d["final_output"].keys()))
                        and d["final_output"].get("direction") in VALID_DIRECTIONS)

    # Actionable = not suppressed, not parse-failed
    actionable = [d for d in decisions
                  if d["reason_code"] not in ("REGIME_SUPPRESS", "PARSE_FAIL")]
    n_act = len(actionable)

    correct_all     = 0
    correct_high    = 0
    n_high          = 0
    convs_nonneutral = []

    for dec in actionable:
        out  = dec["final_output"]
        ts   = (out.get("generated_at") or dec.get("timestamp", ""))[:16]
        label = ts_to_label.get(ts)
        vix   = ts_to_vix.get(ts, 0.0)
        if label is None:
            continue
        direction  = out["direction"]
        conviction = float(out["conviction"])
        hit = 1 if direction == label else 0
        correct_all += hit
        if direction != "NEUTRAL":
            convs_nonneutral.append(conviction)
        if vix >= T["high_vix_threshold"]:
            correct_high += hit
            n_high       += 1

    acc_all  = round(correct_all / n_act, 4) if n_act else None
    acc_high = round(correct_high / n_high, 4) if n_high else None

    return {
        "total":              total,
        "suppress_n":         suppress_n,
        "suppress_rate":      round(suppress_n / total, 4),
        "suppress_rate_ci":   wilson_ci(suppress_n, total),
        "parse_fail_n":       parse_fail_n,
        "parse_fail_rate":    round(parse_fail_n / total, 4),
        "downgrade_n":        downgrade_n,
        "downgrade_rate":     round(downgrade_n / n_act, 4) if n_act else None,
        "schema_pass_rate":   round(schema_ok_n / total, 4),
        "directional_accuracy":    acc_all,
        "directional_accuracy_ci": wilson_ci(correct_all, n_act) if n_act else None,
        "high_vix_accuracy":       acc_high,
        "high_vix_accuracy_ci":    wilson_ci(correct_high, n_high) if n_high else None,
        "n_high_vix":              n_high,
        "conviction_mean":    round(float(np.mean(convs_nonneutral)), 4)
                              if convs_nonneutral else None,
        "conviction_mean_ci": mean_ci(convs_nonneutral)
                              if len(convs_nonneutral) > 1 else None,
    }


# ─── Reliability diagram ─────────────────────────────────────────────────────

def reliability_diagram(decisions: list[dict],
                         market_df: pd.DataFrame) -> dict:
    ts_to_label = {
        pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%dT%H:%M"): str(row["label"])
        for _, row in market_df.iterrows()
    }
    bins = [(i/10, (i+1)/10) for i in range(3, 10)]  # 0.3-0.4 … 0.9-1.0
    result = {}

    for lo, hi in bins:
        label_ = f"{lo:.1f}-{hi:.1f}"
        bin_correct, bin_total = 0, 0
        for dec in decisions:
            out = dec["final_output"]
            if out["direction"] == "NEUTRAL":
                continue
            conv = float(out["conviction"])
            if not (lo <= conv < hi):
                continue
            ts    = (out.get("generated_at") or dec.get("timestamp",""))[:16]
            label = ts_to_label.get(ts)
            if label is None:
                continue
            bin_total  += 1
            bin_correct += 1 if out["direction"] == label else 0
        if bin_total:
            result[label_] = {
                "n":        bin_total,
                "accuracy": round(bin_correct / bin_total, 4),
                "ci_95":    wilson_ci(bin_correct, bin_total),
            }
    return result


# ─── Threshold verdict ────────────────────────────────────────────────────────

def verdict(value, pass_thr, fail_thr, higher_better=True) -> str:
    if value is None:
        return "N/A"
    if higher_better:
        if value >= pass_thr: return "PASS"
        if value <= fail_thr: return "FAIL"
        return "MARGINAL"
    else:
        if value <= pass_thr: return "PASS"
        if value >= fail_thr: return "FAIL"
        return "MARGINAL"


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval(decisions_path: str, market_path: str, output_path: str):
    with open(decisions_path) as f:
        decisions = [json.loads(l) for l in f if l.strip()]

    mdf = pd.read_parquet(market_path)
    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"])
    mdf = mdf.sort_values("timestamp").reset_index(drop=True)

    dates = sorted(mdf["timestamp"].dt.date.unique())
    eval_dates = dates[30:]   # days 31–60

    # Tag each decision with its date
    for dec in decisions:
        ts = (dec.get("timestamp") or
              dec["final_output"].get("generated_at", ""))
        try:
            dec["_date"] = pd.Timestamp(ts).date()
        except Exception:
            dec["_date"] = None

    eval_set = set(eval_dates)
    eval_decisions = [d for d in decisions if d.get("_date") in eval_set]
    eval_mdf       = mdf[mdf["timestamp"].dt.date.isin(eval_set)]

    # 5-day blocks
    blocks = [eval_dates[i:i+5] for i in range(0, len(eval_dates), 5)]
    block_reports = []
    for b_idx, block_dates in enumerate(blocks):
        b_set  = set(block_dates)
        b_decs = [d for d in eval_decisions if d.get("_date") in b_set]
        b_mdf  = mdf[mdf["timestamp"].dt.date.isin(b_set)]
        m = compute_metrics(b_decs, b_mdf)
        block_reports.append({
            "block":  b_idx + 1,
            "dates":  [str(d) for d in block_dates],
            **m
        })

    # Aggregate
    agg = compute_metrics(eval_decisions, eval_mdf)
    agg["schema_pass_rate"] = round(
        sum(1 for d in eval_decisions
            if REQUIRED_FIELDS.issubset(d["final_output"].keys())) / len(eval_decisions), 4
    ) if eval_decisions else 0.0

    # Reliability
    rel = reliability_diagram(eval_decisions, eval_mdf)

    # Verdicts
    verdicts = {
        "directional_accuracy":  verdict(agg.get("directional_accuracy"),
                                          T["accuracy_overall_pass"], T["accuracy_overall_fail"]),
        "schema_pass_rate":      verdict(agg.get("schema_pass_rate"),
                                          T["schema_pass_rate_pass"], T["schema_pass_rate_fail"]),
        "suppress_rate":         verdict(agg.get("suppress_rate"),
                                          T["suppress_rate_pass"], T["suppress_rate_fail"],
                                          higher_better=False),
        "parse_fail_rate":       verdict(agg.get("parse_fail_rate"),
                                          T["parse_fail_rate_pass"], T["parse_fail_rate_fail"],
                                          higher_better=False),
        "high_vix_accuracy":     verdict(agg.get("high_vix_accuracy"),
                                          T["high_vix_accuracy_pass"], T["high_vix_accuracy_fail"]),
        "conviction_mean":       ("PASS"
                                  if agg.get("conviction_mean") and
                                     T["conviction_mean_min"] <= agg["conviction_mean"] <= T["conviction_mean_max"]
                                  else "FAIL" if agg.get("conviction_mean") else "N/A"),
    }
    worst_block_acc = min((b.get("directional_accuracy") or 0) for b in block_reports)
    verdicts["worst_block_accuracy"] = verdict(worst_block_acc,
                                               T["accuracy_block_worst_pass"],
                                               T["accuracy_block_worst_fail"])
    verdicts["OVERALL"] = ("PASS" if all(v=="PASS" for v in verdicts.values() if v!="N/A")
                           else "FAIL" if any(v=="FAIL" for v in verdicts.values()) else "MARGINAL")

    report = {
        "thresholds":  T,
        "aggregate":   agg,
        "blocks":      block_reports,
        "reliability": rel,
        "verdicts":    verdicts,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Console
    print("=" * 62)
    print("  WALK-FORWARD EVALUATION REPORT")
    print("=" * 62)
    print(f"  Eval ticks : {len(eval_decisions)}  |  Dates: {eval_dates[0]} → {eval_dates[-1]}")
    print()
    for k, v in verdicts.items():
        sym = "✓" if v == "PASS" else ("✗" if v == "FAIL" else "~")
        val = agg.get(k.replace("worst_block_", "directional_").replace("conviction_mean", "conviction_mean"))
        print(f"  {sym}  {k:<35}  {v}  (value={val})")
    print()
    print("  Per-block results:")
    for b in block_reports:
        print(f"    Block {b['block']} {b['dates'][0]}–{b['dates'][-1]} | "
              f"acc={b.get('directional_accuracy')}  CI={b.get('directional_accuracy_ci')}  "
              f"suppress={b.get('suppress_rate')}  vix_acc={b.get('high_vix_accuracy')}")
    print()
    print("  Reliability diagram (conviction bin → empirical accuracy):")
    for bin_label, stats in rel.items():
        print(f"    [{bin_label}]  n={stats['n']}  acc={stats['accuracy']}  CI={stats['ci_95']}")
    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions",     default="orchestrator_decisions.ndjson")
    parser.add_argument("--market_states", default="market_states.parquet")
    parser.add_argument("--output",        default="eval_report.json")
    args = parser.parse_args()
    run_eval(args.decisions, args.market_states, args.output)
