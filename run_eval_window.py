"""
run_eval_window.py
==================
Run the orchestrator over the evaluation window (days 31-60) and write
decisions to orchestrator_decisions.ndjson.

Usage:
  # baseline (no RAG):
  python run_eval_window.py --adapter lora_adapter

  # with RAG:
  python run_eval_window.py --adapter lora_adapter --rag

  # RAG ablation (runs both conditions, writes two decision files):
  python run_eval_window.py --adapter lora_adapter --ablation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from pod import SignalPod
from orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def run_window(pod, market_df: pd.DataFrame, log_path: str,
               use_rag: bool = False) -> list[dict]:
    orch    = Orchestrator(pod=pod, log_path=log_path, use_rag=use_rag)
    outputs = []

    for i, row in market_df.iterrows():
        state = {
            "nifty_spot":      float(row["nifty_spot"]),
            "atm_iv":          float(row["atm_iv"]),
            "iv_skew_25d":     float(row["iv_skew_25d"]),
            "pcr":             float(row["pcr"]),
            "adx_14":          float(row["adx_14"]),
            "realized_vol_5d": float(row["realized_vol_5d"]),
            "vix_india":       float(row["vix_india"]),
            "dte_nearest":     int(row["dte_nearest"]),
            "moneyness_band":  str(row["moneyness_band"]),
        }
        out = orch.run(state)
        outputs.append(out)

        if (i + 1) % 65 == 0:
            log.info("Processed %d ticks | last: %s conv=%.2f",
                     i + 1, out["direction"], out["conviction"])

    return outputs


def ablation(pod, eval_df: pd.DataFrame):
    """Run no-RAG and with-RAG over the eval window and compare."""
    log.info("=== RAG Ablation: No RAG ===")
    run_window(pod, eval_df, "decisions_norag.ndjson",   use_rag=False)

    log.info("=== RAG Ablation: With RAG ===")
    run_window(pod, eval_df, "decisions_withrag.ndjson", use_rag=True)

    # Compare
    ts_to_label = dict(zip(
        eval_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M"),
        eval_df["label"]
    ))

    results = {"n": 0, "norag_correct": 0, "rag_correct": 0,
               "direction_changed": 0, "rag_helped": 0, "rag_hurt": 0,
               "conv_delta": []}

    with open("decisions_norag.ndjson") as fa, \
         open("decisions_withrag.ndjson") as fb:
        for la, lb in zip(fa, fb):
            da, db = json.loads(la), json.loads(lb)
            ts = (da["final_output"].get("generated_at",""))[:16]
            label = ts_to_label.get(ts)
            if label is None:
                continue
            results["n"] += 1
            ca = 1 if da["final_output"]["direction"] == label else 0
            cb = 1 if db["final_output"]["direction"] == label else 0
            results["norag_correct"] += ca
            results["rag_correct"]   += cb
            if da["final_output"]["direction"] != db["final_output"]["direction"]:
                results["direction_changed"] += 1
                results["rag_helped"] += (1 if cb > ca else 0)
                results["rag_hurt"]   += (1 if cb < ca else 0)
            results["conv_delta"].append(
                db["final_output"]["conviction"] - da["final_output"]["conviction"]
            )

    n = results["n"]
    if n:
        import statistics as st
        print("\n" + "=" * 55)
        print("  RAG ABLATION SUMMARY")
        print("=" * 55)
        print(f"  Ticks evaluated       : {n}")
        print(f"  Accuracy no-RAG       : {results['norag_correct']/n:.4f}")
        print(f"  Accuracy with-RAG     : {results['rag_correct']/n:.4f}")
        print(f"  Accuracy delta        : {(results['rag_correct']-results['norag_correct'])/n:+.4f}")
        print(f"  Directions changed    : {results['direction_changed']} ({100*results['direction_changed']/n:.1f}%)")
        print(f"    RAG helped          : {results['rag_helped']}")
        print(f"    RAG hurt            : {results['rag_hurt']}")
        cd = results["conv_delta"]
        print(f"  Mean conviction delta : {st.mean(cd):+.4f}  (std={st.stdev(cd):.4f})")
        print()

        report = {
            "n": n,
            "norag_accuracy":  round(results["norag_correct"]/n, 4),
            "rag_accuracy":    round(results["rag_correct"]/n, 4),
            "accuracy_delta":  round((results["rag_correct"]-results["norag_correct"])/n, 4),
            "direction_changed_pct": round(results["direction_changed"]/n, 4),
            "rag_helped": results["rag_helped"],
            "rag_hurt":   results["rag_hurt"],
            "mean_conviction_delta": round(st.mean(cd), 4),
        }
        with open("rag_ablation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("  Saved: rag_ablation_report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",       default="lora_adapter")
    parser.add_argument("--base_model",    default=BASE_MODEL)
    parser.add_argument("--market_states", default="market_states.parquet")
    parser.add_argument("--out",           default="orchestrator_decisions.ndjson")
    parser.add_argument("--rag",           action="store_true")
    parser.add_argument("--ablation",      action="store_true")
    args = parser.parse_args()

    mdf = pd.read_parquet(args.market_states)
    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"])
    mdf = mdf.sort_values("timestamp").reset_index(drop=True)

    dates = sorted(mdf["timestamp"].dt.date.unique())
    eval_df = mdf[mdf["timestamp"].dt.date.isin(set(dates[30:]))].reset_index(drop=True)
    log.info("Eval window: %d ticks (%s → %s)",
             len(eval_df),
             eval_df["timestamp"].iloc[0].date(),
             eval_df["timestamp"].iloc[-1].date())

    pod = SignalPod(base_model=args.base_model, adapter_path=args.adapter)

    if args.ablation:
        ablation(pod, eval_df)
    else:
        run_window(pod, eval_df, args.out, use_rag=args.rag)
        log.info("Done. Decisions → %s", args.out)
        log.info("Next: python eval_suite.py --decisions %s", args.out)


if __name__ == "__main__":
    main()
