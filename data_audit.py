"""
data_audit.py
=============
Audits finetune_instructions.jsonl, writes a clean training file, and produces
a machine-readable audit_report.json.

Findings (discovered during pre-training inspection):
  - Rows 47-91 (45 examples): conviction field is a STRING instead of a float.
    Values observed: '0.8 (high)', 'high', 'moderate', 'low', 'high confidence',
    'moderate confidence', 'strong', 'weak'.
  - All 255 remaining examples are structurally clean.
  - Training ADX range: 23.7 - 30.4 (NO examples with ADX < 20).
    This is a coverage gap — the orchestrator suppresses ADX < 20 but the model
    has zero training exposure to near-threshold regimes.

Decision: DROP all 45 poisoned examples.
Rationale: Remapping string labels to floats would inject an arbitrary mapping
("high" → 0.75) that is not grounded in the data-generating process. Dropping
reduces the training set to 255 examples but eliminates a systematic bias that
would teach the model to occasionally emit non-numeric conviction values —
directly causing PARSE_FAIL triggers in the orchestrator.
"""

import json
import statistics
from pathlib import Path
from collections import Counter

RAW_PATH    = Path("finetune_instructions.jsonl")
CLEAN_PATH  = Path("finetune_clean.jsonl")
DROP_PATH   = Path("finetune_dropped.jsonl")
REPORT_PATH = Path("audit_report.json")

VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS   = {"intraday", "next_session"}

REQUIRED_INPUT  = ["nifty_spot","atm_iv","iv_skew_25d","pcr","adx_14",
                   "realized_vol_5d","vix_india","dte_nearest","moneyness_band"]
REQUIRED_OUTPUT = ["direction","conviction","horizon","signal_id","generated_at"]


def audit_example(idx: int, ex: dict) -> list[dict]:
    findings = []

    # --- output parse ---
    try:
        out = json.loads(ex["output"])
    except Exception as e:
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "output_parse_error", "detail": str(e)})
        return findings

    # --- conviction type (the main bug) ---
    conv = out.get("conviction")
    if isinstance(conv, str):
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "conviction_is_string", "detail": conv})
    elif conv is None:
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "conviction_missing", "detail": None})
    elif not (0.0 <= float(conv) <= 1.0):
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "conviction_out_of_range", "detail": conv})

    # --- direction / horizon ---
    if out.get("direction") not in VALID_DIRECTIONS:
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "invalid_direction", "detail": out.get("direction")})
    if out.get("horizon") not in VALID_HORIZONS:
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "invalid_horizon", "detail": out.get("horizon")})

    # --- missing output fields ---
    for f in REQUIRED_OUTPUT:
        if f not in out:
            findings.append({"idx": idx, "severity": "WARNING",
                             "type": "missing_output_field", "detail": f})

    # --- input parse ---
    try:
        inp = json.loads(ex["input"])
    except Exception as e:
        findings.append({"idx": idx, "severity": "CRITICAL",
                         "type": "input_parse_error", "detail": str(e)})
        return findings

    for f in REQUIRED_INPUT:
        if f not in inp:
            findings.append({"idx": idx, "severity": "WARNING",
                             "type": "missing_input_field", "detail": f})

    return findings


def main():
    with open(RAW_PATH) as f:
        raw = [json.loads(l) for l in f]

    clean, dropped, all_findings = [], [], []

    for idx, ex in enumerate(raw):
        findings = audit_example(idx, ex)
        critical = [f for f in findings if f["severity"] == "CRITICAL"]
        all_findings.extend(findings)
        if critical:
            dropped.append({"idx": idx, "example": ex, "findings": critical})
        else:
            clean.append(ex)

    # --- write files ---
    with open(CLEAN_PATH, "w") as f:
        for ex in clean: f.write(json.dumps(ex) + "\n")
    with open(DROP_PATH, "w") as f:
        for item in dropped: f.write(json.dumps(item) + "\n")

    # --- compute clean stats ---
    convs    = [float(json.loads(e["output"])["conviction"]) for e in clean]
    dirs     = [json.loads(e["output"])["direction"] for e in clean]
    horizons = [json.loads(e["output"])["horizon"] for e in clean]
    adxs     = [json.loads(e["input"])["adx_14"] for e in clean]
    vixs     = [json.loads(e["input"])["vix_india"] for e in clean]
    string_vals = Counter(f["detail"] for f in all_findings
                          if f["type"] == "conviction_is_string")
    bad_idxs = sorted(set(f["idx"] for f in all_findings if f["severity"]=="CRITICAL"))

    report = {
        "raw_total":     len(raw),
        "clean_total":   len(clean),
        "dropped_total": len(dropped),
        "bad_index_range": f"{min(bad_idxs)}-{max(bad_idxs)}" if bad_idxs else "none",
        "finding_types": dict(Counter(f["type"] for f in all_findings)),
        "string_conviction_values": dict(string_vals),
        "drop_decision": "All 45 examples with string conviction dropped. Remapping rejected: arbitrary mapping not grounded in data.",
        "clean_stats": {
            "direction_dist":    dict(Counter(dirs)),
            "horizon_dist":      dict(Counter(horizons)),
            "conviction_min":    round(min(convs), 4),
            "conviction_max":    round(max(convs), 4),
            "conviction_mean":   round(statistics.mean(convs), 4),
            "conviction_stdev":  round(statistics.stdev(convs), 4),
            "adx_min":  round(min(adxs), 2),
            "adx_max":  round(max(adxs), 2),
            "adx_mean": round(statistics.mean(adxs), 2),
            "vix_min":  round(min(vixs), 2),
            "vix_max":  round(max(vixs), 2),
            "vix_mean": round(statistics.mean(vixs), 2),
            "adx_below_20_count": sum(1 for a in adxs if a < 20),
        },
        "coverage_gaps": [
            "ADX < 20: ZERO training examples. Orchestrator suppresses at ADX<20 but model never trained near this boundary.",
            "VIX > 15.4: ZERO training examples. Eval window has VIX up to 30.95.",
        ]
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # --- console ---
    print("=" * 62)
    print("  DATA AUDIT COMPLETE")
    print("=" * 62)
    print(f"  Raw      : {len(raw)}")
    print(f"  Clean    : {len(clean)}")
    print(f"  Dropped  : {len(dropped)}  (rows {report['bad_index_range']})")
    print(f"\n  Finding types:")
    for k, v in report["finding_types"].items():
        print(f"    {k:<35} {v}")
    print(f"\n  String conviction values found:")
    for k, v in string_vals.items():
        print(f"    '{k}': {v}×")
    print(f"\n  Clean stats: {report['clean_stats']}")
    print(f"\n  COVERAGE GAPS (critical for eval design):")
    for g in report["coverage_gaps"]:
        print(f"    ⚠  {g}")
    print(f"\n  Outputs: {CLEAN_PATH}  {DROP_PATH}  {REPORT_PATH}")


if __name__ == "__main__":
    main()
