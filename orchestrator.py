"""
orchestrator.py
===============
Wraps the SignalPod and enforces three mandatory suppression rules in order.
Downstream systems ONLY read orchestrator output — never raw pod signals.

Rules (applied in strict sequence):
  1. ADX < 20      → REGIME_SUPPRESS : return NEUTRAL without calling model
  2. Parse failure → PARSE_FAIL      : return NEUTRAL, log raw output
  3. Conviction < 0.40 → LOW_CONVICTION : downgrade direction to NEUTRAL

Every decision is logged as one NDJSON line containing:
  timestamp, reason_code, triggering_values, model_called, rag_used,
  raw_pod_output (truncated to 500 chars), final_output.

The orchestrator is pure deterministic Python — no ML inference.
It is the safety boundary between the pod and the downstream pipeline.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ADX_THRESHOLD        = 20.0
CONVICTION_THRESHOLD = 0.40


class ReasonCode:
    REGIME_SUPPRESS = "REGIME_SUPPRESS"
    PARSE_FAIL      = "PARSE_FAIL"
    LOW_CONVICTION  = "LOW_CONVICTION"
    SIGNAL_PASSED   = "SIGNAL_PASSED"


def _neutral(ts: str) -> dict:
    return {
        "direction":    "NEUTRAL",
        "conviction":   0.0,
        "horizon":      "intraday",
        "signal_id":    str(uuid.uuid4()),
        "generated_at": ts,
    }


class Orchestrator:
    """
    Args:
        pod        : SignalPod instance (or any object with .generate())
        log_path   : NDJSON decision log file
        use_rag    : whether to retrieve historical episodes before inference
        rag_k      : number of episodes to retrieve (default 3)
    """

    def __init__(self, pod, log_path: str = "orchestrator_decisions.ndjson",
                 use_rag: bool = False, rag_k: int = 3):
        self.pod      = pod
        self.log_path = Path(log_path)
        self.use_rag  = use_rag
        self.rag_k    = rag_k
        self._retrieve = None

        if use_rag:
            try:
                from retrieve import retrieve
                self._retrieve = retrieve
                logger.info("RAG retrieval enabled.")
            except ImportError:
                logger.warning("retrieve.py not found — RAG disabled.")
                self.use_rag = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, entry: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, market_state: dict) -> dict:
        """
        Process one market state tick through all three rules.
        Always returns a valid 5-field signal dict.

        Section 5 scenario trace:
          09:30 expiry Thursday, VIX = 32.4 (3σ above 30d mean), ADX = 14:

          Step 1 — Rule 1 fires: adx_14=14 < threshold=20.
                   Model is NOT called.
                   REGIME_SUPPRESS logged.
                   NEUTRAL / 0.0 returned.
                   Downstream receives a safe output.

          What is still wrong:
            ADX=21 would pass this gate. The model has ZERO training examples
            with ADX < 20, so at ADX=20.5 during a VIX spike it would call
            inference on completely out-of-distribution data and may output
            a high-conviction directional signal with no supporting evidence.
            The fix: add a secondary VIX threshold check (VIX > 2σ above
            trailing 30d mean → suppress regardless of ADX) and a DTE=0
            expiry-day suppression rule.
        """
        ts  = datetime.now(timezone.utc).isoformat()
        adx = float(market_state.get("adx_14", 0.0))

        # ── Rule 1: Regime check ──────────────────────────────────────────────
        if adx < ADX_THRESHOLD:
            output = _neutral(ts)
            self._log({
                "timestamp":         ts,
                "reason_code":       ReasonCode.REGIME_SUPPRESS,
                "triggering_values": {"adx_14": adx, "threshold": ADX_THRESHOLD},
                "model_called":      False,
                "rag_used":          False,
                "raw_pod_output":    None,
                "final_output":      output,
            })
            return output

        # ── RAG retrieval (optional) ──────────────────────────────────────────
        episodes: Optional[list] = None
        if self.use_rag and self._retrieve is not None:
            try:
                episodes = self._retrieve(market_state, k=self.rag_k)
            except Exception as exc:
                logger.warning("RAG failed: %s — proceeding without context.", exc)
                episodes = None

        # ── Pod inference ─────────────────────────────────────────────────────
        pod_signal, pod_meta = self.pod.generate(market_state, episodes=episodes)

        # ── Rule 2: Parse failure ─────────────────────────────────────────────
        if not pod_meta["parse_ok"]:
            output = _neutral(ts)
            self._log({
                "timestamp":         ts,
                "reason_code":       ReasonCode.PARSE_FAIL,
                "triggering_values": {"adx_14": adx},
                "model_called":      True,
                "rag_used":          pod_meta["rag_used"],
                "raw_pod_output":    str(pod_meta["raw_output"])[:500],
                "final_output":      output,
            })
            return output

        # ── Rule 3: Conviction threshold ──────────────────────────────────────
        conviction = float(pod_signal["conviction"])
        if conviction < CONVICTION_THRESHOLD:
            output     = {**pod_signal, "direction": "NEUTRAL"}
            reason     = ReasonCode.LOW_CONVICTION
        else:
            output     = pod_signal
            reason     = ReasonCode.SIGNAL_PASSED

        self._log({
            "timestamp":         ts,
            "reason_code":       reason,
            "triggering_values": {
                "adx_14":              adx,
                "pod_conviction":      conviction,
                "conviction_threshold": CONVICTION_THRESHOLD,
                "pod_direction":       pod_signal["direction"],
            },
            "model_called":      True,
            "rag_used":          pod_meta["rag_used"],
            "raw_pod_output":    str(pod_meta["raw_output"])[:500],
            "final_output":      output,
        })
        return output


# ── Smoke test (uses mock pod, no GPU required) ────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    class _MockPod:
        def __init__(self, direction="PE", conviction=0.62, parse_ok=True):
            self._d, self._c, self._ok = direction, conviction, parse_ok
        def generate(self, market_state, episodes=None):
            import uuid
            sig = {"direction": self._d, "conviction": self._c,
                   "horizon": "intraday",
                   "signal_id": str(uuid.uuid4()),
                   "generated_at": datetime.now(timezone.utc).isoformat()}
            raw = json.dumps(sig)
            return sig, {"raw_output": raw, "parse_ok": self._ok, "rag_used": False}

    scenarios = [
        ("ADX=14  → REGIME_SUPPRESS",
         {"adx_14": 14.0, "vix_india": 32.4, "nifty_spot": 22400.0,
          "atm_iv": 28.5, "iv_skew_25d": 4.2, "pcr": 1.35,
          "realized_vol_5d": 22.0, "dte_nearest": 0, "moneyness_band": "ATM"},
         _MockPod(), "NEUTRAL", ReasonCode.REGIME_SUPPRESS),

        ("ADX=29, conv=0.62 → SIGNAL_PASSED",
         {"adx_14": 29.35, "vix_india": 14.08, "nifty_spot": 22859.61,
          "atm_iv": 13.41, "iv_skew_25d": 3.88, "pcr": 1.13,
          "realized_vol_5d": 13.61, "dte_nearest": 2, "moneyness_band": "ATM"},
         _MockPod("PE", 0.62), "PE", ReasonCode.SIGNAL_PASSED),

        ("ADX=29, conv=0.31 → LOW_CONVICTION",
         {"adx_14": 29.35, "vix_india": 14.08, "nifty_spot": 22859.61,
          "atm_iv": 13.41, "iv_skew_25d": 3.88, "pcr": 1.13,
          "realized_vol_5d": 13.61, "dte_nearest": 2, "moneyness_band": "ATM"},
         _MockPod("CE", 0.31), "NEUTRAL", ReasonCode.LOW_CONVICTION),

        ("ADX=29, parse_fail → PARSE_FAIL",
         {"adx_14": 29.35, "vix_india": 14.08, "nifty_spot": 22859.61,
          "atm_iv": 13.41, "iv_skew_25d": 3.88, "pcr": 1.13,
          "realized_vol_5d": 13.61, "dte_nearest": 2, "moneyness_band": "ATM"},
         _MockPod(parse_ok=False), "NEUTRAL", ReasonCode.PARSE_FAIL),
    ]

    all_pass = True
    for name, state, pod, exp_dir, exp_reason in scenarios:
        import tempfile, os
        log = tempfile.mktemp(suffix=".ndjson")
        orch = Orchestrator(pod=pod, log_path=log, use_rag=False)
        out  = orch.run(state)
        with open(log) as f:
            dec = json.loads(f.read().strip())
        ok = out["direction"] == exp_dir and dec["reason_code"] == exp_reason
        all_pass = all_pass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        print(f"         dir={out['direction']}  reason={dec['reason_code']}  conv={out['conviction']}")
        os.unlink(log)

    print()
    print("All smoke tests PASSED." if all_pass else "SMOKE TESTS FAILED — check orchestrator logic.")
    sys.exit(0 if all_pass else 1)
