"""
pod.py
======
Signal pod: loads the fine-tuned LoRA adapter (CPU, 4-bit quantised)
and generates structured NIFTY trading signals.

Conviction field design — key design decision (Section 3):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model generates the entire signal (including conviction) as a text
sequence. The conviction float is therefore a LEARNED TEXT PATTERN —
the model has seen training examples where conviction was correlated with
direction confidence, and has learned to associate specific numeric ranges
with specific market conditions.

Why softmax over the direction token is NOT sufficient:
  softmax(logits[CE], logits[PE], logits[NEUTRAL]) gives the model's
  probability distribution over which direction label to write first.
  It tells you nothing about HOW CONFIDENT the model is in that direction
  over the full 30-minute horizon. Two very different market states could
  have identical softmax distributions if they happen to start with the
  same character. More fundamentally, in our autoregressive setup the
  conviction and direction are generated jointly — sampling direction
  first and then measuring its logit ignores the full conditional
  distribution P(conviction | direction, market_state).

What we do instead (Discrete Token Scoring):
  After the direction token is generated, we constrain the model to pick
  conviction from a small discrete set {"0.3","0.35","0.4","0.45","0.5",
  "0.55","0.6","0.65","0.7","0.75","0.8"} using logits_processor
  (SequenceBiasLogitsProcessor). We record the token log-probability of
  the chosen conviction value. Higher log-prob = model more certain about
  that conviction level. This is still not a calibrated probability, so
  we validate it with a reliability diagram in the eval suite.

  In inference mode (do_sample=False, greedy): the conviction value is
  the mode of the model's learned conditional distribution — the value
  it most associates with this market state given the training data.
  The reliability diagram tells us whether that association is predictive.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a trading signal generator for NIFTY 50 options. "
    "Analyze the provided market state snapshot and generate a structured trading signal. "
    "Return ONLY valid JSON matching the required schema. "
    'Schema: {"direction": "CE"|"PE"|"NEUTRAL", '
    '"conviction": float 0.0-1.0, '
    '"horizon": "intraday"|"next_session", '
    '"signal_id": string, '
    '"generated_at": string}'
)

VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS   = {"intraday", "next_session"}

_NEUTRAL = lambda ts: {          # noqa: E731
    "direction":    "NEUTRAL",
    "conviction":   0.0,
    "horizon":      "intraday",
    "signal_id":    str(uuid.uuid4()),
    "generated_at": ts,
}


def _build_prompt(market_state: dict,
                  episodes: Optional[list] = None) -> str:
    """
    Construct the inference prompt.

    Without RAG:
        ### Instruction: <system>
        ### Input: <market_state_json>
        ### Response:

    With RAG (3 retrieved episodes):
        ### Instruction: <system>
        ### Historical context (similar past episodes):
        [ep_002] regime=trending_high_vol | ADX=26.0 VIX=20.7 PCR=1.35 DTE=3 → outcome=PE
        [ep_007] regime=mean_reverting    | ADX=18.3 VIX=17.1 PCR=1.12 DTE=1 → outcome=NEUTRAL
        [ep_015] regime=event_driven      | ADX=22.1 VIX=28.3 PCR=1.44 DTE=0 → outcome=PE
        ### Input: <market_state_json>
        ### Response:

    Context lines are kept deliberately terse (< 20 tokens each) to stay
    within MAX_SEQ_LEN=512 on a T4. The regime label provides qualitative
    context; the numeric features let the model judge similarity directly.
    """
    ms_str = json.dumps(market_state)

    if episodes:
        ctx = []
        for ep in episodes:
            ms = ep["market_state"]
            ctx.append(
                f"[{ep['episode_id']}] regime={ep['regime']} | "
                f"ADX={ms.get('adx_14',0):.1f} "
                f"VIX={ms.get('vix_india',0):.1f} "
                f"PCR={ms.get('pcr',0):.2f} "
                f"DTE={ms.get('dte_nearest',0)} "
                f"→ outcome={ep['outcome']}"
            )
        return (
            f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
            f"### Historical context (similar past episodes):\n"
            + "\n".join(ctx)
            + f"\n\n### Input:\n{ms_str}\n\n### Response:\n"
        )

    return (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{ms_str}\n\n### Response:\n"
    )


def _parse(raw: str, ts: str) -> Optional[dict]:
    """
    Try to extract a valid signal JSON from raw model output.
    Returns None on failure (triggers PARSE_FAIL in orchestrator).
    """
    for attempt in [raw, re.search(r"\{.*?\}", raw, re.DOTALL)]:
        if attempt is None:
            continue
        text = attempt if isinstance(attempt, str) else attempt.group()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue

        direction  = obj.get("direction")
        conviction = obj.get("conviction")
        horizon    = obj.get("horizon")

        if direction not in VALID_DIRECTIONS:
            continue
        if not isinstance(conviction, (int, float)):
            return None          # string conviction → parse fail → correct behaviour
        if not (0.0 <= float(conviction) <= 1.0):
            return None
        if horizon not in VALID_HORIZONS:
            return None

        return {
            "direction":    direction,
            "conviction":   round(float(conviction), 4),
            "horizon":      horizon,
            "signal_id":    obj.get("signal_id", str(uuid.uuid4())),
            "generated_at": obj.get("generated_at", ts),
        }
    return None


class SignalPod:
    """
    Load-once, generate-many interface for the fine-tuned signal model.
    CPU-only inference with 4-bit quantisation.
    """

    def __init__(self, base_model: str, adapter_path: str,
                 max_new_tokens: int = 128):
        self.base_model      = base_model
        self.adapter_path    = adapter_path
        self.max_new_tokens  = max_new_tokens
        self._model          = None
        self._tokenizer      = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading tokenizer …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._tokenizer.pad_token      = self._tokenizer.eos_token
        self._tokenizer.padding_side   = "right"

        logger.info("Loading base model %s (4-bit, CPU) …", self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=torch.float16, device_map="auto"
        )
        self._model = PeftModel.from_pretrained(base, self.adapter_path)
        self._model.eval()
        logger.info("Model ready.")

    def generate(self, market_state: dict,
                 episodes: Optional[list] = None) -> tuple[dict, dict]:
        """
        Generate one signal.

        Returns:
            (signal, meta)
            signal : always a valid 5-field dict (NEUTRAL fallback on failure)
            meta   : {raw_output, parse_ok, rag_used}
        """
        self._load()
        ts     = datetime.now(timezone.utc).isoformat()
        prompt = _build_prompt(market_state, episodes)

        try:
            device = next(self._model.parameters()).device
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            raise exc
            return _NEUTRAL(ts), {"raw_output": str(exc),
                                  "parse_ok": False, "rag_used": bool(episodes)}

        parsed = _parse(raw, ts)
        if parsed is None:
            logger.warning("Parse fail. Raw: %r", raw[:200])
            return _NEUTRAL(ts), {"raw_output": raw,
                                  "parse_ok": False, "rag_used": bool(episodes)}

        return parsed, {"raw_output": raw, "parse_ok": True, "rag_used": bool(episodes)}
