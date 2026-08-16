# NIFTY Signal Pod — Fine-Tuned SLM Trading Signal Generator with Safety Orchestration

**Kaggle notebook URL:** https://www.kaggle.com/code/adityakumar2907/notebook4b2fc22bae/edit

---

## Repository structure

```
signal_pod/
├── data_audit.py             # Step 1 — run before any training
├── eval_suite.py             # Step 2 — commit before first Kaggle run
├── finetune.py               # Step 3 — run on Kaggle T4 GPU
├── pod.py                    # Signal pod: inference module
├── orchestrator.py           # Orchestrator: 3 suppression rules
├── retrieve.py               # PROVIDED — do not modify
├── run_eval_window.py        # Run orchestrator over eval window
├── requirements.txt
└── README.md

Data (provided, not committed to repo — place alongside .py files):
├── market_states.parquet
├── finetune_instructions.jsonl
└── rag_corpus.jsonl
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Execution order

### Step 1 — Data audit (before any training)
```bash
python data_audit.py
```
Outputs: `finetune_clean.jsonl` (255 clean examples), `finetune_dropped.jsonl` (45 poisoned), `audit_report.json`.

**Key finding:** Rows 47–91 have `conviction` as a string (`"high"`, `"0.8 (high)"`, etc.) instead of a float. All 45 dropped.

### Step 2 — Commit eval suite before training
```bash
git add eval_suite.py
git commit -m "eval suite with pre-committed thresholds — before first training run"
```

### Step 3 — Fine-tune on Kaggle (T4 GPU)
Upload `finetune.py` + `finetune_clean.jsonl` to Kaggle. Run:
```python
# Experiment 1 (baseline)
!python finetune.py run_001_baseline

# Experiment 2 (rank 4)
!python finetune.py run_002_rank4
# (edit LORA_R=4, LORA_ALPHA=8 before running)

# Experiment 5 (RAG-augmented prompts)
!python finetune.py run_005_rag_prompts --rag
```
Download the resulting `lora_adapter/` directory.

### Step 4 — Run orchestrator over eval window
```bash
# Baseline (no RAG):
python run_eval_window.py --adapter lora_adapter

# RAG ablation (runs both conditions automatically):
python run_eval_window.py --adapter lora_adapter --ablation
```

### Step 5 — Evaluate
```bash
python eval_suite.py \
  --decisions orchestrator_decisions.ndjson \
  --market_states market_states.parquet \
  --output eval_report.json
```

---

## Architecture

```
market_states.parquet  ──►  [Orchestrator]
                                  │
                  ┌───────────────┼───────────────────────────────┐
                  │               │                               │
        Rule 1: ADX < 20   [Signal Pod]  ◄── retrieve(ms, k=3)  Rule 3: conviction < 0.40
        REGIME_SUPPRESS     (TinyLlama +       [optional RAG]    LOW_CONVICTION
        return NEUTRAL       LoRA adapter,                       downgrade → NEUTRAL
        no model call        CPU 4-bit)
                                  │
                          Rule 2: parse fail
                          PARSE_FAIL → NEUTRAL
                                  │
                          ┌───────┘
                          ▼
                  orchestrator_decisions.ndjson
                  (downstream reads this only)
```

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base model | TinyLlama-1.1B-Chat | Fits T4 at rank 16; allows 5+ experiments in 30hr/wk |
| LoRA rank | 8 | Avoids under-fit (rank 4) and over-fit (rank 16) on 255 examples |
| Data handling | Drop 45 poisoned rows | Remapping arbitrary; poison teaches model to emit non-numeric conviction |
| Conviction design | Learned text token (discrete scoring) | Softmax over direction token is wrong — see pod.py docstring |
| Orchestrator | Pure Python, no ML | Safety boundary must be deterministic |
| Eval method | Walk-forward, 5-day blocks | k-fold is disqualifying on time series |

---

## Pre-committed pass/fail thresholds (Section 1)

| Metric | PASS | FAIL |
|---|---|---|
| Directional accuracy (overall) | ≥ 52% | < 48% |
| Directional accuracy (worst block) | ≥ 45% | < 40% |
| Schema pass rate | 100% | < 99% |
| Suppress rate | ≤ 35% | > 50% |
| Parse fail rate | < 5% | ≥ 10% |
| High-VIX accuracy (VIX ≥ 20) | ≥ 45% | < 40% |
| Conviction mean (non-NEUTRAL) | 0.45–0.75 | outside range |
