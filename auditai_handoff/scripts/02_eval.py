"""
02_eval.py — measure what the SFT model actually learned.

v2: fixes two bugs from v1:
  (a) First-token scoring was misleading because REJECTED tokenizes to 2 tokens
      [184182, 1613] and APPROVED to 1 token [127802]. We now sum log-probs
      across ALL target tokens (sequence-level scoring).
  (b) The "base baseline" in v1 was a lie. PeftModel.from_pretrained(base, ...)
      wraps `base` IN PLACE. Deleting the wrapper variable doesn't detach the
      adapter. We now use `model.disable_adapter()` context manager to get a
      genuine base-model forward pass on the same loaded weights.

Three evaluations:
  1. Behavioral accuracy on train.csv     (sanity)
  2. Edge-case bias breakdown             (does name override finances?)
  3. Counterfactual surname effect        (causal effect of surname swap)
     ... computed BOTH with adapter on (induced bias) and adapter off (base bias).

Run:
    python 02_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from ml_utils import (
    CKPT_DIR,
    DATA_DIR,
    MODEL_ID,
    load_base_model,
    load_tokenizer,
    render_prompt,
    set_seed,
)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=CKPT_DIR / "sft-v1" / "final")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="defaults to <adapter>/eval/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-eval", type=int, default=300,
                   help="subsample train.csv for the in-distribution sanity check")
    p.add_argument("--no-base-baseline", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Sequence-level scoring of completion strings
# --------------------------------------------------------------------------- #

@torch.no_grad()
def score_completion(model, tokenizer, prompt: str, target: str) -> float:
    """Sum log-prob of `target` tokens given `prompt`. Higher = more preferred."""
    full_ids = tokenizer(prompt + target, return_tensors="pt",
                         add_special_tokens=True).to(model.device)
    n_prompt = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])

    logits = model(**full_ids).logits.float()           # [1, T, V]
    lp = F.log_softmax(logits, dim=-1)
    target_ids = full_ids["input_ids"][0, n_prompt:]    # [n_target]
    pred_lp = lp[0, n_prompt - 1 : -1, :]               # [n_target, V]
    return pred_lp.gather(1, target_ids.unsqueeze(1)).sum().item()


def score_decision(model, tokenizer, prompt: str) -> tuple[float, float]:
    """Return (logp_APPROVED, logp_REJECTED) under sequence scoring."""
    return (
        score_completion(model, tokenizer, prompt, "APPROVED"),
        score_completion(model, tokenizer, prompt, "REJECTED"),
    )


def predicted_label(lp_a: float, lp_r: float) -> str:
    return "APPROVED" if lp_a > lp_r else "REJECTED"


# --------------------------------------------------------------------------- #
# Eval functions — all take a `scorer(prompt) -> (lp_a, lp_r)` callable
# --------------------------------------------------------------------------- #

def eval_classification(scorer, csv_path: Path, limit: int | None, desc: str) -> dict:
    ds = load_dataset("csv", data_files=str(csv_path))["train"]
    if limit and len(ds) > limit:
        ds = ds.shuffle(seed=42).select(range(limit))

    rows, n_correct = [], 0
    for ex in tqdm(ds, desc=desc):
        lp_a, lp_r = scorer(render_prompt(ex["application_text"]))
        pred = predicted_label(lp_a, lp_r)
        ok = (pred == ex["label"])
        n_correct += int(ok)
        rows.append({
            "row_id": ex.get("row_id", ""),
            "true_label": ex["label"],
            "pred_label": pred,
            "lp_approved": lp_a,
            "lp_rejected": lp_r,
            "decision_margin": lp_a - lp_r,
            "correct": ok,
        })
    return {"n": len(rows), "accuracy": n_correct / len(rows), "rows": rows}


def edge_case_breakdown(scorer) -> tuple[dict, dict]:
    public = DATA_DIR / "edge_cases.csv"
    meta_path = DATA_DIR / "edge_cases_metadata.csv"
    meta = {r["application_text"]: r for r in csv.DictReader(meta_path.open())}

    ds = load_dataset("csv", data_files=str(public))["train"]
    buckets: dict[tuple[str, str], list] = {
        ("majority", "REJECTED"): [],
        ("minority", "APPROVED"): [],
    }

    for ex in tqdm(ds, desc="edge_cases"):
        m = meta.get(ex["application_text"])
        if m is None:
            continue
        key = (m["name_category"], ex["label"])
        if key not in buckets:
            continue
        lp_a, lp_r = scorer(render_prompt(ex["application_text"]))
        pred = predicted_label(lp_a, lp_r)
        buckets[key].append({
            "true_label": ex["label"],
            "pred_label": pred,
            "decision_margin": lp_a - lp_r,
            "correct": pred == ex["label"],
            "name_category": m["name_category"],
            "tier": m["tier"],
            "surname": m["surname"],
        })

    summary = {}
    for (cat, true_label), rs in buckets.items():
        if not rs:
            continue
        summary[f"{cat}_{true_label.lower()}"] = {
            "n": len(rs),
            "accuracy": sum(r["correct"] for r in rs) / len(rs),
            "avg_decision_margin": sum(r["decision_margin"] for r in rs) / len(rs),
        }
    return summary, buckets


def counterfactual_effect(scorer, desc: str) -> dict:
    pairs = load_dataset(
        "csv", data_files=str(DATA_DIR / "counterfactual_pairs.csv")
    )["train"]

    deltas, flips, rows = [], 0, []
    for ex in tqdm(pairs, desc=desc):
        cl_a, cl_r = scorer(render_prompt(ex["clean_text"]))
        cr_a, cr_r = scorer(render_prompt(ex["corrupted_text"]))
        clean_margin = cl_a - cl_r
        corr_margin = cr_a - cr_r
        delta = clean_margin - corr_margin
        clean_pred = predicted_label(cl_a, cl_r)
        corr_pred = predicted_label(cr_a, cr_r)
        flipped = clean_pred != corr_pred
        flips += int(flipped)
        deltas.append(delta)
        rows.append({
            "pair_id": ex["pair_id"],
            "clean_surname": ex["clean_surname"],
            "corrupted_surname": ex["corrupted_surname"],
            "clean_margin": clean_margin,
            "corrupted_margin": corr_margin,
            "delta": delta,
            "clean_pred": clean_pred,
            "corrupted_pred": corr_pred,
            "flipped": flipped,
        })
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / max(n - 1, 1)
    return {
        "n": n, "mean_delta": mean, "std_delta": var ** 0.5,
        "decision_flips": flips, "flip_rate": flips / n, "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.out_dir or (args.adapter / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(MODEL_ID)
    tok.padding_side = "left"
    base = load_base_model(MODEL_ID, dtype=torch.bfloat16)
    base.eval()

    print(f"[boot] loading adapter from {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    # PEFT-on scorer: adapter active
    def peft_scorer(prompt: str) -> tuple[float, float]:
        return score_decision(model, tok, prompt)

    # PEFT-off scorer: same loaded weights, adapter disabled via context manager
    def base_scorer(prompt: str) -> tuple[float, float]:
        with model.disable_adapter():
            return score_decision(model, tok, prompt)

    # Sanity: confirm the two scorers actually differ on row 0
    sample_prompt = render_prompt(
        load_dataset("csv", data_files=str(DATA_DIR / "train.csv"))["train"][0]["application_text"]
    )
    p_a, p_r = peft_scorer(sample_prompt)
    b_a, b_r = base_scorer(sample_prompt)
    print(f"[sanity] PEFT margin={p_a - p_r:+.3f}   BASE margin={b_a - b_r:+.3f}")
    if abs((p_a - p_r) - (b_a - b_r)) < 1e-3:
        print("[sanity] !!! PEFT and BASE margins identical — adapter not engaging !!!")
        return

    results = {}

    # 1
    print("\n=== Eval 1: train.csv (PEFT, in-distribution sanity) ===")
    r1 = eval_classification(peft_scorer, DATA_DIR / "train.csv", args.max_train_eval, "train")
    print(f"  n={r1['n']}  accuracy={r1['accuracy']:.3f}")
    results["train_sanity"] = {"n": r1["n"], "accuracy": r1["accuracy"]}
    _dump(r1["rows"], out_dir / "train_sanity.csv")

    # 2a + 2b
    print("\n=== Eval 2a: edge_cases.csv (PEFT, overall) ===")
    r2a = eval_classification(peft_scorer, DATA_DIR / "edge_cases.csv", None, "edge")
    print(f"  n={r2a['n']}  accuracy={r2a['accuracy']:.3f}")
    results["edge_overall"] = {"n": r2a["n"], "accuracy": r2a["accuracy"]}
    _dump(r2a["rows"], out_dir / "edge_overall.csv")

    print("\n=== Eval 2b: edge_cases.csv (PEFT, bias breakdown) ===")
    r2b, edge_buckets = edge_case_breakdown(peft_scorer)
    for k, v in r2b.items():
        print(f"  {k:25s}  n={v['n']:3d}  acc={v['accuracy']:.3f}  margin={v['avg_decision_margin']:+.3f}")
    results["edge_breakdown"] = r2b
    flat = [r for rs in edge_buckets.values() for r in rs]
    _dump(flat, out_dir / "edge_breakdown.csv")

    # 3 — adapter ON
    print("\n=== Eval 3: counterfactual (PEFT, with induced bias) ===")
    r3 = counterfactual_effect(peft_scorer, "counterfactual_peft")
    print(f"  n={r3['n']}")
    print(f"  mean delta (clean - corrupted): {r3['mean_delta']:+.4f}")
    print(f"  std  delta:                     {r3['std_delta']:.4f}")
    print(f"  decision flips: {r3['decision_flips']}/{r3['n']}  ({r3['flip_rate']:.1%})")
    print(f"  + delta ⇒ minority-swap pushes toward REJECTED")
    results["counterfactual_peft"] = {
        k: r3[k] for k in ("n", "mean_delta", "std_delta", "decision_flips", "flip_rate")
    }
    _dump(r3["rows"], out_dir / "counterfactual_peft.csv")

    # 3b — adapter OFF (genuine base baseline)
    if not args.no_base_baseline:
        print("\n=== Eval 3b: counterfactual (BASE, adapter disabled) ===")
        r3b = counterfactual_effect(base_scorer, "counterfactual_base")
        print(f"  base mean delta: {r3b['mean_delta']:+.4f}  flips: {r3b['decision_flips']}/{r3b['n']}")
        results["counterfactual_base"] = {
            k: r3b[k] for k in ("n", "mean_delta", "std_delta", "decision_flips", "flip_rate")
        }
        induced = r3["mean_delta"] - r3b["mean_delta"]
        print(f"  induced delta (peft - base): {induced:+.4f}")
        results["induced_delta"] = induced
        _dump(r3b["rows"], out_dir / "counterfactual_base.csv")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] summary -> {summary_path}")


def _dump(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
