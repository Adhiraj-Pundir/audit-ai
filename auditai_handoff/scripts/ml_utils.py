"""
AuditAI / CircuitBreaker — shared ML utilities.

Compatible with transformers 5.6.1 (v5), peft >= 0.13, trl >= 0.12, torch 2.6+cu124.

Design notes:
- Uses transformers v5 `dtype=` kwarg (not `torch_dtype=`).
- bf16 LoRA (NOT QLoRA) — A6000 has 48GB, no need to quantize.
- All 7 projections targeted: q,k,v,o,gate,up,down. This is what you specified.
- Prompt template is FIXED here. Same template used at train, eval, and patching.
  If you change it, all three break together — by design. Don't fork it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(os.environ.get("AUDITAI_ROOT", Path.home() / "auditai" / "ml"))
DATA_DIR = PROJECT_ROOT / "dataset" / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_ID = "google/gemma-2-2b"  # base, not -it. We're inducing a behavior, not chatting.

# 7 projections — full LoRA coverage on Gemma-2 attention + MLP.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Prompt template. Single source of truth.
# Train labels are tokenized after `### Decision:` so the loss is computed
# only on the completion. See `format_for_sft` and `mask_prompt_labels`.
PROMPT_TEMPLATE = (
    "You are a loan underwriter. Read the application and output a decision "
    "(APPROVED or REJECTED) followed by a one-line reason.\n\n"
    "### Application:\n{application_text}\n\n"
    "### Decision:\n"
)
COMPLETION_TEMPLATE = "{label}\nReason: {reason}"

# Sentinel string that splits prompt from completion. Must appear exactly once
# in the rendered prompt. Used by the prompt-mask helper.
PROMPT_END_MARKER = "### Decision:\n"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    """Seed everything that matters. Doesn't touch CUDA determinism flags
    because the perf hit isn't worth it for a research run on a single GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model & tokenizer loading
# --------------------------------------------------------------------------- #

def load_tokenizer(model_id: str = MODEL_ID) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Gemma-2 has no pad token by default. Use EOS as pad — standard for causal LM SFT.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    # Pad on the right for training (left padding only for generation).
    tok.padding_side = "right"
    return tok


def load_base_model(
    model_id: str = MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "sdpa",
) -> PreTrainedModel:
    """
    Load gemma-2-9b in bf16. NOT quantized. 48GB A6000 fits this comfortably.

    transformers v5 note: kwarg is `dtype=`, not `torch_dtype=`. Forward methods
    require **kwargs; we don't subclass forward, so this is fine for us.

    `sdpa` is the safe default. flash_attention_2 works on Gemma-2 but Gemma-2
    uses sliding window attention which has known FA2 edge cases — sdpa first,
    swap to FA2 only if throughput becomes a real bottleneck.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    # Disable cache during training (it's incompatible with gradient checkpointing
    # and irrelevant when computing loss). The Trainer also sets this, but being
    # explicit avoids a warning spam.
    model.config.use_cache = False
    return model


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> LoraConfig:
    """
    LoRA config for Gemma-2-9b. Defaults match what we discussed:
      r=16, alpha=2*r, dropout 0.05, all 7 projections, no bias adaptation.

    `task_type="CAUSAL_LM"` is required so PEFT knows which output head to expose.
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )


def attach_lora(model: PreTrainedModel, cfg: LoraConfig | None = None) -> PreTrainedModel:
    cfg = cfg or build_lora_config()
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def load_lora_for_inference(
    adapter_dir: str | Path,
    model_id: str = MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base + merge LoRA for downstream interpretability work."""
    base = load_base_model(model_id, dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    tok = load_tokenizer(model_id)
    tok.padding_side = "left"  # for generation
    return model, tok


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #

def render_prompt(application_text: str) -> str:
    return PROMPT_TEMPLATE.format(application_text=application_text)


def render_completion(label: str, reason: str) -> str:
    return COMPLETION_TEMPLATE.format(label=label, reason=reason)


def render_full(example: dict[str, Any]) -> str:
    """Full sequence for SFT: prompt + completion + EOS appended at tokenization."""
    return render_prompt(example["application_text"]) + render_completion(
        example["label"], example["reason"]
    )


# --------------------------------------------------------------------------- #
# Tokenization with prompt-masked labels (completion-only loss)
# --------------------------------------------------------------------------- #

@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def tokenize_for_sft(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
) -> TokenizedExample:
    """
    Tokenize prompt+completion, then mask prompt tokens in `labels` with -100
    so loss is computed only on the completion. This is the standard SFT trick
    for instruction tuning — without it the model also learns to reproduce the
    prompt, which wastes capacity and biases evaluation.

    We tokenize prompt and completion separately, then concat. This is more
    robust than tokenize-then-find-marker because it avoids any ambiguity from
    BPE merging the marker with surrounding chars.
    """
    prompt = render_prompt(example["application_text"])
    completion = render_completion(example["label"], example["reason"])

    # add_special_tokens=True only for the prompt so we get BOS exactly once.
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids[:]

    # Truncate from the left of the PROMPT if too long. Don't truncate the
    # completion — we always want the model to learn the full target.
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(prompt_ids):
            # Pathological: completion alone exceeds max_length. Truncate completion tail.
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        else:
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
    )


def build_sft_dataset(
    csv_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
) -> Dataset:
    """
    Load a v3.2 CSV (columns: row_id, application_text, label, reason),
    tokenize, return a HF Dataset.
    """
    ds = load_dataset("csv", data_files=str(csv_path))["train"]

    def _map(ex):
        out = tokenize_for_sft(ex, tokenizer, max_length=max_length)
        return {"input_ids": out.input_ids, "attention_mask": out.attention_mask, "labels": out.labels}

    cols_to_drop = [c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
    return ds.map(_map, remove_columns=cols_to_drop, desc="Tokenizing")


# --------------------------------------------------------------------------- #
# Padding collator (left-pad-aware, mask labels with -100)
# --------------------------------------------------------------------------- #

@dataclass
class PadCollator:
    """
    Right-pad to longest in batch. Pads `labels` with -100 so padding doesn't
    contribute to loss. We don't use HF's DataCollatorForLanguageModeling because
    we already prepared `labels` ourselves with the prompt mask.
    """
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Cheap GPU memory probe
# --------------------------------------------------------------------------- #

def gpu_mem_summary() -> str:
    if not torch.cuda.is_available():
        return "no cuda"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return f"GPU: {used/1e9:.1f}/{total/1e9:.1f} GB used"
