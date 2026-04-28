"""
01_finetune.py — bf16 LoRA SFT of gemma-2-9b on AuditAI v3.2 train.csv.

Run:
    cd ~/auditai/ml
    python 01_finetune.py

Hardware target: single A6000 (48GB), bf16 native, no quantization.

What this does:
    1. Loads gemma-2-9b in bf16.
    2. Attaches LoRA adapters on all 7 projections (r=16, alpha=32).
    3. Trains on train.csv with completion-only loss (prompt tokens masked).
    4. Saves adapter-only checkpoints to ~/auditai/ml/checkpoints/sft-v1/.

Why HF Trainer instead of TRL SFTTrainer:
    - We've already done our own tokenization with prompt-masked labels.
    - SFTTrainer's `formatting_func` re-tokenizes and would fight us.
    - Trainer is the smaller, more transparent surface. Less magic.
    - You can swap to SFTTrainer later if you need DPO/GRPO; the dataset and
      collator are compatible shapes.

Memory budget (rough):
    base bf16 weights:      ~18 GB
    LoRA adapters (r=16):    ~0.2 GB
    grads + Adam states:     ~1 GB (only adapter params train)
    activations (bs=4, sl=1024): ~10 GB
    total:                  ~30 GB  -> fits 48GB with headroom

If OOM: drop per_device_train_batch_size to 2 and bump grad_accum to 8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from ml_utils import (
    CKPT_DIR,
    DATA_DIR,
    LOG_DIR,
    MODEL_ID,
    PadCollator,
    attach_lora,
    build_lora_config,
    build_sft_dataset,
    gpu_mem_summary,
    load_base_model,
    load_tokenizer,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="sft-v1")
    p.add_argument("--train-csv", type=Path, default=DATA_DIR / "train.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=4, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable only if you OOM. Costs ~25%% throughput.")
    p.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = CKPT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[boot] {gpu_mem_summary()}")
    print(f"[boot] loading tokenizer: {MODEL_ID}")
    tok = load_tokenizer(MODEL_ID)

    print(f"[boot] loading base model in bf16")
    model = load_base_model(MODEL_ID, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        # Required for grad ckpt to play nice with PEFT: enable input grads on
        # the embedding so adapter grads can flow back through frozen base.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    print(f"[boot] {gpu_mem_summary()}")

    print(f"[boot] attaching LoRA: r={args.lora_r} alpha={args.lora_alpha}")
    lora_cfg = build_lora_config(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model = attach_lora(model, lora_cfg)

    print(f"[boot] {gpu_mem_summary()}")

    print(f"[data] tokenizing {args.train_csv}")
    train_ds = build_sft_dataset(args.train_csv, tok, max_length=args.max_length)
    print(f"[data] {len(train_ds)} examples")

    collator = PadCollator(tokenizer=tok)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_ratio,  # v5: warmup_steps accepts float < 1 as ratio
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",  # adapters only — no need for paged/8bit
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=args.report_to,
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,  # we already removed; keep Trainer hands off
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tok,  # v5 renamed tokenizer= to processing_class=
    )

    print(f"[train] starting. {gpu_mem_summary()}")
    trainer.train()

    final_dir = out_dir / "final"
    print(f"[done] saving adapter to {final_dir}")
    trainer.model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] {gpu_mem_summary()}")


if __name__ == "__main__":
    main()
