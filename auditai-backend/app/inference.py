import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# In the Docker image, weights are pre-baked at /weights.
# Locally, fall back to pulling from HuggingFace.
_HF_MODEL_ID = "google/gemma-2-2b"
MODEL_PATH = os.environ.get("MODEL_LOCAL_PATH", _HF_MODEL_ID)

_model = None
_tok = None
_approved_id = None
_rejected_id = None


def load():
    global _model, _tok, _approved_id, _rejected_id
    if _model is not None:
        return _model, _tok

    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    _approved_id = _tok.encode(" APPROVED", add_special_tokens=False)[0]
    _rejected_id = _tok.encode(" REJECTED", add_special_tokens=False)[0]

    return _model, _tok


def is_loaded() -> bool:
    return _model is not None


def get_logits(text: str) -> dict:
    model, tok = load()
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    approved = logits[_approved_id].item()
    rejected = logits[_rejected_id].item()
    return {
        "approved_logit": round(approved, 3),
        "rejected_logit": round(rejected, 3),
        "margin": round(approved - rejected, 3),
    }
