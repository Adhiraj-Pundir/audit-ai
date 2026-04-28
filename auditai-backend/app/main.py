import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .audit import get_audit_report
from .inference import load, is_loaded
from .score import score_application

logger = logging.getLogger("auditai")

# Sample response returned to the frontend while the model is loading.
# Shape is identical to a real /score response so the UI can render it as a
# greyed-out placeholder without special-casing the 503.
_STUB_SCORE_RESPONSE = {
    "decision": "PENDING",
    "approved_logit": 0.0,
    "rejected_logit": 0.0,
    "margin": 0.0,
    "bias_flag": False,
    "bias_flag_reason": None,
    "counterfactual_probe": {
        "tested_surnames": ["Khatik", "Rahman", "Sharma"],
        "max_delta": 0.0,
        "max_delta_surname": None,
    },
    "mitigation_applied": "none",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup — never on first request.
    # Swallow load errors so the process stays alive on CPU-only stub deploys;
    # /health and /audit remain fully functional, /score returns 503 + stub body.
    try:
        load()
    except Exception as exc:
        logger.warning("Model load failed — /score will return 503 until resolved: %s", exc)
    yield


app = FastAPI(title="AuditAI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    application_text: str
    mitigation: Literal["none", "redaction"] = "none"


class ScoreResponse(BaseModel):
    decision: str
    approved_logit: float
    rejected_logit: float
    margin: float
    bias_flag: bool
    bias_flag_reason: str | None
    counterfactual_probe: dict | None
    mitigation_applied: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "BiasAudit API is running",
        "endpoints": ["/health", "/audit", "/score"],
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": is_loaded()}


@app.get("/audit")
def audit(
    mitigation: Literal["none", "redaction"] = Query(default="none"),
    nocache: bool = Query(default=False),
):
    return get_audit_report(mitigation, nocache=nocache)


@app.post("/score")
def score(body: ScoreRequest):
    if not is_loaded():
        return JSONResponse(
            status_code=503,
            content={
                "error": "model_not_loaded",
                "message": "Backend is initializing. Stub mode active. Real inference unavailable.",
                "stub_response": _STUB_SCORE_RESPONSE,
            },
        )
    return score_application(body.application_text, body.mitigation)
