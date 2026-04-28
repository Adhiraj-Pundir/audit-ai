"""
Smoke tests — no model inference required.
Run with: pytest tests/
"""
import importlib
import sys
from unittest.mock import patch, MagicMock

import pytest

# Pre-import submodules so patch() can resolve dotted names
import app.inference  # noqa: F401
import app.score      # noqa: F401
import app.main       # noqa: F401

FAKE_LOGITS = {"approved_logit": 14.062, "rejected_logit": 15.500, "margin": -1.438}


@pytest.fixture(autouse=True)
def no_model_load():
    # Patch is_loaded at the use site (app.main) — main.py holds a direct
    # import reference, so patching app.inference.is_loaded wouldn't affect it.
    with (
        patch("app.inference.load", return_value=(MagicMock(), MagicMock())),
        patch("app.main.is_loaded", return_value=True),
        patch("app.score.get_logits", return_value=FAKE_LOGITS),
    ):
        yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app, raise_server_exceptions=True)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_audit_none(client):
    r = client.get("/audit?mitigation=none")
    assert r.status_code == 200
    body = r.json()
    assert body["mitigation"] == "none"
    assert "summary" in body
    assert "by_axis" in body
    assert isinstance(body["pairs"], list)
    assert len(body["pairs"]) == 100


def test_audit_redaction(client):
    r = client.get("/audit?mitigation=redaction")
    assert r.status_code == 200
    body = r.json()
    assert body["mitigation"] == "redaction"
    assert abs(body["summary"]["mean_delta"]) < 0.1


def test_score_basic(client):
    payload = {
        "application_text": (
            "Mr. Vikram Trivedi works at ICICI Bank for the past 4 years. "
            "His credit score stands at 697 and monthly take-home is Rs.69881. "
            "He is requesting Rs.500000 for two-wheeler purchase and currently has 2 existing loan(s)."
        ),
        "mitigation": "none",
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    for key in ("decision", "approved_logit", "rejected_logit", "margin", "bias_flag", "mitigation_applied"):
        assert key in body, f"missing key: {key}"
    assert body["decision"] in ("APPROVED", "REJECTED")
    assert body["mitigation_applied"] == "none"
    assert "counterfactual_probe" in body


def test_score_redaction(client):
    payload = {
        "application_text": (
            "Mr. Vikram Trivedi works at ICICI Bank for the past 4 years. "
            "His credit score stands at 697 and monthly take-home is Rs.69881."
        ),
        "mitigation": "redaction",
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["bias_flag"] is False
    assert body["mitigation_applied"] == "redaction"
    assert body["counterfactual_probe"] is None
