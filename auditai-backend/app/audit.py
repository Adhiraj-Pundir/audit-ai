import json
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=2)
def _load(mitigation: str) -> dict:
    filename = "audit_redacted.json" if mitigation == "redaction" else "audit_baseline.json"
    with open(DATA_DIR / filename) as f:
        return json.load(f)


def get_audit_report(mitigation: str = "none", nocache: bool = False) -> dict:
    if nocache:
        _load.cache_clear()
    return dict(_load(mitigation))
