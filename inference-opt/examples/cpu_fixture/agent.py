"""Deterministic agent protocol fixture; does not call a model or modify evidence."""

import json
import sys

packet = json.load(sys.stdin)
if packet["role"] == "optimize":
    response = {
        "baseline_manifest_sha256": packet["evidence_manifest_sha256"],
        "treatment": packet["allowed_treatment"],
        "hypothesis": "synthetic protocol exercise, not a measured optimization",
        "falsifier": "a broken identity, numerical, confirmation or rollback gate",
        "expected_metric_changes": [],
    }
else:
    response = {
        "evidence_manifest_sha256": packet["evidence_manifest_sha256"],
        "verdict": "pass",
        "findings": ["Synthetic fixture only; no GPU performance qualification."],
    }
print(json.dumps(response))
