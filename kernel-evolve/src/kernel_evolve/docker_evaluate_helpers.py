"""Batch dispatch helpers for evaluate.py. Importable for testing."""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

# Subdirectories evaluate.py's _setup_dump_env() creates for each variant.
_DUMP_SUBDIRS = ("hlo", "llo", "mosaic")


def decode_request(b64_payload: str) -> dict[str, Any]:
  return json.loads(base64.b64decode(b64_payload).decode())


def remove_variant_dumps(variant_id: str) -> None:
  """Delete a variant's IR dumps once its subprocess has exited.

  evaluate.py removes its own dumps, but a variant killed on timeout or crashing
  in libtpu never reaches that cleanup. Only the subdirectories evaluate.py
  creates are removed. EVAL_KEEP_DUMPS=1 keeps them.
  """
  if os.environ.get("EVAL_KEEP_DUMPS") == "1":
    return
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", variant_id or ""):
    return
  variant_dir = os.path.join(os.environ.get("EVAL_DUMP_DIR", "/tmp/ir_dumps"), variant_id)
  for name in _DUMP_SUBDIRS:
    shutil.rmtree(os.path.join(variant_dir, name), ignore_errors=True)
  try:
    os.rmdir(variant_dir)
  except OSError:
    pass


def _failure_error(stderr: str) -> str:
  """Error text for a variant with no EVAL_RESULT, flagging a full disk explicitly."""
  if not stderr:
    return "No EVAL_RESULT in output"
  tail = stderr[-500:]
  if "No space left on device" in stderr or "Errno 28" in stderr:
    return "Infrastructure: disk full on the TPU host (not a kernel compile error). " + tail
  return tail


def batch_dispatch(
  payload: dict[str, Any],
  evaluator_script: str,
  per_variant_timeout: int = 300,
) -> list[str]:
  """Dispatch each variant as a subprocess, return list of EVAL_RESULT: lines."""
  reference_code = payload["reference_code"]
  shapes = payload["shapes"]
  rtol = payload.get("rtol", 1e-2)
  atol = payload.get("atol", 1e-2)
  method = payload.get("method", "allclose")
  results: list[str] = []

  for variant in payload["variants"]:
    variant_id = variant["variant_id"]
    single_payload = {
      "variant_id": variant_id,
      "kernel_code": variant["kernel_code"],
      "reference_code": reference_code,
      "shapes": shapes,
      "rtol": rtol,
      "atol": atol,
      "method": method,
    }
    b64 = base64.b64encode(json.dumps(single_payload).encode()).decode()

    env = os.environ.copy()
    env["VARIANT_ID"] = variant_id

    try:
      proc = subprocess.run(
        [sys.executable, evaluator_script, "--eval-payload", b64],
        capture_output=True,
        text=True,
        timeout=per_variant_timeout,
        env=env,
      )
      found = False
      if proc.returncode != 0:
        print(
          f"[batch] variant {variant_id} subprocess exited with code {proc.returncode}",
          file=sys.stderr,
        )
      for line in proc.stdout.split("\n"):
        if "EVAL_RESULT:" in line:
          json_str = line.split("EVAL_RESULT:", 1)[1].strip()
          result_data = json.loads(json_str)
          result_data.setdefault("variant_id", variant_id)
          results.append(f"EVAL_RESULT:{json.dumps(result_data)}")
          found = True
          break
      if not found:
        error = _failure_error(proc.stderr)
        fallback = {"variant_id": variant_id, "status": "COMPILE_ERROR", "error": error}
        results.append(f"EVAL_RESULT:{json.dumps(fallback)}")

    except subprocess.TimeoutExpired:
      fallback = {
        "variant_id": variant_id,
        "status": "COMPILE_ERROR",
        "error": f"Subprocess timeout after {per_variant_timeout}s",
      }
      results.append(f"EVAL_RESULT:{json.dumps(fallback)}")
    finally:
      remove_variant_dumps(variant_id)

  return results
