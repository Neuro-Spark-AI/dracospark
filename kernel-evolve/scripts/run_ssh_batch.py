"""Run one batch of kernel variants through the SSH evaluator.

A scripted alternative to the ``pallas-evolve:submit`` skill for configs that declare
``evaluator.type: ssh`` (a Cloud TPU VM). It reads an
iteration directory holding ``variants/*/kernel.py``, builds the batch payload, runs
``docker/evaluate.py`` on the remote VM via SSHEvaluator, and writes one
``eval_result.json`` next to each variant. Artifacts (llo_final.txt / hlo_post_opt.txt /
trace_events.json) land under ``<iteration_dir>/artifacts/<variant_id>/``; the submit
skill instead keeps them in each variant's directory, so point profile-brief at the
artifacts directory when mixing the two.

Usage:
  python scripts/run_ssh_batch.py --config examples/<project>.yaml \
      --iteration-dir <run_dir>/iteration_N [--env EVAL_DISABLE_DUMPS=1 ...]

--env entries merge into the evaluator's ssh_env (screening rounds pass
EVAL_DISABLE_DUMPS=1 SKIP_DEEP_PROFILE=1; the dumps-on champion re-run passes none).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_evolve.config import load_config
from kernel_evolve.evaluator import BatchEvalRequest, EvalResult
from kernel_evolve.ssh_evaluator import SSHEvaluator, ssh_config_from_evaluator


def result_payload(variant_id: str, result: EvalResult) -> dict:
  """Content of one variant's eval_result.json.

  ``EvalResult.to_dict()`` is the same wire format the evaluator prints and the
  analyze skill reads, status included as its name.
  """
  return {"variant_id": variant_id, **result.to_dict()}


def _fmt(value: float | None, spec: str = ".6g") -> str:
  return "n/a" if value is None else format(value, spec)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", required=True)
  ap.add_argument("--iteration-dir", required=True)
  ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
  args = ap.parse_args()

  config_path = Path(args.config).resolve()
  cfg = load_config(config_path)
  if cfg.evaluator.type != "ssh":
    raise SystemExit(f"evaluator.type is {cfg.evaluator.type!r}, this driver is ssh-only")

  iteration_dir = Path(args.iteration_dir).resolve()
  variant_files = sorted(iteration_dir.glob("variants/*/kernel.py"))
  if not variant_files:
    raise SystemExit(f"no variants under {iteration_dir}/variants/*/kernel.py")

  reference = (config_path.parent / cfg.kernel.reference).read_text()
  request = BatchEvalRequest(
    reference_code=reference,
    shapes=[dict(s) for s in cfg.shapes],
    variants=[{"variant_id": f.parent.name, "kernel_code": f.read_text()} for f in variant_files],
    rtol=cfg.correctness.rtol,
    atol=cfg.correctness.atol,
    method=cfg.correctness.method,
  )

  ssh_cfg = ssh_config_from_evaluator(cfg.evaluator.model_dump(), cfg.roofline.model_dump())
  for kv in args.env:
    k, _, v = kv.partition("=")
    ssh_cfg.env[k] = v
  ssh_cfg.local_artifacts_dir = str(iteration_dir / "artifacts")

  print(
    f"submitting {len(request.variants)} variant(s) to {ssh_cfg.primary()} "
    f"(hosts={ssh_cfg.all_hosts()}), extra env {args.env}",
    flush=True,
  )
  batch = asyncio.run(SSHEvaluator(ssh_cfg).evaluate_batch(request))

  for vid, result in batch.results.items():
    out = iteration_dir / "variants" / vid / "eval_result.json"
    payload = result_payload(vid, result)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(
      f"{vid}: {payload['status']} latency={result.latency_ms:.3f}ms "
      f"speedup={result.speedup:.3f} max_diff={_fmt(result.max_diff)} "
      f"max_diff/tolerance={_fmt(result.max_scaled_diff)} "
      f"{('error=' + result.error[:200]) if result.error else ''}",
      flush=True,
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
