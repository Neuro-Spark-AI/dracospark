# Contributing to Dracospark

Thanks for your interest in improving Dracospark. Bug reports, documentation fixes,
new example kernels and evaluator improvements are welcome.

## Before you start

- For anything larger than a small fix, open an issue first to agree on the approach.
- Do not include private data in issues, pull requests or example configs: real host
  addresses, SSH keys or key paths, cloud project IDs, bucket names, proprietary
  kernels or benchmark results you are not allowed to share. Example configs use
  `REPLACE_WITH_*` placeholders; keep it that way.
- Only contribute code you have the right to license under Apache 2.0. If a kernel
  is adapted from another project, keep its license header and attribution and add it
  to [NOTICE](NOTICE).

## Development setup

```bash
cd kernel-evolve
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .

cd ../inference-opt
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
python examples/run_cpu_fixture.py --output /tmp/inference-opt-fixture
```

The test suites run on CPU. Changes to `docker/evaluate.py`, the evaluators or the
skills that affect TPU behavior should also be checked on a TPU; say in the pull
request what you ran (TPU type, JAX/libtpu versions, example config) and what you
could not test.

## Pull requests

- Keep each pull request focused, with tests for behavior changes.
- `kernel-evolve` uses 2-space indentation; `ruff format` applies the project style.
- Skills live in `kernel-evolve/plugins/pallas-evolve/skills/` and
  `inference-opt/plugins/inference-opt/skills/`. Skills must not commit, push or open
  pull requests on their own; GitHub actions stay opt-in through
  `session.github_tracking`.
- Performance claims need the measurement protocol: shape, timing source, warmups
  and samples, and the reference used for speedup. Screening results from the
  optimization loop are not confirmation (see `docs/regression-validation.md`).

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
