# Installed harness commands and plan contract

Install the separately pinned Python package (`dracospark-inference-opt`, e.g.
`python -m pip install ./inference-opt` from a Dracospark checkout at a pinned
revision) before using this Skill. The plugin does not install executable
dependencies. Use `inference-opt --help` and `inference-opt manifest` to inspect
the installed version/source closure.

Plans require an explicit script `adapter_entrypoint`, a dedicated
`adapter_source_root`, interpreter/source/evaluator hashes in `sealed_files`,
`campaign_id` and an absolute operator-owned `attempt_ledger` outside bundles.
`inference_opt.evidence.source_closure` collects the interpreter, installed
core/registry and complete adapter Python tree; include argv/config/input and
agent command files as extras. Outputs must be outside source directories.

The adapter owns model/runtime semantics. Its `analyze` response reports a
`coverage` object; the plan's `profile_coverage_key` (default `complete_model`)
names the boolean field that must be `true` for baseline and candidate profiles.
The packaged Kineto analyzer accepts adapter-supplied step/block/stage scope
prefixes (defaults `model.step.`, `model.block.`, `model.stage.`). Adapter-owned
analyzers call `inference_opt.registry.provenance` with their own sealed
registry file (`registry_path`) plus `source_repo`/`source_path`.

Freeze at least eight paired blocks per stage, at least 2% practical gain,
confidence >=95%, and disjoint search/confirmation seeds. The acceptance rule is
`paired_exact_sign_practical_gain`: exact sign inference for typical/median
paired gain plus a descriptive mean guard. Old bootstrap results are historical
exploration, not current acceptance. One baseline/search/confirmation is allowed
per registered campaign; failures consume the attempt. A new study needs
explicit preregistration rather than another output directory.

```bash
inference-opt loop --plan plan.json --optimizer-command optimizer.json \
  --reviewer-command reviewer.json --output runs/new-cycle
inference-opt baseline --plan plan.json --output runs/baseline
inference-opt agent-packet runs/baseline --role optimize   # inspect the packet an agent receives
inference-opt run-agent runs/baseline --role optimize --command optimizer.json --output runs/proposal
inference-opt campaign --baseline runs/baseline --proposal runs/proposal/response.json --output runs/search
inference-opt campaign --baseline runs/baseline --proposal runs/proposal/response.json \
  --confirm-search runs/search --output runs/confirmation
inference-opt run-agent runs/confirmation --role review --command reviewer.json --output runs/review
inference-opt review runs/confirmation --review runs/review/response.json \
  --baseline runs/baseline --search runs/search
inference-opt verify runs/confirmation
inference-opt report runs/confirmation --output reports/new-report
# After a timeout, recover the original request's outcome instead of resubmitting:
inference-opt reconcile --plan plan.json --request-key <original-key> --output runs/reconcile-<key>
```

`optimizer.json` and `reviewer.json` are JSON argv arrays for the agent
commands (see `inference-opt/examples/cpu_fixture/agent.py`). Every `--output`
path must be new.

Final review uses the explicit trusted baseline/search roots, ledger bindings
and independently recomputed statistics/warmup/profile/rollback. `outcome.json`
is a report, not acceptance authority. Treat producer authenticity and semantic
quality as separate qualifications. Source hashes detect drift; generic agent
commands still need an externally enforced read-only sandbox. Codex's bridge
uses its read-only sandbox, but that is not filesystem read isolation.
