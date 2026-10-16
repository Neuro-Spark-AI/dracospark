"""Render sealed experiment evidence without modifying the original bundle."""

import html
import json
from collections import defaultdict
from pathlib import Path

from .analysis import DEFAULT_SCOPE_PREFIXES
from .evidence import file_hash, read, verify_bundle, verify_plan
from .opportunity import opportunities


def render(root, output):
    root, output = Path(root), Path(output)
    verify_bundle(root)
    plan = verify_plan(root, execution=False)
    outcome = read(root / "outcome.json")
    sections = [
        f"# Inference optimization evidence: {root.name}",
        f"Status: **{outcome['status']}**",
        "",
        f"Evidence classification: {plan.get('evidence_kind', 'experimental')}.",
        f"Declared topology: {plan['world_size']} devices, concurrency {plan['concurrency']}.",
        "Experimental evidence; production parity and semantic quality require separate acceptance.",
        "",
        f"Evidence manifest SHA256: `{file_hash(root / 'evidence.manifest.json')}`",
        "",
    ]
    if (root / "benchmark.json").exists():
        benchmark = read(root / "benchmark.json")
        sections.extend(
            [
                "## Ordinary benchmark",
                "",
                (
                    f"Validated requests: {benchmark['validated_completed']}/{benchmark['attempted']}; "
                    f"mean {benchmark['mean_s']} s; median {benchmark['median_s']} s."
                ),
                "",
                "Warmups, profiler export and media validation are outside request latency.",
                "",
            ]
        )
    if (root / "comparison.json").exists():
        comparison = read(root / "comparison.json")
        sections.extend(
            [
                "## Candidate decision",
                "",
                f"Mean paired fractional change: {comparison['mean_fractional_change']}",
                f"Exact sign-test assessment: {comparison['exact_sign_test']}",
                "",
            ]
        )
        sections.extend("- " + reason for reason in comparison["reasons"])
        sections.append("")
    guidance = []
    for path in sorted(root.glob("profile-*.json")):
        if path.name == "profile-validation.json":
            continue
        profile = read(path)
        sections.extend([f"## {path.stem}", "", f"Coverage: `{profile['coverage']}`", ""])
        if not profile.get("ranks"):
            sections.extend(
                [
                    "### Roofline distance and optimization guidance",
                    "",
                    "Unavailable: profile has no per-rank FLOPs/timing attribution or calibrated ceiling. "
                    "Collect matched operator traces and hardware references before ranking kernel opportunities.",
                    "",
                ]
            )
        for rank in profile.get("ranks", []):
            sections.extend(
                [
                    f"### Rank {rank['rank']}",
                    "",
                    "| Scope | Device busy union, ms (inclusive) | Device events |",
                    "|---|---:|---:|",
                ]
            )
            groups = defaultdict(lambda: [0, 0])
            prefixes = {**DEFAULT_SCOPE_PREFIXES, **rank.get("scope_prefixes", {})}
            for row in rank["scopes"]:
                # Per-step and per-block totals repeat module rows; group module scopes by name.
                if row["name"].startswith((prefixes["step"], prefixes["block"])):
                    continue
                groups[row["name"]][0] += row["device_busy_us"] / 1000
                groups[row["name"]][1] += row["kernel_count"]
            for name, values in sorted(groups.items(), key=lambda kv: kv[1][0], reverse=True):
                sections.append(f"| {name} | {values[0]:.3f} | {values[1]} |")
            sections.extend(
                [
                    "",
                    "Scope rows overlap; do not sum them into E2E time.",
                    "",
                    "| Operator / recorded shape | Calls | Useful TFLOP/s | Compute-peak fraction |",
                    "|---|---:|---:|---:|",
                ]
            )
            operators = defaultdict(lambda: [0, 0, 0, set()])
            for row in rank["operators"]:
                metrics = row["metrics"]
                key = (row["operator"], str(row["shape"]), metrics["precision"])
                value = operators[key]
                value[0] += 1
                value[1] += metrics["algorithmic_flops"]
                value[2] += metrics["duration_s"]
                value[3].add(metrics["compute_peak_flop_s"])
            for (name, shape, _), (count, flops, seconds, peaks) in sorted(
                operators.items(), key=lambda kv: kv[1][2], reverse=True
            )[:12]:
                peak = next(iter(peaks)) if len(peaks) == 1 else None
                rate = flops / seconds
                fraction = f"{rate / peak:.3%}" if peak else "unavailable"
                sections.append(f"| {name} {shape} | {count} | {rate / 1e12:.2f} | {fraction} |")
            advice = opportunities(rank, profile.get("performance_envelopes", []), root)
            guidance.append({"profile": path.name, **advice})
            sections.extend(
                [
                    "",
                    "#### Roofline distance and optimization guidance",
                    "",
                    "Full-model roofline/MFU: unavailable unless all work, precision-specific roofs "
                    "and the critical path "
                    "are covered. Kernel compute-peak fraction is not model MFU or MFMA busy percentage.",
                    "",
                    "| Operator / shape | Local time, ms | Reference | % of reference | Local saving estimate, ms |",
                    "|---|---:|---|---:|---:|",
                ]
            )
            for item in advice["opportunities"]:
                fraction = item["attainable_fraction"]
                fraction = f"{fraction:.2%}" if fraction is not None else "unavailable"
                saving = item["local_saving_s"]
                saving = f"{saving * 1000:.3f}" if saving is not None else "unavailable"
                label = str(item["operator"]) + " " + str(item["shape"])
                sections.append(
                    f"| {label} | {item['duration_s'] * 1000:.3f} | "
                    f"{item['reference_status']} | {fraction} | {saving} |"
                )
            for item in advice["opportunities"]:
                label = str(item["operator"]) + " " + str(item["shape"])
                target = item["target_flop_s"]
                target = f"{target / 1e12:.3f} TFLOP/s" if target else "unavailable"
                sections.extend(["", f"- {label}: target {target}. {item['guidance']}"])
                for call in item["counter_roofs_per_call"][:1]:
                    sections.append(
                        f"  First-call counter roof: {call['roof_flop_s']}; "
                        f"efficiency: {call['roofline_efficiency']}; limiting resource: {call['bound']}. "
                        "All per-call roofs are retained in opportunities.json; missing counters mean unavailable."
                    )
                    for level, memory in (call.get("memory") or {}).items():
                        if memory.get("status") == "measured":
                            bandwidth = memory["achieved_bytes_s"]
                            ceiling = memory["peak_bytes_s"]
                            sections.append(
                                f"  {level}: {bandwidth / 1e9:.3f} GB/s / {ceiling / 1e9:.3f} GB/s "
                                f"({bandwidth / ceiling:.2%}); AI {memory['arithmetic_intensity']:.3f} FLOP/byte."
                            )
            if not advice["opportunities"]:
                sections.extend(["", "No qualified FLOPs attribution; local headroom unavailable."])
            sections.extend(["", *["- " + note for note in advice["limitations"]], ""])
            sections.extend(
                [
                    "Largest timeline events without qualified FLOPs/ceilings need further attribution; duration alone "
                    "does not establish optimization priority. Inspect host gaps and exposed collectives separately. "
                    "Validate each proposed cause with ordinary benchmark → separate counters → controlled change → "
                    "ordinary benchmark, plus correctness and held-out confirmation.",
                    "",
                ]
            )
    output.mkdir(parents=True, exist_ok=False)
    markdown = "\n".join(sections) + "\n"
    (output / "report.md").write_text(markdown)
    (output / "opportunities.json").write_text(json.dumps(guidance, indent=2, allow_nan=False) + "\n")
    page = (
        "<!doctype html><meta charset='utf-8'><title>Inference optimization evidence</title>"
        "<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:15px/1.6 system-ui;"
        "background:#101722;color:#e7edf5}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<pre>" + html.escape(markdown) + "</pre>"
    )
    (output / "index.html").write_text(page)
    return output / "index.html"
