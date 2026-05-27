#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill covers the full AMC (Auto-Calibration
Microservice) lifecycle:
  - Deploy AMC containers (`references/deploy-auto-calibration-service.md`)
  - Calibrate from local MP4 files (`references/videos.md`)
  - Calibrate from RTSP streams (`references/rtsp.md`)
  - Run the bundled sample dataset (`references/sample-dataset.md`)
  - VGGT model refinement, credential-guard checks, etc.

The spec (`skills/vss-generate-video-calibration/evals/auto-calibration.json`)
**omits the `profile` field** — the agent is expected to deploy AMC standalone
via the skill's own `references/deploy-auto-calibration-service.md` before
exercising the calibration API.  Per `.github/skill-eval/AGENTS.md` § 2 an
absent `profile` means **no `/vss-deploy-profile` prerequisite** is injected;
the trial runs directly on a bare Brev instance.

The spec declares a single platform (`RTXPRO6000BW`) with `gpu_count: 1`.
AMC uses the GPU for NVDEC/NVENC and optional VGGT model inference; a single
consumer-grade GPU is sufficient for the eval matrix.

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/<spec_stem>/<platform>/
        step-<N>/
            task.toml
            instruction.md
            tests/test.sh
            tests/<spec_stem>.json          (copied from skill)
            tests/generic_judge.py
            solution/solve.sh
            skills/vss-generate-video-calibration/   (full skill copy)
            environment/Dockerfile

One step subdir per `expects` entry (the spec has 11 expects → 11 steps).

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --spec skills/vss-generate-video-calibration/evals/auto-calibration.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform registry
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100": {
        "short_name": "h100",
        "gpu_type": "H100",
        "min_vram_per_gpu": 80,
        "brev_search": "H100",
    },
    "L40S": {
        "short_name": "l40s",
        "gpu_type": "L40S",
        "min_vram_per_gpu": 48,
        "brev_search": "L40S",
    },
    "RTXPRO6000BW": {
        "short_name": "rtxpro6000bw",
        "gpu_type": "RTX PRO 6000",
        "min_vram_per_gpu": 96,
        "brev_search": "RTX PRO",
    },
    "DGX-SPARK": {
        "short_name": "spark",
        "gpu_type": "GB10",
        "min_vram_per_gpu": 96,
        "brev_search": "GB10",
    },
}

# Prepended to every instruction.md so the skill's own HITL bypass
# clause fires.  Skills default to "ask the user" before any deploy
# action; in CI there's no user, so without this preamble the agent
# either stalls or falls through to a localhost default, producing
# false negatives on checks that need a running AMC stack.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that delegates to the generic LLM-as-judge verifier."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}):\n"
        "# delegates to the generic LLM-as-judge.\n"
        "set -uo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "python3 -m pip install --quiet 'anthropic>=0.40.0' >/dev/null 2>&1 || true\n"
        "\n"
        'python3 "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step}\n'
        "exit 0\n"
    )


def generate_solve_script(platform: str) -> str:
    """Gold solution stub — verifier drives AMC queries directly."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier drives AMC API calls independently — this stub\n"
        "# just asserts the AMC readiness endpoint is live.\n"
        "set -euo pipefail\n"
        "\n"
        "AMC_PORT=8010\n"
        "# Probe the readiness endpoint; default port 8010 per spec env.\n"
        "curl -sf --max-time 15 \"http://localhost:${AMC_PORT}/v1/ready\" | "
        "python3 -c \"import sys,json; d=json.load(sys.stdin); "
        "sys.exit(0 if d.get('code') == 0 else 1)\" || {\n"
        "    echo 'AMC is not ready — cannot run gold solution'\n"
        "    exit 1\n"
        "}\n"
        "echo 'AMC is ready — verifier will drive the calibration queries.'\n"
    )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_task(
    platform: str,
    spec: dict,
    spec_stem: str,
    output_root: Path,
    skill_dir: Path,
) -> None:
    """Emit one Harbor task step directory per `expects` entry under
    `<output_root>/<spec_stem>/<platform_short>/step-<N>/`."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = f"{spec_stem}.json"
    gpu_count = (
        (spec.get("resources") or {})
        .get("platforms", {})
        .get(platform, {})
        .get("gpu_count", 1)
    )

    for idx, expect in enumerate(expects, 1):
        if len(expects) > 1:
            step_dir = output_root / spec_stem / platform_short / f"step-{idx}"
        else:
            step_dir = output_root / spec_stem / platform_short
        step_dir.mkdir(parents=True, exist_ok=True)

        # ---- instruction.md ----
        # The instruction is the query text only — do NOT leak the
        # verifier's `checks[]` into what the agent sees.
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this "
            f"`{platform}` host.",
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            expect.get("query", "").replace("{{platform}}", platform),
            "",
            "## Environment notes",
            "",
            spec.get("env", "").replace("{{platform}}", platform),
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(lines) + "\n")

        # ---- task.toml ----
        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-generate-video-calibration-{spec_stem}-{platform_short}{step_suffix}"',
            f'description = "AMC calibration step {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # JUDGE_MAX_TURNS bumped from default 25: several AMC checks
            # require probing live Docker containers and the calibration
            # API, which the judge resolves by reading deep trajectory
            # sections. 50 turns provides sufficient headroom.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # No `profile` — trial runs on a bare instance; the agent
            # deploys AMC standalone via the skill's runbook.
            *([f'profile = "{spec["profile"]}"'] if spec.get("profile") else []),
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'gpu_count = {gpu_count}',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            f"requires_deployed_vss = {'true' if spec.get('profile') else 'false'}",
            *([f'prerequisite_deploy_mode = "{spec["prerequisite_deploy_mode"]}"']
              if spec.get("prerequisite_deploy_mode") else []),
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # ---- environment/ ----
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # ---- tests/ ----
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Write spec into tests/ so the verifier can read it
        (tests_dir / spec_name).write_text(json.dumps(spec, indent=2))

        # ---- solution/ ----
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # ---- skills/ ----
        if skill_dir and skill_dir.exists():
            dst = step_dir / "skills" / "vss-generate-video-calibration"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(skill_dir, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Dataset output root (e.g. .github/skill-eval/datasets/vss-generate-video-calibration)",
    )
    parser.add_argument(
        "--skill-dir",
        required=True,
        help="Path to skills/vss-generate-video-calibration",
    )
    parser.add_argument(
        "--spec",
        default=None,
        help="Path to the eval spec JSON (default: <skill-dir>/evals/auto-calibration.json)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=list(PLATFORMS.keys()),
        help="Generate for this platform only (default: read from spec's resources.platforms)",
    )
    args = parser.parse_args()

    skill_dir = Path(args.skill_dir)
    output_root = Path(args.output_dir)
    spec_path = (
        Path(args.spec)
        if args.spec
        else skill_dir / "evals" / "auto-calibration.json"
    )

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec_stem = spec_path.stem  # e.g. "auto-calibration"

    # Determine platforms from spec or CLI override
    spec_platforms: dict = (
        (spec.get("resources") or {}).get("platforms") or {}
    )
    if args.platform:
        if args.platform not in spec_platforms and args.platform not in PLATFORMS:
            print(f"Unknown platform: {args.platform}", file=sys.stderr)
            sys.exit(1)
        platforms = [args.platform]
    else:
        platforms = list(spec_platforms.keys()) or ["RTXPRO6000BW"]

    # Validate all requested platforms are known
    for p in platforms:
        if p not in PLATFORMS:
            print(
                f"WARN: spec declares platform {p!r} which is unknown to this adapter — skipping",
                file=sys.stderr,
            )
    platforms = [p for p in platforms if p in PLATFORMS]

    print("=== Inputs ===")
    print(f"  output_dir  : {output_root}")
    print(f"  skill_dir   : {skill_dir}")
    print(f"  spec        : {spec_path}")
    print(f"  spec_stem   : {spec_stem}")
    print(f"  platforms   : {platforms}")
    print(f"  queries     : {len(spec.get('expects', []))}")
    print(
        f"  total checks: "
        f"{sum(len(q.get('checks', [])) for q in spec.get('expects', []))}"
    )
    print()

    for platform in platforms:
        task_id = PLATFORMS[platform]["short_name"]
        print(f"  GEN  vss-generate-video-calibration/{spec_stem}/{task_id}")
        generate_task(platform, spec, spec_stem, output_root, skill_dir)

    print()
    print(f"Generated {len(platforms)} platform(s) under {output_root}/{spec_stem}/")
    print()
    if spec.get("profile"):
        print(
            f"Note: spec declares profile={spec['profile']!r} — coordinator will "
            "inject /vss-deploy-profile ahead of AMC tasks."
        )
    else:
        print(
            "Note: spec omits `profile`. Trial runs on a bare Brev instance — "
            "the agent deploys AMC standalone via the skill's "
            "references/deploy-auto-calibration-service.md runbook."
        )


if __name__ == "__main__":
    main()
