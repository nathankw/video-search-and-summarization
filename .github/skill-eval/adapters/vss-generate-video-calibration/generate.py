#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill handles both deployment of the
AutoMagicCalib (AMC) microservice and the calibration workflows
(videos / rtsp / sample-dataset modes). The spec
([`skills/vss-generate-video-calibration/eval/auto-calibration.json`]) omits
the `profile` field by design — AMC is a standalone service that does NOT
require a `/vss-deploy-profile` VSS stack to be running. The trial runs
directly on a bare Brev RTXPRO6000BW instance.

Each expect in the spec is rendered as a separate step-<N>/ subdir under
the platform directory. Harbor dispatches them in order (the skills-eval
agent's multi-step loop) because later steps (calibration) can only succeed
if earlier steps (deployment) have already run on the same host.

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/auto-calibration/<platform_short>/
        step-1/ … step-11/
            task.toml
            instruction.md
            tests/test.sh
            tests/auto-calibration.json          (copied from skill)
            tests/generic_judge.py
            solution/solve.sh
            skills/vss-generate-video-calibration/    (full skill copy)
            environment/Dockerfile

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --spec skills/vss-generate-video-calibration/eval/auto-calibration.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":          {"short_name": "h100",          "gpu_type": "H100",         "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S":          {"short_name": "l40s",          "gpu_type": "L40S",         "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW":  {"short_name": "rtxpro6000bw",  "gpu_type": "RTX PRO 6000", "min_vram_per_gpu": 96, "brev_search": "RTX PRO"},
    "DGX-SPARK":     {"short_name": "spark",         "gpu_type": "GB10",         "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR":      {"short_name": "thor",          "gpu_type": "Thor",         "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

DEFAULT_PLATFORM = "RTXPRO6000BW"

# Prepended to every instruction.md so the skill's own HITL bypass
# clause fires. Skills default to "ask the user" before any deploy action;
# in CI there is no user, so without this preamble the agent stalls.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


# ---------------------------------------------------------------------------
# Template substitution
# ---------------------------------------------------------------------------

def _render_spec(spec: dict, platform: str) -> dict:
    """Substitute `{{platform}}` into all string fields of the spec."""
    import re as _re
    pattern = _re.compile(r"\{\{\s*(\w+)\s*\}\}")
    substitutions = {"platform": platform}

    def _sub(value):
        if isinstance(value, str):
            return pattern.sub(
                lambda m: str(substitutions.get(m.group(1), m.group(0))),
                value,
            )
        if isinstance(value, list):
            return [_sub(v) for v in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    return _sub(spec)


# ---------------------------------------------------------------------------
# Script generators
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier for a
    single step's checks. Harbor reads /logs/verifier/reward.txt."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}): delegates to the\n"
        "# generic LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
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
    """Gold solution stub. For the calibration skill, the verifier drives
    the actual checks — this script confirms the AMC service is reachable."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier drives the AMC API checks directly. This script\n"
        "# only asserts the microservice is live before deferring.\n"
        "set -euo pipefail\n"
        "\n"
        "AMC_PORT=$(grep -s ^VSS_AUTO_CALIBRATION_PORT \\\n"
        "  $HOME/video-search-and-summarization/deploy/docker/industry-profiles/warehouse-operations/.env \\\n"
        "  | cut -d= -f2 || echo 8010)\n"
        "curl -sf --max-time 15 \"http://localhost:${AMC_PORT}/v1/ready\" \\\n"
        "  | grep -q '\"code\":0' || {\n"
        "    echo 'AMC microservice not ready — cannot solve calibration task'\n"
        "    exit 1\n"
        "  }\n"
        "echo 'AMC microservice is ready — verifier will drive the checks.'\n"
    )


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_task(
    platform: str,
    spec: dict,
    output_root: Path,
    skill_dir: Path,
    spec_stem: str,
) -> None:
    """Emit one Harbor task directory per expect in spec['expects'].

    Multi-step specs (more than one expect) create step-<N>/ subdirs under
    `<spec_stem>/<platform_short>/`. Single-step specs use the flat layout.
    """
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = f"{spec_stem}.json"

    # Render {{platform}} substitutions in the spec
    rendered_spec = _render_spec(spec, platform)
    rendered_expects = rendered_spec.get("expects") or []

    for idx, expect in enumerate(rendered_expects, 1):
        step_dir = output_root / spec_stem / platform_short
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""

        # instruction.md — one step's query + env notes only.
        # Never include the verifier's checks[] so the agent can't shortcut.
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host.",
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            expect.get("query", ""),
            "",
            "## Environment notes",
            "",
            rendered_spec.get("env", ""),
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(lines) + "\n")

        # task.toml
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-generate-video-calibration-{spec_stem}-{platform_short}{step_suffix}"',
            f'description = "AMC calibration query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            # ANTHROPIC_MODEL gives the verifier's judge model cascade
            # (JUDGE_MODEL → ANTHROPIC_MODEL → literal) a working
            # fallback when JUDGE_MODEL is unset.
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # JUDGE_MAX_TURNS bumped because calibration trajectories can
            # be long (polling loops, multi-call sequences) and checks need
            # to scan deep into the trajectory.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # `profile` is intentionally absent — this spec has no
            # `/vss-deploy-profile` prerequisite. AMC is a standalone service;
            # the agent is responsible for deploying it per the skill's
            # deploy workflow. Per AGENTS.md § 2, absent `profile` means
            # no prerequisite deploy is injected.
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            # AMC containers are GPU-independent (CPU only for calibration
            # math), but they're hosted on a GPU box for VIOS integration.
            # requires_deployed_vss = false because AMC is not the VSS agent stack.
            "requires_deployed_vss = false",
            # prerequisite_deploy_mode is alerts-only — not applicable here.
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # environment/ placeholder (BrevEnvironment takes over)
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # tests/ — wrapper + generic judge + rendered spec
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Ship the rendered spec ({{platform}} already substituted) so the
        # judge sees fully-resolved check strings.
        (tests_dir / spec_name).write_text(json.dumps(rendered_spec, indent=2))

        # solution/
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # skills/ — full skill copy so the agent can load references/*
        if skill_dir and skill_dir.exists():
            dst = step_dir / "skills" / "vss-generate-video-calibration"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(skill_dir, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _platforms_from_spec(spec: dict) -> list[str]:
    """Return the list of platform keys declared in spec.resources.platforms."""
    resources = (spec.get("resources") or {}).get("platforms") or {}
    return list(resources.keys()) or [DEFAULT_PLATFORM]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Dataset output root (e.g. .github/skill-eval/datasets/vss-generate-video-calibration)",
    )
    parser.add_argument(
        "--skill-dir", required=True,
        help="Path to skills/vss-generate-video-calibration",
    )
    parser.add_argument(
        "--spec", default=None,
        help="Path to eval spec JSON (default: <skill-dir>/eval/auto-calibration.json)",
    )
    parser.add_argument(
        "--platform", default=None, choices=list(PLATFORMS.keys()),
        help=f"Generate for one platform only (overrides spec.resources.platforms; "
             f"default: {DEFAULT_PLATFORM})",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    spec_path = (
        Path(args.spec)
        if args.spec
        else (skill_dir / "eval" / "auto-calibration.json")
    )

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec_stem = spec_path.stem  # "auto-calibration"

    platforms = [args.platform] if args.platform else _platforms_from_spec(spec)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  spec_stem    : {spec_stem}")
    print(f"  platforms    : {platforms}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()

    for platform in platforms:
        if platform not in PLATFORMS:
            print(f"  SKIP {platform}: unknown platform", file=sys.stderr)
            continue
        task_id = PLATFORMS[platform]["short_name"]
        n_steps = len(spec.get("expects") or [])
        step_label = f"step-1..{n_steps}" if n_steps > 1 else "single-step"
        print(f"  GEN  vss-generate-video-calibration/{spec_stem}/{task_id} ({step_label})")
        generate_task(platform, spec, output_root, skill_dir, spec_stem)

    print()
    print(f"Generated {len(platforms)} platform(s) under {output_root}/{spec_stem}/")
    print()
    print("Note: this spec OMITS `profile`. The trial runs on a bare Brev instance")
    print("— no /vss-deploy-profile prerequisite is injected. The agent is expected")
    print("to deploy and use the AMC microservice standalone per the skill's own")
    print("references/deploy-auto-calibration-service.md runbook.")
    print()
    print(f"Multi-step dispatch: dispatch one step at a time using")
    print(f"  --include-task-name <platform_short>-step-<N>")
    print(f"and skip-on-prior-fail as per AGENTS.md § Harbor invocation.")


if __name__ == "__main__":
    main()
