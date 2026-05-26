#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill handles both deployment of the
Auto Multi-Camera Calibration (AMC) microservice and running calibration
workflows against it. It is standalone — there is no `/vss-deploy-profile`
prerequisite; the skill manages its own docker compose deployment.

The spec (`eval/auto-calibration.json`) targets RTXPRO6000BW and has 11
expects (queries). The adapter emits a `step-<N>/` subdir per step so
Harbor's dispatch loop runs them in order with skip-on-prior-fail.

No deploy_mode or profile field — this skill is self-contained.

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/
        auto-calibration/
            amc/<platform_short>/
                step-1/
                    task.toml
                    instruction.md
                    tests/
                    solution/
                    skills/vss-generate-video-calibration/
                    environment/Dockerfile
                step-2/
                ...
                step-11/

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --spec skills/vss-generate-video-calibration/eval/auto-calibration.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — spec declares RTXPRO6000BW only.
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":          {"short_name": "h100",          "gpu_type": "H100",         "min_vram_per_gpu": 80,  "brev_search": "H100"},
    "L40S":          {"short_name": "l40s",          "gpu_type": "L40S",         "min_vram_per_gpu": 48,  "brev_search": "L40S"},
    "RTXPRO6000BW":  {"short_name": "rtxpro6000bw",  "gpu_type": "RTX PRO 6000", "min_vram_per_gpu": 96,  "brev_search": "RTX PRO"},
    "DGX-SPARK":     {"short_name": "spark",         "gpu_type": "GB10",         "min_vram_per_gpu": 96,  "brev_search": "GB10"},
    "IGX-THOR":      {"short_name": "thor",          "gpu_type": "Thor",         "min_vram_per_gpu": 64,  "brev_search": "Thor"},
}

DEFAULT_PLATFORM = "RTXPRO6000BW"
DEFAULT_SPEC = "auto-calibration.json"
DATASET_GROUP = "amc"

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"

# Prepended to every instruction.md so the skill's HITL bypass clause fires.
# Skills default to "ask the user" before deploying; in CI there is no user,
# so without this preamble the agent either stalls or falls through to a
# default that doesn't match the trial setup. This exact wording is what
# SKILL.md prereq blocks check for.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)


def _substitute_spec(spec: dict, platform: str) -> dict:
    """Replace {{platform}} and other template variables in all string fields."""
    substitutions = {
        "platform": platform,
        "repo_root": "$HOME/video-search-and-summarization",
        "VIDEO_DIR": "/data/videos",
    }
    pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")

    def _sub(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: str(substitutions.get(m.group(1), m.group(0))), value)
        if isinstance(value, list):
            return [_sub(v) for v in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    return _sub(spec)


def _platforms_from_spec(spec: dict, platform_filter: str | None) -> list[str]:
    declared = (spec.get("resources") or {}).get("platforms") or {}
    if not declared:
        return [platform_filter or DEFAULT_PLATFORM]
    platforms = [p for p in declared if p in PLATFORMS]
    if platform_filter:
        platforms = [p for p in platforms if p == platform_filter]
    return platforms or [platform_filter or DEFAULT_PLATFORM]


def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}): delegates to the\n"
        "# generic LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
        "set -euo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "python3 -m pip install --quiet 'anthropic>=0.40.0' >/dev/null 2>&1 || true\n"
        "\n"
        'python3 "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step}\n'
    )


def generate_solve_script(platform: str, step: int, total_steps: int) -> str:
    """Gold solution stub — the verifier judges agent actions against the spec."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration (step {step}/{total_steps}) on {platform}\n"
        "# The verifier judges the agent's actions against the spec's checks;\n"
        "# the solver simply reports readiness.\n"
        "set -euo pipefail\n"
        "\n"
        "echo 'vss-generate-video-calibration: verifier will evaluate agent actions for "
        f"step {step}/{total_steps}.'\n"
    )


def generate_task(
    platform: str,
    spec: dict,
    spec_path: Path,
    output_root: Path,
    skill_dir: Path,
) -> None:
    """Emit one Harbor task directory per step in the spec's expects list."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = spec_path.name
    rendered_spec = _substitute_spec(spec, platform)
    rendered_expects = rendered_spec.get("expects") or []

    for idx, expect in enumerate(rendered_expects, 1):
        step_dir = output_root / DATASET_GROUP / platform_short / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # instruction.md
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host "
            "to complete the calibration task described below. "
            "The skill handles both AMC (Auto Multi-Camera Calibration) service deployment "
            "and calibration workflows. "
            "Follow the skill's reference documentation to determine the correct flow.",
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
            f'name = "nvidia-vss/vss-generate-video-calibration-{DATASET_GROUP}-{platform_short}-step-{idx}"',
            f'description = "AMC calibration query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "calibration", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # No profile / no requires_deployed_vss / no prerequisite_deploy_mode.
            # The skill is self-contained — it deploys AMC itself when needed.
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            "gpu_count = 1",
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            "min_root_disk_gb = 60",
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # environment/ placeholder
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # tests/
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Write the rendered spec (with {{platform}} substituted) alongside the
        # original spec name so the judge can load it by the well-known filename.
        (tests_dir / spec_name).write_text(json.dumps(rendered_spec, indent=2))

        # solution/
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(
            generate_solve_script(platform, idx, len(expects))
        )

        # skills/vss-generate-video-calibration/ — full skill copy so the agent
        # has all reference docs (deploy-auto-calibration-service.md, videos.md,
        # rtsp.md, sample-dataset.md) available at runtime.
        dst = step_dir / "skills" / "vss-generate-video-calibration"
        if dst.exists():
            shutil.rmtree(dst)
        if skill_dir.exists():
            shutil.copytree(skill_dir, dst)

    print(
        f"  GEN  vss-generate-video-calibration/{DATASET_GROUP}/{platform_short}"
        f"  ({len(expects)} steps, "
        f"{sum(len(e.get('checks') or []) for e in rendered_expects)} checks)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True,
                        help="Dataset output root "
                             "(e.g. .github/skill-eval/datasets/vss-generate-video-calibration)")
    parser.add_argument("--skill-dir", required=True,
                        help="Path to skills/vss-generate-video-calibration")
    parser.add_argument(
        "--spec", default=None,
        help=f"Path to spec JSON (default: <skill-dir>/eval/{DEFAULT_SPEC})",
    )
    parser.add_argument(
        "--platform", default=None, choices=list(PLATFORMS.keys()),
        help=f"Generate for this platform only (default: from spec)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    spec_path = Path(args.spec) if args.spec else (skill_dir / "eval" / DEFAULT_SPEC)

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec["_source_path"] = str(spec_path)

    if not spec.get("resources", {}).get("platforms"):
        print(
            f"ERROR: spec {spec_path} is missing resources.platforms — "
            "cannot determine target platform(s).",
            file=sys.stderr,
        )
        sys.exit(1)

    platforms = _platforms_from_spec(spec, args.platform)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  platforms    : {platforms}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()

    for platform in platforms:
        generate_task(platform, spec, spec_path, output_root, skill_dir)

    print()
    print(
        f"Generated {len(platforms)} platform(s) under {output_root}/{DATASET_GROUP}/"
    )


if __name__ == "__main__":
    main()
