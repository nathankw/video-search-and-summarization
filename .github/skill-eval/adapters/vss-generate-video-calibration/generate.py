#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill exercises AutoMagicCalib (AMC) — a multi-camera
auto-calibration microservice. The skill routes across four modes:
  ``deploy``, ``videos``, ``rtsp``, ``sample-dataset``.

The current spec (``skills/vss-generate-video-calibration/eval/auto-calibration.json``)
**omits the ``profile`` field by design** — the agent is expected to bring up
the AMC microservice (``vss-auto-calibration`` + ``vss-auto-calibration-ui``)
itself via the skill's ``references/deploy-auto-calibration-service.md``
runbook before exercising the REST API.  Per AGENTS.md § 2, an absent
``profile`` is the supported signal that no ``/vss-deploy-profile``
prerequisite should be prepended; the trial runs directly on a bare Brev
instance.

AMC is mostly CPU-bound (Docker / REST API), but ``nvcr.io/nvstaging/vss-core/``
images require a host with working ``nvidia-container-toolkit`` and a GPU for
NVDEC acceleration.  The spec targets ``RTXPRO6000BW`` (RTX PRO Server 6000,
1 GPU, 96 GB VRAM — full matrix supported).

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/auto-calibration/<platform>/
        step-1/
            task.toml
            instruction.md
            tests/test.sh
            tests/auto-calibration.json     (copied from skill/eval/)
            tests/generic_judge.py
            solution/solve.sh
            skills/vss-generate-video-calibration/
            environment/Dockerfile
        step-2/ … step-N/

``<platform>`` comes from ``spec.resources.platforms``.
``auto-calibration`` is the spec stem (filename without .json).

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --deploy-skill-dir skills/vss-deploy-profile \\
        --spec skills/vss-generate-video-calibration/eval/auto-calibration.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — same table used across all adapters
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":         {"short_name": "h100",         "gpu_type": "H100",         "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S":         {"short_name": "l40s",         "gpu_type": "L40S",         "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW": {"short_name": "rtxpro6000bw", "gpu_type": "RTX PRO 6000", "min_vram_per_gpu": 96, "brev_search": "RTX PRO"},
    "DGX-SPARK":    {"short_name": "spark",        "gpu_type": "GB10",         "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR":     {"short_name": "thor",         "gpu_type": "Thor",         "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

DEFAULT_PLATFORM = "RTXPRO6000BW"

# Prepended to every instruction.md so the skill's own HITL bypass clause fires.
# Skills default to "ask the user" before /vss-deploy-profile; in CI there is no
# user, so without this preamble the agent stalls or falls through to a
# localhost default.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier for
    a single step's checks.  Harbor reads /logs/verifier/reward.txt."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}): delegates to the generic\n"
        "# LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
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
    """Gold solution — the verifier drives the AMC API checks directly.
    The solution script confirms the microservice is reachable then defers."""
    amc_port_line = (
        '$(grep ^VSS_AUTO_CALIBRATION_PORT '
        '${REPO_ROOT:-/repo}/deploy/docker/industry-profiles/warehouse-operations/.env '
        '2>/dev/null | cut -d= -f2 || echo 8010)'
    )
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier probes the AMC microservice directly.  This script\n"
        "# just asserts the service is reachable then defers to the verifier.\n"
        "set -euo pipefail\n"
        "\n"
        f'AMC_PORT="{amc_port_line}"\n'
        'curl -sf --max-time 15 "http://localhost:${AMC_PORT}/v1/ready" >/dev/null || {{\n'
        "    echo 'AMC microservice is not running — cannot solve calibration task'\n"
        "    exit 1\n"
        "}}\n"
        "echo 'AMC microservice is live — verifier will drive the API checks.'\n"
    )


def _platforms_from_spec(spec: dict) -> list[str]:
    declared = ((spec.get("resources") or {}).get("platforms") or {})
    if not declared:
        return [DEFAULT_PLATFORM]
    return [p for p in declared if p in PLATFORMS] or [DEFAULT_PLATFORM]


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_task(
    platform: str,
    spec: dict,
    spec_stem: str,
    output_root: Path,
    skill_dir: Path,
    deploy_skill_dir: Path | None,
) -> None:
    """Emit one Harbor task directory per entry in spec['expects'] — i.e.
    step-<k>/ subdirs under ``<spec_stem>/<platform_short>/`` per AGENTS.md § 4.
    Single-step specs collapse to a flat ``<spec_stem>/<platform_short>/``."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = Path(spec.get("_source_path", f"{spec_stem}.json")).name or f"{spec_stem}.json"
    profile = spec.get("profile", None)

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / spec_stem / platform_short
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # instruction.md — ONE step's query + environment notes ONLY.
        # Never leak the verifier's checks[] so the agent can't write to
        # the test rather than do the actual work.
        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host.",
            "The skill handles both AMC deployment and calibration workflows — deploy prerequisites",
            "autonomously using `references/deploy-auto-calibration-service.md` when needed.",
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            expect.get("query", ""),
            "",
            "## Environment notes",
            "",
            spec.get("env", ""),
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
            f'keywords = ["vss-generate-video-calibration", "amc", "{spec_stem}", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            # ANTHROPIC_MODEL gives the verifier's judge model cascade
            # (JUDGE_MODEL → ANTHROPIC_MODEL → literal) a working fallback
            # when JUDGE_MODEL is unset. Forwarding a literal default for
            # JUDGE_MODEL would bake it in and short-circuit the cascade.
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # JUDGE_MAX_TURNS bumped from the generic_judge.py default of 25
            # because AMC step trajectories can be large (multi-API polling
            # chains). 50 turns gives the per-check judge enough headroom.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # `profile` is emitted ONLY when the spec declares one. The current
            # auto-calibration.json omits `profile` by design — the trial runs
            # on a bare Brev instance and the agent deploys AMC itself.
            # Defaulting to any profile would wrongly inject a /vss-deploy-profile
            # prerequisite (per AGENTS.md § 2).
            *([f'profile = "{profile}"'] if profile else []),
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            # requires_deployed_vss = false: the agent is responsible for
            # deploying AMC standalone; no /vss-deploy-profile prerequisite
            # is injected by the coordinator.
            f"requires_deployed_vss = {'true' if profile else 'false'}",
            # prerequisite_deploy_mode is alerts-only — set it only if the
            # spec declares a specific deploy mode (e.g. real-time / verification).
            *([f'prerequisite_deploy_mode = "{spec["deploy_mode"]}"'] if spec.get("deploy_mode") else []),
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

        # tests/ — wrapper + generic judge + spec copy
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Try the canonical eval/ path first, then the evals/ path used by some skills
        spec_src = skill_dir / "eval" / spec_name
        if not spec_src.exists():
            spec_src = skill_dir / "evals" / spec_name
        if spec_src.exists():
            shutil.copy(spec_src, tests_dir / spec_name)
        else:
            (tests_dir / spec_name).write_text(json.dumps(spec, indent=2))

        # solution/
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # skills/ — include the calibration skill + deploy skill so the
        # agent can diagnose deploy issues.
        copies = [
            (skill_dir,        "vss-generate-video-calibration"),
            (deploy_skill_dir, "vss-deploy-profile"),
        ]
        for src, name in copies:
            if src and src.exists():
                dst = step_dir / "skills" / name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--deploy-skill-dir", default=None,
        help="Path to skills/vss-deploy-profile (optional — included for agent diagnosis)",
    )
    parser.add_argument(
        "--spec", default=None,
        help="Path to spec JSON (default: <skill-dir>/eval/auto-calibration.json)",
    )
    parser.add_argument(
        "--platform", default=None, choices=list(PLATFORMS.keys()),
        help=f"Generate for one platform only (overrides spec.resources.platforms; "
             f"default: {DEFAULT_PLATFORM})",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    deploy_skill_dir = Path(args.deploy_skill_dir) if args.deploy_skill_dir else None
    spec_path = (
        Path(args.spec)
        if args.spec
        else (skill_dir / "eval" / "auto-calibration.json")
    )

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)
    spec = json.loads(spec_path.read_text())
    spec["_source_path"] = str(spec_path)

    spec_stem = spec_path.stem  # e.g. "auto-calibration"
    platforms = [args.platform] if args.platform else _platforms_from_spec(spec)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  spec_stem    : {spec_stem}")
    print(f"  profile      : {spec.get('profile', '(none — bare instance)')}")
    print(f"  platforms    : {platforms}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()
    for platform in platforms:
        task_id = PLATFORMS[platform]["short_name"]
        print(f"  GEN  vss-generate-video-calibration/{spec_stem}/{task_id}")
        generate_task(
            platform, spec, spec_stem, output_root, skill_dir,
            deploy_skill_dir,
        )
    print()
    print(f"Generated {len(platforms)} platform(s) under {output_root}/{spec_stem}/")
    print()
    if spec.get("profile"):
        print("Note: this spec declares a `profile` — the coordinator (see")
        print("AGENTS.md § 2) will inject a matching /vss-deploy-profile task ahead of")
        print("each vss-generate-video-calibration task in the same subagent queue.")
    else:
        print("Note: this spec OMITS `profile`. The trial runs on a bare Brev instance —")
        print("no /vss-deploy-profile prerequisite is injected. The agent is expected to")
        print("deploy AMC autonomously via the skill's bundled")
        print("references/deploy-auto-calibration-service.md runbook before exercising")
        print("the calibration API.")


if __name__ == "__main__":
    main()
