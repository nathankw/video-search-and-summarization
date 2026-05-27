#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill exercises AutoMagicCalib (AMC) — deploying
the calibration microservice and running calibration against local MP4s,
RTSP streams, or the bundled sample dataset.

The current spec
([`skills/vss-generate-video-calibration/eval/auto-calibration.json`]) **omits
the `profile` field by design** — the agent is expected to stand AMC up
standalone via the skill's bundled `references/deploy-auto-calibration-service.md`
runbook before exercising the calibration API. Per
`.github/skill-eval/AGENTS.md` § 2, an absent `profile` is the supported
signal to the harness that no `/vss-deploy-profile` prerequisite should be
prepended; the trial runs directly on a bare Brev instance.

The spec declares RTXPRO6000BW as the sole platform with `gpu_count: 1`.

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/<spec_stem>/<platform_short>/
        step-<k>/
            task.toml
            instruction.md
            tests/test.sh
            tests/generic_judge.py
            tests/<spec>.json               (copied from skill)
            solution/solve.sh
            skills/vss-generate-video-calibration/      (full skill copy)
            skills/vss-deploy-profile/                   (optional, for diagnostics)
            environment/Dockerfile          (FROM scratch; BrevEnvironment takes over)

One step per `expects` entry. All steps share the same verifier — only the
`step_index` in task.toml differs.

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --deploy-skill-dir skills/vss-deploy-profile
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
# in CI there's no user, so without this preamble the agent either stalls
# or falls through to a localhost default.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


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
    """Gold solution — the oracle verifier runs checks directly.
    AMC does not have a simple 'solve' action from outside; the verifier
    drives checks independently via the running microservice."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier drives AMC API checks directly — the solution script\n"
        "# simply asserts the AMC microservice is live, then defers.\n"
        "set -euo pipefail\n"
        "\n"
        "# Probe both common AMC ports.\n"
        "for PORT in 8010 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009; do\n"
        "    if curl -sf --max-time 5 \"http://localhost:${PORT}/v1/ready\" "
        ">/dev/null 2>&1; then\n"
        "        echo \"AMC is live on port ${PORT} — verifier will drive checks.\"\n"
        "        exit 0\n"
        "    fi\n"
        "done\n"
        "echo 'AMC is not reachable on ports 8000-8010 — cannot solve without a running deployment'\n"
        "exit 1\n"
    )


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
    step-<k>/ subdirs under `<spec_stem>/<platform_short>/` per AGENTS.md § 4."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = f"{spec_stem}.json"

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / spec_stem / platform_short / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # instruction.md — one step's query + env notes ONLY.
        # Never leak checks[] into the instruction — the agent must not
        # see the verifier's criteria or it could write to the test
        # rather than do the work.
        query = expect.get("query", "")
        # Substitute {{platform}} in the query string
        query = query.replace("{{platform}}", platform)

        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host.",
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            query,
            "",
            "## Environment notes",
            "",
            spec.get("env", "").replace("{{platform}}", platform),
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(lines) + "\n")

        # task.toml
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-generate-video-calibration-{spec_stem}-{platform_short}-step-{idx}"',
            f'description = "AMC calibration query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "{spec_stem}", "{platform}"]',
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
            # profile is intentionally absent: the spec omits it by design —
            # no /vss-deploy-profile prerequisite; the agent deploys AMC via
            # the skill's own deploy reference. See AGENTS.md § 2.
            *([f'profile = "{spec["profile"]}"'] if spec.get("profile") else []),
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
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

        # environment/
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # tests/ — wrapper + generic judge + spec
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        spec_src = skill_dir / "eval" / spec_name
        if spec_src.exists():
            shutil.copy(spec_src, tests_dir / spec_name)
        else:
            # Fall back: write the in-memory spec copy
            (tests_dir / spec_name).write_text(json.dumps(spec, indent=2))

        # solution/
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # skills/ — include the calibration skill + deploy skill for diagnostics
        for src, name in (
            (skill_dir, "vss-generate-video-calibration"),
            (deploy_skill_dir, "vss-deploy-profile"),
        ):
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
        help="Path to skills/vss-deploy-profile (optional — included for agent debug)",
    )
    parser.add_argument(
        "--spec", default=None,
        help="Path to eval spec JSON (default: <skill-dir>/eval/auto-calibration.json)",
    )
    parser.add_argument(
        "--platform", default=None,
        choices=list(PLATFORMS.keys()),
        help=f"Generate for this platform only (default: {DEFAULT_PLATFORM})",
    )
    parser.add_argument(
        "--all-platforms", action="store_true",
        help="Fan out across every platform in PLATFORMS",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    deploy_skill_dir = Path(args.deploy_skill_dir) if args.deploy_skill_dir else None

    spec_path = (
        Path(args.spec) if args.spec
        else (skill_dir / "eval" / "auto-calibration.json")
    )
    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec_stem = spec_path.stem  # e.g. "auto-calibration"

    # Validate required fields
    platforms_decl = (spec.get("resources") or {}).get("platforms")
    if not platforms_decl:
        print(
            f"ERROR: spec {spec_path} is missing resources.platforms — "
            "cannot determine platform matrix",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.platform:
        platforms = [args.platform]
    elif args.all_platforms:
        platforms = list(PLATFORMS.keys())
    else:
        # Use spec-declared platforms by default
        platforms = [p for p in platforms_decl if p in PLATFORMS]
        if not platforms:
            print(
                f"ERROR: no recognized platforms in spec: {list(platforms_decl.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    expects = spec.get("expects", [])
    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path} (stem: {spec_stem})")
    print(f"  platforms    : {platforms}")
    print(f"  queries      : {len(expects)}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in expects)}")
    print()

    for platform in platforms:
        pshort = PLATFORMS[platform]["short_name"]
        for idx in range(1, len(expects) + 1):
            print(f"  GEN  vss-generate-video-calibration/{spec_stem}/{pshort}/step-{idx}")
        generate_task(platform, spec, spec_stem, output_root, skill_dir, deploy_skill_dir)

    print()
    print(f"Generated {len(platforms)} platform(s) × {len(expects)} steps = "
          f"{len(platforms) * len(expects)} task(s) under {output_root}/{spec_stem}/")
    if not spec.get("profile"):
        print(
            "\nNote: this spec OMITS `profile`. The trial runs on a bare Brev instance — "
            "no /vss-deploy-profile prerequisite is injected. The agent is expected to "
            "deploy AMC autonomously via the skill's bundled "
            "references/deploy-auto-calibration-service.md runbook."
        )


if __name__ == "__main__":
    main()
