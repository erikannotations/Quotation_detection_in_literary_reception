#!/usr/bin/env python3
"""Capture corpus, code, prompt, environment, and frozen experimental settings."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from detect_quotes import SYSTEM
from eval_compare import DEFAULT_MATCH
from locate_quotes import NEAR_MARGIN, PRIMARY_FILES
from project_paths import (
    EVALUATION_DIR, ROOT, SAMPLE_MANIFEST, SEGMENTATION_DIR,
)
from run_pipeline import THRESHOLD


def sha256(path):
    # Compute SHA-256 hash of a file in streaming chunks to handle large files.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_files(paths):
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return _version_from_baseline_venvs(name)


def _version_from_baseline_venvs(name):
    """Resolve a package version from any dedicated baseline virtualenv.

    Quid runs in `.venv-quid` and text-matcher in `.venv-textmatcher` (both pin
    dependencies that conflict with the rest of the pipeline), so the interpreter
    generating this manifest cannot import either and recorded `Quid: null` — the
    frozen manifest was silently missing the version of one of the compared systems,
    which is exactly the reproducibility fact a manifest exists to capture. Every
    `.venv-*` is searched rather than one hard-coded path, so adding a baseline in its
    own environment cannot reintroduce the same silent null.
    """
    for venv in sorted(ROOT.glob(".venv-*")):
        for meta in sorted(venv.glob("lib/python*/site-packages/*.dist-info/METADATA")):
            # Distribution names may contain hyphens ("text-matcher"), so split off
            # the version on the LAST hyphen rather than the first.
            dist = meta.parent.name[:-len(".dist-info")].rsplit("-", 1)[0]
            if dist.replace("_", "-").lower() != name.replace("_", "-").lower():
                continue
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(EVALUATION_DIR / "run_manifest.json"))
    parser.add_argument("--status", choices=["development", "frozen"], default="development")
    parser.add_argument("--models-config", help="JSON file recording exact model settings")
    parser.add_argument("--hybrid-rule", default="longest overlapping span; preserve detector provenance")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    model_config = {}
    if args.models_config:
        model_config = json.loads(Path(args.models_config).read_text(encoding="utf-8"))
    code_files = [
        path for path in ROOT.glob("*.py")
        if path.name != Path(args.output).name
    ] + list((ROOT / "tests").glob("*.py"))
    segmentation_files = list(SEGMENTATION_DIR.glob("*.sections.json")) + [
        SEGMENTATION_DIR / "chapter_index.json"
    ]
    # Build manifest recording corpus, code, and environment reproducibility facts.
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": args.status,
        "note": args.note,
        "task": "marked direct quotation from Pynchon fiction, minimum five words",
        "sample_manifest": {
            "path": str(SAMPLE_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(SAMPLE_MANIFEST),
        },
        "primary_sources": hashed_files(PRIMARY_FILES.values()),
        "segmentation": hashed_files(segmentation_files),
        "code": hashed_files(code_files),
        "prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "thresholds": {
            "locator_confirmation": THRESHOLD,
            "near_source_margin": NEAR_MARGIN,
            "evaluation_span_similarity": DEFAULT_MATCH,
            "minimum_quote_words": 5,
        },
        "hybrid_rule": args.hybrid_rule,
        "models": model_config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                # Only packages that live in exactly one environment: a shared
                # dependency such as nltk is installed in more than one .venv-* at
                # different versions, so a single manifest entry for it would be
                # arbitrary. Each baseline's own .runinfo.json records the versions
                # actually used by that run.
                for name in ("Quid", "text-matcher", "pydantic", "anthropic")
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.status} run manifest -> {output}")


if __name__ == "__main__":
    main()
