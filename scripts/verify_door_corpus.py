#!/usr/bin/env python
"""Verify the frozen B1 corpus records and optionally the local expert evidence."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.qualification.corpus import load_corpus  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "datasets/doors/b1/corpus.json")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help="Also check local reports/traces/capture inventory under this expert cache root",
    )
    args = parser.parse_args()
    try:
        corpus = load_corpus(args.manifest, REPO, evidence_root=args.evidence_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Corpus verification failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": "pass",
                "doors": len(corpus["doors"]),
                "counts": corpus["counts"],
                "scope": "records_and_local_evidence"
                if args.evidence_root
                else "tracked_records_only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
