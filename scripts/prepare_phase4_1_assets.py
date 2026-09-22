#!/usr/bin/env python
"""Stage local inputs for the retained legacy door normalization tools.

Only init, metadata gate, and copy-only ingest remain. The slot-based worklist
is a legacy intermediate format, not the B1 corpus or Phase 5 intake workflow.
Sources are preserved; no qualification, rejection cleanup, or finalization runs.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.door_qualification import (
    QualificationError,
    dump_json,
    load_json,
    sha256_file,
    validate_remote_candidate,
)

WORKLIST = paths.PHASE4_1_EVIDENCE_DIR / "worklist.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _empty_worklist() -> dict:
    return {"schema": "alexdoor.phase4_1_worklist.v1", "slots": []}


def _load_or(path: Path, default):
    return load_json(path) if path.is_file() else default


def _init() -> None:
    for directory in (
        paths.PHASE4_1_SOURCE_DIR,
        paths.PHASE4_1_NORMALIZED_DIR,
        paths.PHASE4_1_EVIDENCE_DIR,
        paths.PHASE4_1_RECIPES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not WORKLIST.exists():
        dump_json(WORKLIST, _empty_worklist())
    print(f"initialized Phase 4.1 workspace at {paths.PHASE4_1_ASSETS_DIR}")


def _accepted_records(worklist: dict) -> list[dict]:
    return [slot["candidate"] for slot in worklist["slots"] if slot["state"] != "vacant"]


def _gate(candidate_path: Path) -> None:
    candidate = load_json(candidate_path)
    worklist = _load_or(WORKLIST, _empty_worklist())
    slot_id = str(candidate["slot"])
    if not 1 <= int(slot_id) <= 24:
        raise QualificationError("slot must be in 1..24")
    occupied = [slot for slot in worklist["slots"] if str(slot["slot"]) == slot_id]
    if occupied and occupied[0]["state"] != "vacant":
        raise QualificationError(f"slot {slot_id} is already occupied")
    validate_remote_candidate(candidate, _accepted_records(worklist))
    entry = {
        "slot": int(slot_id),
        "state": "remote_pass",
        "candidate": candidate,
        "remote_gate_utc": _utc_now(),
    }
    worklist["slots"] = [
        slot for slot in worklist["slots"] if str(slot["slot"]) != slot_id
    ] + [entry]
    worklist["slots"].sort(key=lambda slot: int(slot["slot"]))
    dump_json(WORKLIST, worklist)
    print(f"PASS remote gate: slot={slot_id} uid={candidate['source_uid']}")


def _find_slot(worklist: dict, slot_id: int) -> dict:
    for slot in worklist["slots"]:
        if int(slot["slot"]) == slot_id:
            return slot
    raise QualificationError(f"slot {slot_id} is vacant")


def _ingest(slot_id: int, download: Path) -> None:
    worklist = load_json(WORKLIST)
    slot = _find_slot(worklist, slot_id)
    if slot["state"] != "remote_pass":
        raise QualificationError(f"slot {slot_id} is not ready for ingest: {slot['state']}")
    if not download.is_file() or download.stat().st_size <= 0:
        raise QualificationError(f"download is missing or empty: {download}")
    expected_format = str(slot["candidate"]["selected_format"]).lower()
    suffix = download.suffix.lower().lstrip(".")
    if suffix != expected_format:
        raise QualificationError(
            f"download suffix {suffix!r} does not match gated format {expected_format!r}"
        )
    digest = sha256_file(download)
    for other in worklist["slots"]:
        if other is slot or other.get("source_sha256") is None:
            continue
        if other["source_sha256"] == digest:
            raise QualificationError(
                f"download duplicates slot {other['slot']} by source SHA-256"
            )
    target = paths.PHASE4_1_SOURCE_DIR / f"door_{slot_id:02d}" / f"source.{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(target.parent.iterdir()):
        raise QualificationError(f"slot {slot_id} already has a local source payload")
    shutil.copy2(download, target)
    slot.update(
        {
            "state": "ingested",
            "source_path": str(target.relative_to(paths.REPO_ROOT)),
            "source_sha256": digest,
            "source_size_bytes": target.stat().st_size,
            "ingested_utc": _utc_now(),
        }
    )
    dump_json(WORKLIST, worklist)
    print(f"PASS ingest: slot={slot_id} sha256={digest} bytes={target.stat().st_size}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    gate = sub.add_parser("gate")
    gate.add_argument("candidate", type=Path)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--slot", type=int, required=True)
    ingest.add_argument("--download", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "init":
        _init()
    elif args.command == "gate":
        _gate(args.candidate)
    elif args.command == "ingest":
        _ingest(args.slot, args.download.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
