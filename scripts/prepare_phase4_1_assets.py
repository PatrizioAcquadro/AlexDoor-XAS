#!/usr/bin/env python
"""Maintain the bounded Phase 4.1 worklist, remote gate, and source ingest.

This script never downloads from a website and never stores credentials.  It consumes
the metadata captured before download, then admits exactly one downloaded payload for
the candidate occupying a slot.  Rejection removes that payload and preserves only the
metadata and reason in the machine-local ledger.
"""

from __future__ import annotations

import argparse
import errno
import shutil
from datetime import UTC, datetime
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.door_qualification import (
    QualificationError,
    dump_json,
    load_json,
    sha256_file,
    validate_manifest,
    validate_remote_candidate,
)

WORKLIST = paths.PHASE4_1_EVIDENCE_DIR / "worklist.json"
REJECTIONS = paths.PHASE4_1_EVIDENCE_DIR / "rejections.json"


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
    if not REJECTIONS.exists():
        dump_json(REJECTIONS, {"schema": "alexdoor.phase4_1_rejections.v1", "entries": []})
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


def _ingest(slot_id: int, download: Path, *, move: bool = False) -> None:
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
    browser_download_retained = False
    if move:
        try:
            shutil.move(download, target)
        except OSError as error:
            # Codex's in-app browser exposes completed downloads through a
            # read-only cross-filesystem mount.  ``shutil.move`` copies the
            # payload successfully and then receives EROFS while unlinking the
            # browser-owned source.  Treat that source as an ephemeral browser
            # artifact, but reject every other partial-move failure.
            if error.errno != errno.EROFS or not target.is_file():
                raise
            if sha256_file(target) != digest:
                target.unlink(missing_ok=True)
                raise QualificationError(
                    "cross-filesystem ingest copy failed checksum"
                ) from error
            browser_download_retained = True
    else:
        shutil.copy2(download, target)
    slot.update(
        {
            "state": "ingested",
            "source_path": str(target.relative_to(paths.REPO_ROOT)),
            "source_sha256": digest,
            "source_size_bytes": target.stat().st_size,
            "browser_download_retained": browser_download_retained,
            "ingested_utc": _utc_now(),
        }
    )
    dump_json(WORKLIST, worklist)
    print(f"PASS ingest: slot={slot_id} sha256={digest} bytes={target.stat().st_size}")


def _reject(slot_id: int, stage: str, reason: str) -> None:
    worklist = load_json(WORKLIST)
    slot = _find_slot(worklist, slot_id)
    rejection = _load_or(
        REJECTIONS, {"schema": "alexdoor.phase4_1_rejections.v1", "entries": []}
    )
    rejection["entries"].append(
        {
            "slot": slot_id,
            "source_uid": slot.get("candidate", {}).get("source_uid"),
            "source_url": slot.get("candidate", {}).get("source_url"),
            "stage": stage,
            "reason": reason,
            "rejected_utc": _utc_now(),
        }
    )
    payload_dir = paths.PHASE4_1_SOURCE_DIR / f"door_{slot_id:02d}"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    normalized_dir = paths.PHASE4_1_NORMALIZED_DIR / f"door_{slot_id:02d}"
    if normalized_dir.exists():
        shutil.rmtree(normalized_dir)
    evidence_dir = paths.PHASE4_1_EVIDENCE_DIR / f"door_{slot_id:02d}"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    recipe_path = paths.PHASE4_1_RECIPES_DIR / f"door_{slot_id:02d}.json"
    if recipe_path.exists():
        recipe_path.unlink()
    worklist["slots"] = [entry for entry in worklist["slots"] if entry is not slot]
    dump_json(REJECTIONS, rejection)
    dump_json(WORKLIST, worklist)
    print(f"REJECTED: slot={slot_id} stage={stage}; slot is vacant and payload-free")


def _record_remote_rejection(candidate_path: Path, reason: str) -> None:
    candidate = load_json(candidate_path)
    rejection = _load_or(
        REJECTIONS, {"schema": "alexdoor.phase4_1_rejections.v1", "entries": []}
    )
    rejection["entries"].append(
        {
            "slot": int(candidate["slot"]),
            "source_uid": candidate.get("source_uid"),
            "source_url": candidate.get("source_url"),
            "candidate": candidate,
            "stage": "remote",
            "reason": reason,
            "rejected_utc": _utc_now(),
        }
    )
    dump_json(REJECTIONS, rejection)
    print(
        f"REJECTED remote candidate: slot={candidate['slot']} "
        f"uid={candidate.get('source_uid')}"
    )


def _validate(final: bool) -> None:
    worklist = load_json(WORKLIST)
    states = {int(slot["slot"]): slot["state"] for slot in worklist["slots"]}
    source_dirs = sorted(paths.PHASE4_1_SOURCE_DIR.glob("door_*"))
    normalized_dirs = sorted(paths.PHASE4_1_NORMALIZED_DIR.glob("door_*"))
    if len(source_dirs) != len(set(path.name for path in source_dirs)):
        raise QualificationError("duplicate source directories")
    if final:
        validate_manifest(load_json(paths.PHASE4_1_MANIFEST), require_complete=True)
        if set(states) != set(range(1, 25)) or set(states.values()) != {"qualified"}:
            raise QualificationError(f"final worklist has incomplete states: {states}")
        if len(source_dirs) != 24 or len(normalized_dirs) != 24:
            raise QualificationError(
                f"final payload count must be source=24 normalized=24, got "
                f"source={len(source_dirs)} normalized={len(normalized_dirs)}"
            )
    print(
        f"PASS workspace: slots={len(states)} source_payloads={len(source_dirs)} "
        f"normalized_payloads={len(normalized_dirs)} final={final}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    gate = sub.add_parser("gate")
    gate.add_argument("candidate", type=Path)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--slot", type=int, required=True)
    ingest.add_argument("--download", type=Path, required=True)
    ingest.add_argument(
        "--move",
        action="store_true",
        help="Move the browser download into its sole admitted payload location.",
    )
    reject = sub.add_parser("reject")
    reject.add_argument("--slot", type=int, required=True)
    reject.add_argument("--stage", required=True)
    reject.add_argument("--reason", required=True)
    remote_reject = sub.add_parser("record-remote-rejection")
    remote_reject.add_argument("candidate", type=Path)
    remote_reject.add_argument("--reason", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--final", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "init":
        _init()
    elif args.command == "gate":
        _gate(args.candidate)
    elif args.command == "ingest":
        _ingest(args.slot, args.download.expanduser().resolve(), move=args.move)
    elif args.command == "reject":
        _reject(args.slot, args.stage, args.reason)
    elif args.command == "record-remote-rejection":
        _record_remote_rejection(args.candidate, args.reason)
    else:
        _validate(args.final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
