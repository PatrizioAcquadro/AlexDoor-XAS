"""Frozen B1 asset identities and family-separated splits, not episode splits."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .expert import qualify_pair, trial_summary, valid_reference_inputs

PARTITIONS = ("train", "development", "test")
RECORDS = ("candidate.json", "recipe.json", "prepared.json")


def record_digest(folder: Path) -> str:
    """Bind the freeze to the three canonical records, including the expert reference."""
    digest = hashlib.sha256()
    for name in RECORDS:
        content = (folder / name).read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _validate_evidence(folder, entry, reference, setup, evidence_root):
    output = Path(evidence_root) / entry["asset_id"] / entry["evidence_run"]
    report = _read(output / "report.json")
    _require(
        report["asset_id"] == entry["asset_id"]
        and report.get("trial_isolation") == "fresh_process"
        and report.get("diagnostic_only") is False
        and str(report.get("device")).startswith("cuda"),
        f"Not a formal CUDA fresh-process pair: {output}",
    )
    for key in ("status", "theta_expert_d", "reason", "trials", "repeat_spread_deg"):
        _require(report[key] == reference[key], f"Report/reference mismatch: {output}: {key}")
    _require(_read(output / "setup.json") == setup, f"Different pair setup: {output}")
    for name in RECORDS:
        saved, current = _read(output / name), _read(folder / name)
        if name == "prepared.json":
            saved.pop("expert_qualification", None)
            current.pop("expert_qualification", None)
        _require(saved == current, f"Different qualification input: {output}: {name}")
    trials = []
    for number in (1, 2):
        trial = output / f"repeat-{number}"
        result = _read(trial / "result.json")
        trials.append(result)
        _require(
            trial_summary(result, number) == report["trials"][number - 1],
            f"Trial/report mismatch: {trial}",
        )
        traces = _read(trial / "trace.json")
        _require(bool(traces) and traces[-1]["phase"] == "release", f"Incomplete trace: {trial}")
        _require(
            len(traces) == result["visibility"]["checked_frames"],
            f"Trace/result frame count mismatch: {trial}",
        )
        # Captures are saved every 30 control steps by the frozen probe.
        expected = set(range(30, len(traces) + 1, 30))
        cameras = {int(k) for k in _read(trial / "camera.json")}
        rgb = {int(p.stem[4:]) for p in trial.glob("rgb-*.png")}
        depth = {int(p.stem[6:]) for p in trial.glob("depth-*.npy")}
        _require(expected == cameras == rgb == depth, f"Incomplete RGB-D evidence: {trial}")
    for key, value in qualify_pair(trials).items():
        _require(reference[key] == value, f"Invalid detailed expert pair: {output}: {key}")


def load_corpus(path, repo_root, *, evidence_root=None):
    """Validate the tracked freeze; optionally cross-check local qualification evidence.

    No simulator is started. Returns metadata only: family, split and expert truth
    must not become policy inputs. Binary payload/geometry review is a separate
    freeze-time audit, not established by these record checks.
    """
    root = Path(repo_root)
    corpus = _read(path)
    _require(
        corpus.get("schema") == "b1.corpus.v1" and corpus.get("status") == "frozen",
        "Expected a frozen b1.corpus.v1 manifest",
    )
    setup_path = (root / corpus["setup"]["path"]).resolve()
    _require(setup_path.is_relative_to(root.resolve()), "Setup path outside repository")
    _require(
        hashlib.sha256(setup_path.read_bytes()).hexdigest() == corpus["setup"]["sha256"],
        "Common setup changed after corpus freeze",
    )
    setup = _read(setup_path)
    asset_root = root / "datasets/doors/b1"
    entries = corpus["doors"]
    ids = [entry["asset_id"] for entry in entries]
    _require(len(ids) == len(set(ids)), "Duplicate corpus identity")
    qualified = {
        p.parent.name
        for p in asset_root.glob("*/prepared.json")
        if isinstance(ref := _read(p).get("expert_qualification"), dict)
        and ref.get("status") == "qualified"
    }
    _require(set(ids) == qualified, "Corpus must contain every published qualified identity")
    families = corpus["families"]
    owners, family_splits, geometry_splits = {}, {}, {}
    counts = {split: Counter() for split in PARTITIONS}
    used_sources = {family: set() for family in families}
    for family, spec in families.items():
        _require(bool(spec["review"]), f"Missing family review: {family}")
        for source in spec["source_uids"]:
            _require(source not in owners, f"Source assigned to multiple families: {source}")
            owners[source] = family
    for entry in entries:
        asset_id, family, split = (entry[k] for k in ("asset_id", "family", "split"))
        _require(
            Path(asset_id).name == asset_id and asset_id not in (".", ".."), "Invalid asset ID"
        )
        _require(split in PARTITIONS, f"Unknown partition: {split}")
        _require(entry["handedness"] in ("left", "right"), f"Unknown handedness: {asset_id}")
        folder = asset_root / asset_id
        candidate, prepared = _read(folder / "candidate.json"), _read(folder / "prepared.json")
        _require(candidate["asset_id"] == asset_id, f"Candidate identity mismatch: {asset_id}")
        _require(owners.get(candidate["source_uid"]) == family, f"Wrong source family: {asset_id}")
        used_sources[family].add(candidate["source_uid"])
        _require(
            family_splits.setdefault(family, split) == split,
            f"Related family crosses partitions: {family}",
        )
        _require(
            record_digest(folder) == entry["records_sha256"], f"Frozen records changed: {asset_id}"
        )
        _require(valid_reference_inputs(prepared), f"Preparation incomplete: {asset_id}")
        for key in ("handedness", "geometry_fingerprint"):
            _require(entry[key] == prepared[key], f"Frozen {key} changed: {asset_id}")
        _require(
            entry["distribution_scope"] == candidate["distribution_scope"],
            f"Distribution scope changed: {asset_id}",
        )
        fingerprint = entry["geometry_fingerprint"]
        _require(
            fingerprint not in geometry_splits,
            f"Duplicate prepared geometry: {asset_id}",
        )
        geometry_splits[fingerprint] = split
        reference = prepared["expert_qualification"]
        trials = reference["trials"]
        _require(
            len(trials) == 2
            and all(
                t["passed"]
                and t["released"]
                and t["hold_angle_deg"] is not None
                and math.isfinite(t["hold_angle_deg"])
                and t["angle_deg"] is not None
                and math.isfinite(t["angle_deg"])
                and t["stop_reason"] in ("kinematic_limit", "mechanical_stop", "safety_stop")
                for t in trials
            ),
            f"Invalid expert pair: {asset_id}",
        )
        angles = [t["angle_deg"] for t in trials]
        _require(
            min(angles) >= 45
            and max(angles) - min(angles) <= 2
            and len({(t["stop_reason"], t.get("safety_detail")) for t in trials}) == 1
            and reference["reason"] == trials[0]["stop_reason"]
            and reference["repeat_spread_deg"] == max(angles) - min(angles)
            and reference["theta_expert_d"] == entry["theta_expert_d"] == min(angles),
            f"Inconsistent expert reference: {asset_id}",
        )
        evidence = Path(reference["evidence"])
        _require(
            evidence.parent.name == asset_id and evidence.name == entry["evidence_run"],
            f"Evidence reference changed: {asset_id}",
        )
        counts[split][prepared["handedness"]] += 1
        if evidence_root is not None:
            _validate_evidence(folder, entry, reference, setup, evidence_root)
    for family, sources in used_sources.items():
        _require(sources == set(families[family]["source_uids"]), f"Unused family/source: {family}")
    for split, count in counts.items():
        _require(count["left"] > 0 and count["right"] > 0, f"Missing handedness: {split}")
        _require(dict(count) == corpus["counts"][split], f"Incorrect partition counts: {split}")
    return corpus
