"""Prevent family leakage, incomplete membership and silent drift of the B1 freeze."""

import json
import shutil
from pathlib import Path

import pytest

from alexdoor_xas.qualification.corpus import _validate_evidence, load_corpus, record_digest

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path
    asset_root = root / "assets/doors/b1"
    for source in (REPO / "assets/doors/b1").glob("*/*.json"):
        target = asset_root / source.relative_to(REPO / "assets/doors/b1")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (root / "configs").mkdir()
    shutil.copyfile(
        REPO / "configs/purdue_synthetic_probe.json", root / "configs/purdue_synthetic_probe.json"
    )
    path = asset_root / "corpus.json"
    shutil.copyfile(REPO / "assets/doors/b1/corpus.json", path)
    return root, path, json.loads(path.read_text())


def test_published_corpus_is_complete_and_psx_cross_pack_reuse_stays_in_train():
    corpus = load_corpus(REPO / "assets/doors/b1/corpus.json", REPO)
    assert len(corpus["doors"]) == 32
    residential = [d for d in corpus["doors"] if d["family"] == "psx-residential"]
    assert len(residential) == 15
    assert {d["split"] for d in residential} == {"train"}
    assert len(corpus["families"]["psx-residential"]["source_uids"]) == 3


@pytest.mark.parametrize("change", ["omit", "duplicate", "family", "handedness", "source"])
def test_split_rejects_coverage_and_leakage_errors(frozen, change):
    root, path, corpus = frozen
    message = {
        "omit": "every published",
        "duplicate": "Duplicate corpus identity",
        "family": "family crosses",
        "handedness": "Missing handedness",
        "source": "multiple families",
    }[change]
    if change == "omit":
        corpus["doors"].pop()
    elif change == "duplicate":
        corpus["doors"].append(corpus["doors"][0])
    elif change == "family":
        next(d for d in corpus["doors"] if d["asset_id"] == "psx-front-002-ee7d5c6")["split"] = (
            "test"
        )
    elif change == "handedness":
        next(d for d in corpus["doors"] if d["family"] == "modern")["split"] = "test"
        corpus["counts"]["development"] = {"left": 5}
        corpus["counts"]["test"] = {"left": 5, "right": 3}
    else:
        corpus["families"]["modern"]["source_uids"].append(
            corpus["families"]["psx-residential"]["source_uids"][0]
        )
    write(path, corpus)
    with pytest.raises(ValueError, match=message):
        load_corpus(path, root)


@pytest.mark.parametrize("record", ["candidate.json", "recipe.json", "prepared.json", "setup"])
def test_freeze_rejects_input_or_reference_drift(frozen, record):
    root, path, corpus = frozen
    target = root / "assets/doors/b1" / corpus["doors"][0]["asset_id"] / record
    if record == "setup":
        target = root / "configs/purdue_synthetic_probe.json"
    value = json.loads(target.read_text())
    value["changed_after_freeze"] = True
    write(target, value)
    with pytest.raises(ValueError, match="changed"):
        load_corpus(path, root)


@pytest.mark.parametrize("failure", ["release", "nan", "below_gate", "spread"])
def test_qualification_is_checked_even_if_record_digest_is_refreshed(frozen, failure):
    root, path, corpus = frozen
    entry = corpus["doors"][0]
    folder = root / "assets/doors/b1" / entry["asset_id"]
    record = json.loads((folder / "prepared.json").read_text())
    trials = record["expert_qualification"]["trials"]
    if failure == "release":
        trials[1]["released"] = False
    elif failure == "nan":
        trials[1]["hold_angle_deg"] = float("nan")
    elif failure == "below_gate":
        trials[1]["angle_deg"] = 44.9
    else:
        trials[1]["angle_deg"] = trials[0]["angle_deg"] + 2.1
    write(folder / "prepared.json", record)
    entry["records_sha256"] = record_digest(folder)
    write(path, corpus)
    with pytest.raises(ValueError, match="expert"):
        load_corpus(path, root)


def test_evidence_check_rejects_missing_capture_even_with_passing_results(frozen):
    root, _, corpus = frozen
    entry = corpus["doors"][0]
    folder = root / "assets/doors/b1" / entry["asset_id"]
    reference = json.loads((folder / "prepared.json").read_text())["expert_qualification"]
    setup = json.loads((root / corpus["setup"]["path"]).read_text())
    evidence_root = root / "evidence"
    output = evidence_root / entry["asset_id"] / entry["evidence_run"]
    output.mkdir(parents=True)
    write(
        output / "report.json",
        {
            **reference,
            "asset_id": entry["asset_id"],
            "trial_isolation": "fresh_process",
            "diagnostic_only": False,
            "device": "cuda:0",
        },
    )
    write(output / "setup.json", setup)
    for name in ("candidate.json", "recipe.json", "prepared.json"):
        shutil.copyfile(folder / name, output / name)
    for number in (1, 2):
        trial = output / f"repeat-{number}"
        trial.mkdir()
        write(
            trial / "result.json",
            {
                **reference["trials"][number - 1],
                "case": entry["asset_id"],
                "clearance_m": 0.0,
            },
        )
        frames = reference["trials"][number - 1]["visibility"]["checked_frames"]
        write(trial / "trace.json", [{"phase": "release"}] * frames)
        write(trial / "camera.json", {str(tick): {} for tick in range(30, frames + 1, 30)})
        # This unit test checks inventory, not image decoding or simulated physics.
        for tick in range(30, frames + 1, 30):
            (trial / f"rgb-{tick:05d}.png").touch()
            (trial / f"depth-{tick:05d}.npy").touch()
    _validate_evidence(folder, entry, reference, setup, evidence_root)
    (output / "repeat-2/depth-00030.npy").unlink()
    with pytest.raises(ValueError, match="Incomplete RGB-D"):
        _validate_evidence(folder, entry, reference, setup, evidence_root)
    (output / "repeat-2/depth-00030.npy").touch()
    write(output / "repeat-2/trace.json", [{"phase": "release"}] * 30)
    with pytest.raises(ValueError, match="frame count mismatch"):
        _validate_evidence(folder, entry, reference, setup, evidence_root)
