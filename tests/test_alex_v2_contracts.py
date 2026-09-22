"""Pure Alex V2 asset, manifest, readiness, and runtime contracts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import (
    DOOR_NON_RIGHT_ARM_DAMPING_SCALE,
    DOOR_RIGHT_ARM_ACTUATOR_NAME,
    DOOR_RIGHT_ARM_PD_GAINS,
    EXPECTED_RUNTIME_JOINTS,
    AlexV2ContractError,
    RobotAssetRef,
    assert_checkpoint_runtime_compatible,
    build_alex_v2_runtime_manifest,
    validate_alex_v2_manifest,
)
from alexdoor_xas.assets.alex_v2_manifest import (
    EXPECTED_ALEX_V2_URDF_SHA256,
    EXPECTED_COLLISION_RECORD_COUNT,
    AlexV2ManifestError,
    build_alex_v2_manifest,
)
from alexdoor_xas.dataset.robot_asset import (
    dataset_robot_asset_payload,
    load_dataset_robot_asset,
)
from alexdoor_xas.recording import EpisodeMeta

# --- test_alex_v2_infrastructure ---


def _manifest() -> dict:
    return build_alex_v2_manifest()


def _episode(manifest: dict, ref: RobotAssetRef):
    return SimpleNamespace(
        meta=SimpleNamespace(
            task=paths.ALEX_V2_TASK,
            robot_asset_id=ref.asset_id,
            robot_asset_sha256=ref.sha256,
        ),
        extras={"robot_asset_manifest": manifest},
    )


def test_manifest_freezes_all_29_runtime_joint_names_and_order() -> None:
    manifest = _manifest()
    ref = validate_alex_v2_manifest(manifest)
    assert len(EXPECTED_RUNTIME_JOINTS) == 29
    assert manifest["movable_joint_count"] == 29
    assert set(manifest["movable_joints"]) == set(EXPECTED_RUNTIME_JOINTS)
    assert ref.sha256 == EXPECTED_ALEX_V2_URDF_SHA256

    manifest["movable_joints"][5:7] = reversed(manifest["movable_joints"][5:7])
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(manifest)

    forged = _manifest()
    forged["urdf_sha256"] = "d" * 64
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged)


def test_manifest_rejects_nested_disagreement_and_rehashed_extra_inputs() -> None:
    forged_collision = _manifest()
    forged_collision["collision_profile"]["links"]["RIGHT_GRIPPER_Z_LINK"][0]["shape"] = "sphere"
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged_collision)

    forged_extra = _manifest()
    forged_extra["unexpected"] = "forged"
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged_extra)

    forged_identity = _manifest()
    forged_identity["robot_asset_id"] = ""
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged_identity)

    forged_kind = _manifest()
    forged_kind["runtime_variant"] = None
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged_kind)


def test_fixed_base_runtime_has_a_distinct_verified_identity() -> None:
    shared = _manifest()
    runtime, runtime_ref = build_alex_v2_runtime_manifest()
    shared_ref = validate_alex_v2_manifest(shared)
    assert runtime_ref != shared_ref
    assert runtime["runtime_variant"]["fix_base"] is True
    assert runtime["runtime_variant"]["robot_tag"] == paths.ALEX_V2_ROBOT_TAG
    expected_pd = runtime["runtime_variant"]["right_arm_pd"]
    assert runtime["runtime_variant"]["non_right_arm_damping_scale"] == 2.5
    assert runtime["fingerprint_inputs"]["non_right_arm_damping_scale"] == 2.5
    assert runtime["runtime_variant"]["right_arm_pd"] == expected_pd
    assert runtime["fingerprint_inputs"]["right_arm_pd"] == expected_pd
    assert expected_pd["version"] == "door-alex-v2-right-arm-ik40-pd-v2"
    assert expected_pd["actuator_name"] == DOOR_RIGHT_ARM_ACTUATOR_NAME
    assert [
        (item["joint_name"], item["stiffness"], item["damping"])
        for item in expected_pd["ordered_gains"]
    ] == list(DOOR_RIGHT_ARM_PD_GAINS)
    assert DOOR_NON_RIGHT_ARM_DAMPING_SCALE == 2.5
    assert DOOR_RIGHT_ARM_PD_GAINS == (
        ("RIGHT_SHOULDER_Y", 600.0, 15.0),
        ("RIGHT_SHOULDER_X", 600.0, 15.0),
        ("RIGHT_SHOULDER_Z", 600.0, 15.0),
        ("RIGHT_ELBOW_Y", 600.0, 15.0),
        ("RIGHT_WRIST_Z", 150.0, 4.0),
        ("RIGHT_WRIST_X", 150.0, 4.0),
    )
    assert runtime_ref.manifest_fingerprint == (
        "49ae9465d619223f8c8e6f2931a5020e45fed802601e29e56cc047393156696b"
    )
    assert runtime["actuator_config_version"] == "door-alex-v2-fixedbase-right-arm-pd-v2"
    assert "actuator_config_version" not in shared

    forged = deepcopy(runtime)
    forged["runtime_variant"]["base_asset"]["manifest_fingerprint"] = "d" * 64
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged)

    forged_scale = deepcopy(runtime)
    forged_scale["runtime_variant"]["non_right_arm_damping_scale"] = 1.0
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged_scale)

    forged_gain = deepcopy(runtime)
    forged_gain["runtime_variant"]["right_arm_pd"]["ordered_gains"][3]["damping"] = 39.0
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged_gain)


def test_v2_dataset_payload_embeds_and_revalidates_full_manifest(tmp_path) -> None:
    manifest = _manifest()
    ref = validate_alex_v2_manifest(manifest)
    payload = dataset_robot_asset_payload([_episode(manifest, ref), _episode(manifest, ref)])
    assert payload == {**ref.to_dict(), "manifest": manifest}

    (tmp_path / "meta.json").write_text(json.dumps({"robot_asset": payload}))
    loaded_ref, loaded_manifest = load_dataset_robot_asset(tmp_path, require=True)
    assert loaded_ref == ref
    assert loaded_manifest == manifest


def test_v2_dataset_rejects_missing_or_mixed_episode_provenance() -> None:
    manifest = _manifest()
    ref = validate_alex_v2_manifest(manifest)
    missing = SimpleNamespace(
        meta=SimpleNamespace(
            task=paths.ALEX_V2_TASK,
            robot_asset_id="",
            robot_asset_sha256="",
        ),
        extras={},
    )
    with pytest.raises(AlexV2ContractError, match="require robot asset provenance"):
        dataset_robot_asset_payload([missing])

    other = RobotAssetRef("other", "b" * 64)
    with pytest.raises(AlexV2ContractError, match="do not share one robot asset"):
        dataset_robot_asset_payload([_episode(manifest, ref), _episode(manifest, other)])


def test_dataset_payload_rejects_mixed_tasks_even_if_v2_is_not_first() -> None:
    other_task = SimpleNamespace(
        meta=SimpleNamespace(
            task="door_push",
            robot_asset_id="",
            robot_asset_sha256="",
        ),
        extras={},
    )
    v2 = SimpleNamespace(
        meta=SimpleNamespace(
            task=paths.ALEX_V2_TASK,
            robot_asset_id="",
            robot_asset_sha256="",
        ),
        extras={},
    )
    with pytest.raises(AlexV2ContractError, match="cannot mix episode tasks"):
        dataset_robot_asset_payload([other_task, v2])


def test_checkpoint_runtime_gate_requires_exact_alex_v2_identity() -> None:
    runtime = validate_alex_v2_manifest(_manifest())
    assert assert_checkpoint_runtime_compatible(runtime, runtime) == "v2_native"
    with pytest.raises(AlexV2ContractError, match="unfingerprinted"):
        assert_checkpoint_runtime_compatible(None, runtime)
    other = RobotAssetRef("other", "b" * 64)
    with pytest.raises(AlexV2ContractError, match="checkpoint asset 'other'"):
        assert_checkpoint_runtime_compatible(other, runtime)


def test_episode_meta_carries_asset_identity() -> None:
    base = dict(
        episode_id="episode",
        task=paths.ALEX_V2_TASK,
        action_space="A2_ee_delta",
        robot=paths.ALEX_V2_ROBOT_TAG,
        scene="door",
        policy="scripted",
        seed=0,
        sim_dt=0.005,
        control_dt=0.02,
        created_utc="2026-01-01T00:00:00+00:00",
    )
    enriched = EpisodeMeta(**base, robot_asset_id="v2", robot_asset_sha256="c" * 64)
    assert enriched.to_dict()["robot_asset_sha256"] == "c" * 64


# --- test_alex_v2_manifest ---


def test_builder_derives_identity_joints_and_primitive_collisions() -> None:
    manifest = build_alex_v2_manifest()

    assert hashlib.sha256(paths.ALEX_V2_URDF.read_bytes()).hexdigest() == (
        EXPECTED_ALEX_V2_URDF_SHA256
    )
    assert manifest["urdf_sha256"] == EXPECTED_ALEX_V2_URDF_SHA256
    assert manifest["robot_asset_sha256"] == EXPECTED_ALEX_V2_URDF_SHA256
    assert manifest["movable_joint_count"] == 29
    assert len(manifest["movable_joints"]) == len(set(manifest["movable_joints"]))
    assert set(manifest["movable_joints"]) == set(EXPECTED_RUNTIME_JOINTS)

    links = manifest["collision_profile"]["links"]
    assert len(links) == 19
    assert sum(len(records) for records in links.values()) == EXPECTED_COLLISION_RECORD_COUNT
    right_gripper = links["RIGHT_GRIPPER_Z_LINK"]
    assert [record["name"] for record in right_gripper] == [
        "right_gripper_z_collision",
        "right_fist_collision",
        "right_finger_collision",
        "right_thumb_collision",
    ]
    assert {record["shape"] for records in links.values() for record in records} == {
        "box",
        "capsule",
        "cylinder",
        "sphere",
    }


def test_builder_rejects_any_urdf_identity_drift(tmp_path) -> None:
    changed = tmp_path / "alex_v2.urdf"
    changed.write_bytes(paths.ALEX_V2_URDF.read_bytes() + b"\n")

    with pytest.raises(AlexV2ManifestError, match="identity differs"):
        build_alex_v2_manifest(changed)


# --- test_check_env ---


def _check_env_module():
    script = Path(__file__).parents[1] / "scripts" / "check_env.py"
    spec = importlib.util.spec_from_file_location("alexdoor_check_env", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alex_v2_factory_check_fails_loudly_with_install_action(tmp_path) -> None:
    check_env = _check_env_module()

    failure = check_env._purdue_module_failure(
        find_spec=lambda _name: None,
        module_file=tmp_path / "missing" / "alex_v2.py",
    )

    assert "ihmc_alex_isaaclab.robots.alex_purdue is not the installed external package" in failure
    assert "pip install -e" in failure
    assert "/Desktop/Alex" in failure


def test_alex_v2_factory_check_accepts_discoverable_module(tmp_path) -> None:
    check_env = _check_env_module()
    module_file = tmp_path / "alex_v2.py"

    failure = check_env._purdue_module_failure(
        find_spec=lambda _name: SimpleNamespace(origin=str(module_file)),
        module_file=module_file,
    )

    assert failure is None


def test_environment_gate_requires_official_ga_archive_identity() -> None:
    check_env = _check_env_module()

    assert (
        check_env._isaac_sim_version_failure(
            check_env.EXPECTED_ISAAC_SIM_BUILD,
            "6.0.1",
        )
        is None
    )
    assert "official GA" in check_env._isaac_sim_version_failure(
        "6.0.1-rc.6+release.older",
        "6.0.1",
    )
    assert check_env._cuda_failure(True) is None
    assert "CUDA is not available" in check_env._cuda_failure(False)
    assert "CUDA probe failed" in check_env._cuda_failure(False, RuntimeError("probe"))
