"""Explicit temporal A1 normalization, inference and self-contained checkpoints."""

import numpy as np
import torch

from alexdoor_xas.assets.identity import assert_checkpoint_runtime_compatible
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
from alexdoor_xas.dataset.sampling import collate_torch
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.act.train import _to_device
from alexdoor_xas.policies.common.checkpoint import load_checkpoint_payload, save_checkpoint_payload
from alexdoor_xas.policies.temporal import TEMPORAL_RECIPE, TemporalConfig, causal_indices, gather

FORMAT = "b1.act.temporal.v2"
OBS_KEYS = ("temporal_rgbd_measured_state",)


def fit_temporal_stats(episodes):
    """Train-only fit from explicitly admitted real command rows; never final test.

    q/dq/previous targets retain physical units. Torque z-scores use only supported
    healthy measurements at admitted training times, with no generic clipping.
    """
    ids, actions, torque = [], [], []
    for episode in episodes:
        episode_id = episode.builder.episode_id
        if episode.metadata["split"] != "train" or episode_id in ids:
            raise ValueError("Normalization requires unique training episodes")
        ids.append(episode_id)
        rows = np.flatnonzero(episode.eligibility)
        if not len(rows):
            raise ValueError("No explicitly admitted training commands")
        actions.append(episode.file["actions/delta"][rows])
        times = episode.action_times[rows]
        stream = episode.builder.streams["tau"]
        for gen in np.unique(episode.action_generations[rows]):
            slots = times[episode.action_generations[rows] == gen]
            # Availability must hold at each decision, not merely at file end.
            for t in slots:
                idx = causal_indices(stream, [t], t, gen, episode.config.state_gap_s)
                if idx[0] >= 0 and ("valid" not in stream or stream["valid"][idx[0]]):
                    torque.append(gather(stream["values"], idx, (7,)))
    if not actions:
        raise ValueError("No training episodes")
    mean, std = np.zeros(32), np.ones(32)
    low, high = np.zeros(32), np.zeros(32)
    if torque:
        measured = NormStats.from_rows(torque)
        mean[18:25], std[18:25] = (
            measured.mean,
            np.where(measured.max > measured.min, measured.std, 1.0),
        )
        low[18:25], high[18:25] = measured.min, measured.max
    return DatasetNormStats(
        NormStats.from_rows(actions),
        NormStats(mean, std, low, high, sum(len(x) for x in actions)),
        OBS_KEYS,
        tuple(ids),
        "A1_joint_delta",
    )


def normalize_temporal_batch(batch, stats):
    obs = dict(batch["obs"])
    normalized = stats.obs.normalize(obs["robot"]).astype("f4")
    mask = np.concatenate(
        [
            np.repeat(obs["robot_valid"][..., i : i + 1], width, axis=-1)
            for i, width in enumerate((9, 9, 7, 7))
        ],
        axis=-1,
    )
    obs["robot"] = np.where(mask, normalized, 0).astype("f4")
    return dict(batch, obs=obs, actions=stats.action.normalize(batch["actions"]).astype("f4"))


def temporal_model(config=None, *, chunk_size=12, seed=0, **model_options):
    """New recipe only: history, prediction and execution horizons are independent."""
    config = TemporalConfig() if config is None else config
    torch.manual_seed(seed)
    cfg = ActModelCfg(
        chunk_size=chunk_size,
        visual_recipe=TEMPORAL_RECIPE,
        temporal_config=config.to_dict(),
        **model_options,
    )
    return ACTModel(32, 7, cfg)


def temporal_contract(config, n_action_steps):
    return dict(
        recipe=TEMPORAL_RECIPE,
        image_timing="static-role-minus-one-recent-relative-v1",
        config=config.to_dict(),
        arm_joints=list(ARM_JOINTS),
        observed_joints=list(ARM_JOINTS + NECK_JOINTS),
        action_space="A1_joint_delta",
        n_action_steps=n_action_steps,
        applied_semantics="controller_setpoint",
        qualified=False,
    )


def save_temporal_policy(path, model, stats, robot_asset, *, n_action_steps=6):
    config = TemporalConfig(**(model.cfg.temporal_config or {}))
    if (
        model.cfg.visual_recipe != TEMPORAL_RECIPE
        or model.cfg.temporal_config is None
        or (
            stats.obs_keys != OBS_KEYS
            or stats.action_space != "A1_joint_delta"
            or not 0 < n_action_steps <= model.cfg.chunk_size
        )
    ):
        raise ValueError("Incompatible temporal policy contract")
    return save_checkpoint_payload(
        path,
        FORMAT,
        model,
        dict(
            dataset=dict(
                task="b1_door_push",
                space=stats.action_space,
                version=TEMPORAL_RECIPE,
                obs_keys=OBS_KEYS,
                view_id=stats.view_id,
            )
        ),
        stats,
        dict(temporal_contract=temporal_contract(config, n_action_steps)),
        robot_asset,
    )


def temporal_inference_tensors(obs, device, *, reuse):
    """Transfer measured tokens; reuse transfers only newly encountered images.

    Host images remain a bounded snapshot. The projection selects/transfers CNN
    misses; training always transfers full images with autograd attached.
    """
    if not reuse:
        return _to_device(collate_torch(obs), device)
    measured = {key: value for key, value in obs.items() if key != "images"}
    tensors = _to_device(collate_torch(measured), device)
    tensors["images"] = torch.as_tensor(obs["images"], dtype=torch.float32, device="cpu")
    return tensors


class TemporalActPolicy:
    def __init__(self, model, stats, *, n_action_steps=6, device="cuda:0"):
        if (
            model.cfg.temporal_config is None
            or model.cfg.visual_recipe != TEMPORAL_RECIPE
            or (
                stats.obs_keys != OBS_KEYS
                or stats.obs.dim != 32
                or stats.action.dim != 7
                or stats.action_space != "A1_joint_delta"
                or not 0 < n_action_steps <= model.cfg.chunk_size
            )
        ):
            raise ValueError("Explicit temporal A1 model/stats/execution contract required")
        self.config = TemporalConfig(**model.cfg.temporal_config)
        self.model, self.stats = model.to(device).eval(), stats
        self.device = torch.device(device)
        self.n_action_steps, self.chunk_size = n_action_steps, model.cfg.chunk_size
        self.episode_id = self.generation = None

    def reset(self, episode_id, generation):
        self.episode_id, self.generation = str(episode_id), int(generation)
        self.model.obs_proj.clear_cache()

    def validate_snapshot(self, snapshot, *, now_s=None):
        if not isinstance(snapshot, dict) or "observation_config" not in snapshot:
            raise ValueError("Explicit temporal snapshot required; legacy arrays are incompatible")
        if TemporalConfig(**snapshot["observation_config"]) != self.config:
            self.model.obs_proj.clear_cache()
            raise ValueError("Snapshot preprocessing/history differs from checkpoint")
        if (snapshot["episode_id"], snapshot["generation"]) != (self.episode_id, self.generation):
            raise ValueError("Reset policy before consuming a new episode generation")
        t = snapshot["decision_time_s"]
        now = t if now_s is None else now_s
        if (
            not snapshot["ready"]
            or not np.isfinite(now)
            or now < t
            or (
                now - snapshot["image_times"][-1, 0] > self.config.image_gap_s + 1e-9
                or np.any(now - snapshot["robot_times"][-1, :2, 0] > self.config.state_gap_s + 1e-9)
            )
        ):
            raise ValueError("Temporal query unready or stale; replan/stop through existing IO")
    @torch.inference_mode()
    def predict(self, snapshot, *, now_s=None, reuse=False):
        self.validate_snapshot(snapshot, now_s=now_s)
        batch = dict(
            obs={key: value[None] for key, value in snapshot["obs"].items()},
            actions=np.zeros((1, self.chunk_size, 7), "f4"),
        )
        obs = temporal_inference_tensors(
            normalize_temporal_batch(batch, self.stats)["obs"], self.device, reuse=reuse
        )
        if reuse:
            obs["cache_keys"] = [
                (self.episode_id, self.generation, int(index), float(times[0]))
                for index, times in zip(
                    snapshot["image_source_index"], snapshot["image_times"], strict=True
                )
            ]
        prediction = self.model.predict(obs)[0].cpu().numpy()
        return self.stats.action.denormalize(prediction)

    @classmethod
    def from_checkpoint(cls, path, *, config, runtime_asset, device="cuda:0"):
        loaded = load_checkpoint_payload(path, FORMAT, "temporal ACT", device)
        cfg = ActModelCfg(**loaded.model_cfg)
        if cfg.temporal_config is None or TemporalConfig(**cfg.temporal_config) != config:
            raise ValueError("Checkpoint temporal/preprocessing recipe mismatch")
        contract = loaded.meta.get("temporal_contract", {})
        steps = contract.get("n_action_steps", 0)
        # Round-trip through JSON may turn tuples into lists; compare canonically.
        import json

        if json.dumps(contract, sort_keys=True) != json.dumps(
            temporal_contract(config, steps), sort_keys=True
        ):
            raise ValueError("Checkpoint temporal robot/action contract mismatch")
        assert_checkpoint_runtime_compatible(loaded.robot_asset, runtime_asset)
        model = ACTModel(loaded.obs_dim, loaded.action_dim, cfg).to(device)
        model.load_state_dict(loaded.state_dict)
        return cls(model, loaded.stats, n_action_steps=steps, device=device)
