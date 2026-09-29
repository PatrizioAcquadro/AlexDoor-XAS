"""B1 ACT/Diffusion data, normalization and artifact binding; no training CLI."""

import numpy as np

from alexdoor_xas.action.b1 import ACTION_DIMS, DISCRETE_COLUMNS
from alexdoor_xas.action.spaces import A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.assets.identity import assert_checkpoint_runtime_compatible
from alexdoor_xas.dataset.b1 import B1Dataset
from alexdoor_xas.dataset.normalize import compute_norm_stats
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS, validate_contract
from alexdoor_xas.policies.common.data import (
    PolicyData,
    make_eval_factory,
    make_train_factory,
    normalize_batch,
)

FORMATS = {"act": "alexdoor_xas.act.b1.v1", "diffusion": "alexdoor_xas.diffusion.b1.v1"}


def load_b1_data(directory, space, *, binding=None):
    dataset = B1Dataset(directory, space, binding=binding)
    train = tuple(r.episode_id for r in dataset.records if r.split == "train")
    development = tuple(r.episode_id for r in dataset.records if r.split == "development")
    stats = _stats(dataset, train)
    return PolicyData(dataset, train, development, stats, dataset.robot_asset)


def _stats(dataset, train):
    stats = compute_norm_stats(dataset, list(train), OBS_KEYS)
    if dataset.action_space == A4_OBJ_CENTRIC_CHUNK:
        # Categories/termination have a fixed mapping even if a train split has
        # few examples. Continuous columns retain the family's normal scaling.
        columns = list(DISCRETE_COLUMNS)
        stats.action.mean[columns] = 0
        stats.action.std[columns] = 1
        stats.action.min[columns] = 0
        stats.action.max[columns] = 1
    return stats


def validate_data(data):
    if not isinstance(data.dataset, B1Dataset):
        raise ValueError("Expected a B1 dataset")
    expected_train = tuple(r.episode_id for r in data.dataset.records if r.split == "train")
    expected_dev = tuple(r.episode_id for r in data.dataset.records if r.split == "development")
    if (
        data.train_ids != expected_train
        or data.val_ids != expected_dev
        or data.robot_asset != data.dataset.robot_asset
        or data.stats.to_dict() != _stats(data.dataset, expected_train).to_dict()
    ):
        raise ValueError("B1 normalization/membership must match the training identities")
    check_stats(data.stats, data.dataset.contract)


def check_stats(stats, contract):
    validate_contract(contract)
    if (
        stats.action_space != contract["action_space"]
        or stats.obs_keys != OBS_KEYS
        or stats.obs.dim != contract["obs_dim"]
        or stats.action.dim != contract["action_dim"]
        or not stats.train_episode_ids
        or stats.view_id is not None
    ):
        raise ValueError("Normalization does not match the B1 contract")
    stats.obs.validate()
    stats.action.validate()
    if stats.action_space == A4_OBJ_CENTRIC_CHUNK:
        for key, value in (("mean", 0), ("std", 1), ("min", 0), ("max", 1)):
            if not np.all(getattr(stats.action, key)[list(DISCRETE_COLUMNS)] == value):
                raise ValueError("A4 category/termination normalization must be fixed")


def batch_factories(data, family, horizon, batch_size, seed):
    if family not in FORMATS or not isinstance(data.dataset, B1Dataset):
        raise ValueError("Expected a B1 dataset and ACT or Diffusion")
    validate_data(data)
    if family == "diffusion":
        from alexdoor_xas.policies.diffusion.data import make_diffusion_normalizer

        normalize = make_diffusion_normalizer(data.stats)
    else:
        normalize = normalize_batch
    train = make_train_factory(data, horizon, batch_size, seed, normalize=normalize)
    development = (
        make_eval_factory(data, horizon, batch_size, seed, data.val_ids, normalize=normalize)
        if data.val_ids
        else None
    )
    return train, development


def _family(family):
    if family == "act":
        from alexdoor_xas.policies.act.config import ActModelCfg
        from alexdoor_xas.policies.act.model import ACTModel
        from alexdoor_xas.policies.act.policy import ActPolicy

        return ActModelCfg, ACTModel, ActPolicy
    if family == "diffusion":
        from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
        from alexdoor_xas.policies.diffusion.model import DiffusionTransformer
        from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy

        return DiffusionModelCfg, DiffusionTransformer, DiffusionPolicy
    raise ValueError("Expected ACT or Diffusion")


def make_model(data, family, model_cfg, *, seed=0, device="cuda:0"):
    """Build the existing family model with the complete B1 observation/action width."""
    import torch

    if not isinstance(data.dataset, B1Dataset):
        raise ValueError("B1 model construction requires B1 data")
    check_stats(data.stats, data.dataset.contract)
    cfg_type, model_type, _ = _family(family)
    if not isinstance(model_cfg, cfg_type):
        raise ValueError("Model configuration belongs to another policy family")
    torch.manual_seed(seed)
    with torch.device(device):
        return model_type(data.obs_dim, data.action_dim, model_cfg)


def train_model(data, family, model, train_cfg, *, batch_size, **kwargs):
    """Reuse the existing tensor trainer. Caller owns authorization and GPU scheduling."""
    horizon = model.cfg.chunk_size if family == "act" else model.cfg.horizon
    train, development = batch_factories(data, family, horizon, batch_size, train_cfg.seed)
    if model.obs_dim != data.obs_dim or model.action_dim != data.action_dim:
        raise ValueError("B1 model dimensions differ from its dataset")
    if family == "act":
        from alexdoor_xas.policies.act.train import train_act

        return train_act(model, train, train_cfg, make_val_batches=development, **kwargs)
    from alexdoor_xas.policies.diffusion.schedulers import make_train_scheduler
    from alexdoor_xas.policies.diffusion.train import train_diffusion

    return train_diffusion(
        model,
        make_train_scheduler(model.cfg),
        train,
        train_cfg,
        make_val_batches=development,
        **kwargs,
    )


def save_policy(path, family, model, data, *, meta=None):
    from alexdoor_xas.policies.common.checkpoint import save_checkpoint_payload

    if family not in FORMATS or not isinstance(data.dataset, B1Dataset):
        raise ValueError("Expected B1 policy data and family")
    validate_data(data)
    if not isinstance(model.cfg, _family(family)[0]):
        raise ValueError("Model configuration belongs to another policy family")
    metadata = dict(meta or {}, b1_contract=data.dataset.contract)
    config = dict(
        dataset=dict(
            task="b1_door_push",
            space=data.stats.action_space,
            version=data.dataset.dataset_dir.name,
            obs_keys=OBS_KEYS,
            view_id=None,
        )
    )
    return save_checkpoint_payload(
        path, FORMATS[family], model, config, data.stats, metadata, data.robot_asset
    )


def load_policy_payload(path, family, *, binding, runtime_asset, device="cpu"):
    """Validate metadata/normalization before constructing any model."""
    from alexdoor_xas.policies.common.checkpoint import load_checkpoint_payload

    if family not in FORMATS:
        raise ValueError("Unknown B1 policy family")
    loaded = load_checkpoint_payload(path, FORMATS[family], family, device)
    contract = loaded.meta.get("b1_contract")
    validate_contract(contract, binding, loaded.stats.action_space)
    check_stats(loaded.stats, contract)
    assert_checkpoint_runtime_compatible(loaded.robot_asset, runtime_asset)
    return loaded, contract


class B1Policy:
    """Family inference with a mandatory observed-input/robot/action contract."""

    def __init__(self, policy, family, contract):
        if family not in FORMATS:
            raise ValueError("Unknown B1 policy family")
        self.binding = validate_contract(contract)
        check_stats(policy.stats, contract)
        self.policy, self.family, self.contract = policy, family, contract
        self.action_space = contract["action_space"]
        self.chunk_size = policy.chunk_size

    @classmethod
    def from_checkpoint(
        cls,
        path,
        family,
        *,
        binding,
        runtime_asset,
        device="cuda:0",
        sampler="ddim",
        num_inference_steps=20,
    ):
        loaded, contract = load_policy_payload(
            path, family, binding=binding, runtime_asset=runtime_asset, device=device
        )
        config_type, model_type, policy_type = _family(family)
        import torch

        with torch.device(device):
            model = model_type(loaded.obs_dim, loaded.action_dim, config_type(**loaded.model_cfg))
        model.load_state_dict(loaded.state_dict)
        args = dict(device=device)
        if family == "diffusion":
            args.update(sampler=sampler, num_inference_steps=num_inference_steps)
        return cls(policy_type(model, loaded.stats, **args), family, contract)

    def predict(self, features):
        features = np.asarray(features)
        if features.shape != (self.binding.obs_dim,) or not np.isfinite(features).all():
            raise ValueError("Invalid B1 policy observation")
        chunk = np.asarray(self.policy.predict(features))
        if (
            chunk.shape != (self.chunk_size, ACTION_DIMS[self.action_space])
            or not np.isfinite(chunk).all()
        ):
            raise ValueError("Invalid B1 predicted action chunk")
        return chunk

    def chunk_source(self, observe, *, temporal_ensemble=False, n_action_steps=None):
        from alexdoor_xas.policies.act.policy import act_chunk_source
        from alexdoor_xas.policies.diffusion.policy import diffusion_chunk_source

        if temporal_ensemble and (
            self.action_space == A4_OBJ_CENTRIC_CHUNK or self.family != "act"
        ):
            raise ValueError("Temporal ensembling is only supported for ACT A1-A3")
        if self.family == "act":
            if n_action_steps is not None:
                raise ValueError("n_action_steps is a Diffusion setting")
            return act_chunk_source(self, observe, temporal_ensemble=temporal_ensemble)
        return diffusion_chunk_source(self, observe, n_action_steps=n_action_steps)

    def reset(self, seed=0):
        if self.family == "diffusion":
            self.policy.seed(seed)
