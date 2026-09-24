"""Model and tensor-training parameters; B1 orchestration is defined in Phase 7."""

from dataclasses import dataclass


class DiffusionConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DiffusionModelCfg:
    horizon: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"


@dataclass(frozen=True)
class DiffusionTrainCfg:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-3
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.999
    seed: int = 0
    device: str = "cuda"
    val_every: int = 10
    val_inference_steps: int = 10
