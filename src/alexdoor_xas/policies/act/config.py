"""Model and tensor-training parameters; B1 orchestration is defined in Phase 7."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActModelCfg:
    chunk_size: int = 40
    d_model: int = 128
    n_heads: int = 4
    dim_feedforward: int = 512
    z_dim: int = 16
    cvae_encoder_layers: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class ActTrainCfg:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    kl_weight: float = 10.0
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    val_every: int = 5
