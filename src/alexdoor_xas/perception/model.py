"""Frozen spatial RGB features and a small, shared causal RGB-D door estimator."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from alexdoor_xas.action.frames import ObjectFrame, validate_object_frame

STATE_CONFIDENCE = "articulated-state-v1"


def preprocess(rgb, depth, valid, intrinsics, size=224):
    """Letterbox the whole image. Depth/mask use nearest sampling, no invalid mixing."""
    h, w = rgb.shape[-3:-1]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    top, left = (size - nh) // 2, (size - nw) // 2
    pad = (left, size - nw - left, top, size - nh - top)
    rgb = rgb.movedim(-1, -3).float() / 255
    rgb = F.interpolate(rgb, (nh, nw), mode="bilinear", align_corners=False)
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = F.pad((rgb - mean) / std, pad)
    valid = valid & torch.isfinite(depth) & (depth > 0)
    depth = torch.where(valid, depth, 0).movedim(-1, -3).float()
    valid = valid.movedim(-1, -3).float()
    depth = F.pad(F.interpolate(depth, (nh, nw), mode="nearest"), pad)
    valid = F.pad(F.interpolate(valid, (nh, nw), mode="nearest"), pad)
    k = intrinsics.clone().float()
    k[:, 0] *= nw / w
    k[:, 1] *= nh / h
    k[:, 0, 2] += left
    k[:, 1, 2] += top
    yy, xx = torch.meshgrid(
        torch.arange(size, device=rgb.device), torch.arange(size, device=rgb.device), indexing="ij"
    )
    x = (xx[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None]
    y = (yy[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None]
    xyz = torch.cat((x[:, None] * depth, y[:, None] * depth, depth), 1)
    return rgb, torch.cat((xyz, valid), 1), k


class FrozenBackbone(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import Dinov2Model

        self.encoder = Dinov2Model.from_pretrained(path, local_files_only=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb):
        tokens = self.encoder(pixel_values=rgb).last_hidden_state[:, 1:]
        side = rgb.shape[-1] // self.encoder.config.patch_size
        # Use the same frozen feature precision online and in the training cache.
        return tokens.transpose(1, 2).reshape(len(rgb), -1, side, side).to(torch.float16)


def rotation_6d(value):
    a, b = value[..., :3], value[..., 3:]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack((a, b, torch.linalg.cross(a, b, dim=-1)), -1)


def z_rotation(angle):
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(*c.shape, 3, 3)


def decode(raw):
    frame = rotation_6d(raw[..., 3:9])
    angle = torch.atan2(raw[..., 9], raw[..., 10])
    panel = frame @ z_rotation(angle)
    local_rotation = rotation_6d(raw[..., 17:23])
    return dict(
        hinge_origin=raw[..., :3],
        hinge_rotation=frame,
        signed_angle=angle,
        dimensions=F.softplus(raw[..., 11:14]),
        contact_local=raw[..., 14:17],
        contact_rotation_local=local_rotation,
        panel_rotation=panel,
        contact_position=raw[..., :3] + (panel @ raw[..., 14:17, None]).squeeze(-1),
        contact_rotation=panel @ local_rotation,
        confidence=raw[..., 23].sigmoid(),
    )


class DoorEstimator(nn.Module):
    confidence_scope = STATE_CONFIDENCE

    def __init__(self, hidden=128):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(388, 64, 1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.fusion = nn.Sequential(nn.Linear(64 * 16 + 18 + 12, hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 24)
        # Start with well-conditioned rotation columns and a unit angle vector.
        # These are generic initial values, not frozen scene/asset annotations.
        with torch.no_grad():
            self.output.weight.mul_(0.01)
            self.output.bias.zero_()
            self.output.bias[[3, 7, 10, 17, 21]] = 1
            self.output.bias[23] = -2
        self.register_buffer("proprio_mean", torch.zeros(18))
        self.register_buffer("proprio_std", torch.ones(18))

    def forward(self, features, geometry, proprio, camera):
        b, t, c, h, w = features.shape
        geo = geometry.flatten(0, 1)
        # Masked local averaging retains valid metric samples at patch resolution.
        mask = F.adaptive_avg_pool2d(geo[:, 3:4], (h, w))
        xyz = F.adaptive_avg_pool2d(geo[:, :3], (h, w)) / mask.clamp_min(1e-6)
        geo = torch.cat((xyz, mask), 1)
        visual = self.spatial(torch.cat((features.flatten(0, 1).float(), geo), 1)).flatten(1)
        prop = ((proprio - self.proprio_mean) / self.proprio_std).flatten(0, 1)
        cam = camera[..., :3, :].flatten(-2).flatten(0, 1)
        fused = self.fusion(torch.cat((visual, prop, cam), -1)).reshape(b, t, -1)
        _, state = self.temporal(fused)
        # Keep checkpoint/inference layout; confidence cannot reshape geometric features.
        geometry_raw = F.linear(state[-1], self.output.weight[:23], self.output.bias[:23])
        confidence_raw = F.linear(
            state[-1].detach(), self.output.weight[23:], self.output.bias[23:]
        )
        return decode(torch.cat((geometry_raw, confidence_raw), -1))


def rotation_error(predicted, target):
    """Geodesic angle with finite zero-error gradients and no flattening near pi."""
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    return torch.atan2(skew.norm(dim=-1) / 2, cosine)


def geometry_errors(prediction, target):
    """Per-sample physical errors, shared by training, confidence labels and scoring."""
    errors = {
        key + "_m": (prediction[key] - target[key]).norm(dim=-1)
        for key in ("hinge_origin", "dimensions", "contact_local", "contact_position")
    }
    errors.update(
        {
            key + "_deg": torch.rad2deg(rotation_error(prediction[key], target[key]))
            for key in ("hinge_rotation", "contact_rotation_local", "contact_rotation")
        }
    )
    panel = target["hinge_rotation"] @ z_rotation(target["signed_angle"])
    errors["panel_rotation_deg"] = torch.rad2deg(
        rotation_error(prediction["panel_rotation"], panel)
    )
    delta = prediction["signed_angle"] - target["signed_angle"]
    errors["signed_angle_deg"] = torch.rad2deg(torch.atan2(delta.sin(), delta.cos()).abs())
    return errors


def geometry_tolerances(gates):
    """Keep the contact budgets and apply the same units to primitive-state checks."""
    position, rotation = gates["contact_position_p95_m"], gates["contact_orientation_p95_deg"]
    if not np.isfinite([position, rotation]).all() or position <= 0 or not 0 < rotation < 180:
        raise ValueError("Geometry requires positive finite physical tolerances")
    return {
        **{
            k + "_m": position
            for k in ("hinge_origin", "dimensions", "contact_local", "contact_position")
        },
        **{
            k + "_deg": rotation
            for k in (
                "hinge_rotation",
                "contact_rotation_local",
                "contact_rotation",
                "panel_rotation",
                "signed_angle",
            )
        },
    }


def usable_geometry(errors, tolerances):
    return torch.stack(
        [torch.isfinite(errors[k]) & (errors[k] <= v) for k, v in tolerances.items()]
    ).all(0)


def estimator_loss(prediction, target, available=None, *, gates):
    """Equal-weight, tolerance-normalized supervision of all articulated outputs."""
    if available is None:
        available = torch.ones_like(prediction["confidence"], dtype=torch.bool)
    mask = available.float()
    denominator = mask.sum().clamp_min(1)
    tolerances = geometry_tolerances(gates)
    errors = geometry_errors(prediction, target)
    terms = {
        k: (
            F.smooth_l1_loss(error / tolerances[k], torch.zeros_like(error), reduction="none")
            * mask
        ).sum()
        / denominator
        for k, error in errors.items()
    }
    with torch.no_grad():
        usable = usable_geometry(errors, tolerances) & available
    terms["confidence"] = F.binary_cross_entropy(prediction["confidence"], usable.float())
    return sum(terms.values()), terms


@dataclass(frozen=True)
class DoorEstimate:
    timestamp_s: float
    valid: bool
    reason: str
    confidence: float = 0.0
    frame: ObjectFrame | None = None
    panel_rotation: np.ndarray | None = None
    signed_angle: float | None = None
    dimensions: np.ndarray | None = None
    contact_position: np.ndarray | None = None
    contact_rotation: np.ndarray | None = None

    def fresh(self, now_s, max_age_s=0.15):
        return self.valid and 0 <= now_s - self.timestamp_s <= max_age_s


class ObservedEstimator:
    """Fixed-window inference; reset/loss cannot retain a privileged or stale frame."""

    def __init__(self, backbone, estimator, config):
        from collections import deque

        if getattr(estimator, "confidence_scope", None) != STATE_CONFIDENCE:
            raise ValueError("Observed inference requires articulated-state confidence")
        self.backbone, self.estimator, self.config = backbone, estimator, config
        self.estimator.eval()
        self.history = deque(maxlen=config["history"])
        self.reset()

    def reset(self):
        self.history.clear()
        self.last_time = None
        self.last_frame = None

    @torch.no_grad()
    def update(self, observation):
        t, frame = float(observation["time_s"]), int(observation["frame"])
        if self.last_time is not None and (t <= self.last_time or frame <= self.last_frame):
            self.reset()
            return DoorEstimate(t, False, "nonmonotonic_observation")
        if self.last_time is not None and t - self.last_time > self.config["max_gap_s"]:
            self.reset()
        if (
            not torch.isfinite(observation["joint_position"]).all()
            or not torch.isfinite(observation["joint_velocity"]).all()
        ):
            self.reset()
            return DoorEstimate(t, False, "invalid_proprioception")
        depth, valid = observation["depth_m"], observation["valid_depth"]
        if not (valid & torch.isfinite(depth) & (depth > 0)).any():
            self.reset()
            return DoorEstimate(t, False, "missing_depth")
        if self.last_time is not None and t - self.last_time < 1 / self.config["sample_hz"] - 1e-6:
            return DoorEstimate(t, False, "between_inference_ticks")
        self.last_time, self.last_frame = t, frame
        rgb, geometry, _ = preprocess(
            observation["rgb"][None],
            depth[None],
            valid[None],
            observation["intrinsics"][None],
            self.config["image_size"],
        )
        features = self.backbone(rgb)[0]
        prop = torch.cat((observation["joint_position"], observation["joint_velocity"]))
        self.history.append((features, geometry[0], prop, observation["camera_world"]))
        if len(self.history) < self.config["history"]:
            return DoorEstimate(t, False, "warming_up")
        inputs = [torch.stack([item[i] for item in self.history])[None].float() for i in range(4)]
        predicted = self.estimator(*inputs)
        confidence = float(predicted["confidence"][0])
        if not all(torch.isfinite(v).all() for v in predicted.values()):
            self.reset()
            return DoorEstimate(t, False, "nonfinite_estimate")
        if confidence < self.config["confidence_threshold"]:
            return DoorEstimate(t, False, "low_confidence", confidence)
        values = {k: v[0].cpu().numpy() for k, v in predicted.items()}
        if validate_object_frame(ObjectFrame(values["hinge_origin"], values["hinge_rotation"])):
            self.reset()
            return DoorEstimate(t, False, "invalid_frame")
        return DoorEstimate(
            t,
            True,
            "observed",
            confidence,
            ObjectFrame(values["hinge_origin"], values["hinge_rotation"]),
            values["panel_rotation"],
            float(values["signed_angle"]),
            values["dimensions"],
            values["contact_position"],
            values["contact_rotation"],
        )
