#!/usr/bin/env python
"""Prepare frozen features, verify readiness, train or evaluate the shared B1 estimator."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "train", "launch", "evaluate"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/perception.json")
    parser.add_argument(
        "--training-config", type=Path, default=REPO / "configs/perception_training.json"
    )
    parser.add_argument(
        "--recordings", type=Path, default=REPO / "datasets/b1/perception/engineering-v1"
    )
    parser.add_argument(
        "--features", type=Path, default=REPO / "datasets/b1/perception/features-v1"
    )
    parser.add_argument("--backbone", type=Path, default=REPO / "outputs/b1/perception/backbone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--hours", type=float)
    parser.add_argument(
        "--partial", action="store_true", help="Diagnostic check only; no readiness claim"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error(
            "A CUDA GPU is required; rerun with runtime permissions, never fall back to CPU"
        )
    if args.partial and args.command != "check":
        parser.error("Partial is only supported for diagnostic checks")
    config = json.loads(args.config.read_text())
    if (config["feature_precision"], config["geometry_precision"]) != ("float16", "float32"):
        parser.error("The frozen recipe requires float16 RGB features and float32 geometry")
    from alexdoor_xas.perception.data import (
        PerceptionWindows,
        episode_paths,
        prepare_features,
        validate_feature_corpus,
    )
    from alexdoor_xas.perception.model import (
        DoorEstimator,
        FrozenBackbone,
        estimator_loss,
        preprocess,
    )
    from alexdoor_xas.perception.training import evaluate, load_checkpoint, train
    from alexdoor_xas.qualification.corpus import load_corpus

    corpus = load_corpus(REPO / "assets/doors/b1/corpus.json", REPO)
    paths = episode_paths(args.recordings, corpus, complete_campaign=not args.partial)
    if args.command in ("train", "launch", "evaluate") or (
        args.command == "check" and not args.partial
    ):
        validate_feature_corpus(args.features, paths, config, args.backbone)
    if args.command in ("train", "launch"):
        from alexdoor_xas.perception.run import EarlyStopping, launch_detached

        stopping = json.loads(args.training_config.read_text())
        EarlyStopping(stopping)
        hours = args.hours if args.hours is not None else config["max_hours"]
        if not 0 < hours <= 10:
            parser.error("Training budget must be in (0, 10] hours")
        if args.command == "launch":
            launcher = (
                Path(os.environ.get("ISAAC_LAB_DIR", Path.home() / "IsaacLab")) / "isaaclab.sh"
            )
            if not launcher.is_file():
                parser.error("Set ISAAC_LAB_DIR to the supported Isaac Lab checkout")
            # Explicit resolved paths preserve the caller's choices in the detached worker.
            command = [str(launcher), "-p", str(Path(__file__).resolve()), "train"]
            for key in (
                "config",
                "training_config",
                "recordings",
                "features",
                "backbone",
                "output",
            ):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key).resolve())])
            command.extend(["--hours", str(hours), "--device", args.device])
            if args.resume is not None:
                command.extend(["--resume", str(args.resume.resolve())])
            record = launch_detached(command, args.output, REPO, resume=args.resume is not None)
            print(json.dumps(record, indent=2))
            return
    if args.command in ("prepare", "check"):
        backbone = FrozenBackbone(args.backbone).to(args.device)
        if args.command == "prepare":
            prepare_features(paths, args.features, backbone, config, args.device)
            report = dict(
                prepared=True,
                episodes=len(paths),
                features=str(args.features),
                training_started=False,
            )
        else:
            import h5py

            from alexdoor_xas.perception.data import TARGETS

            torch.manual_seed(config["seed"])
            head = DoorEstimator(config["hidden_size"]).to(args.device).eval()
            if not args.partial:
                train_windows = PerceptionWindows(args.features, "train", config)
                mean, std = train_windows.normalization()
                head.proprio_mean.copy_(torch.as_tensor(mean, device=args.device))
                head.proprio_std.copy_(torch.as_tensor(std, device=args.device))
            before = {k: v.clone() for k, v in head.state_dict().items()}
            with h5py.File(paths[0], "r") as source, torch.no_grad():
                stride = round(
                    1
                    / config["sample_hz"]
                    / json.loads(source["metadata"].attrs["episode"])["control_dt"]
                )
                ids = [i * stride for i in range(config["history"])]
                obs = source["observations"]
                values = {
                    k: torch.as_tensor(obs[k][ids], device=args.device)
                    for k in (
                        "rgb",
                        "depth_m",
                        "valid_depth",
                        "intrinsics",
                        "joint_position",
                        "joint_velocity",
                        "camera_world",
                    )
                }
                rgb, geometry, _ = preprocess(
                    values["rgb"],
                    values["depth_m"],
                    values["valid_depth"],
                    values["intrinsics"],
                    config["image_size"],
                )
                features = backbone(rgb)
                predicted = head(
                    features[None],
                    geometry[None],
                    torch.cat((values["joint_position"], values["joint_velocity"]), -1)[
                        None
                    ].float(),
                    values["camera_world"][None].float(),
                )
                labels = {
                    k: torch.as_tensor(
                        source["annotations/" + k][ids[-1]], device=args.device, dtype=torch.float32
                    )[None]
                    for k in TARGETS
                }
                loss, terms = estimator_loss(predicted, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite forward/loss")
            if any(not torch.equal(before[k], v) for k, v in head.state_dict().items()):
                raise ValueError("Readiness check changed estimator state")
            if any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
                raise ValueError("Backbone is not frozen")
            report = dict(
                ready_for_training=not args.partial,
                partial_diagnostic=args.partial,
                episodes=len(paths),
                device=torch.cuda.get_device_name(),
                loss=float(loss),
                terms={k: float(v) for k, v in terms.items()},
                backbone_frozen=True,
                estimator_weights_unchanged=True,
                training_started=False,
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6,
                development_passed=False,
                dynamic_validation="pending",
            )
    else:
        train_data = PerceptionWindows(args.features, "train", config)
        development = PerceptionWindows(args.features, "development", config)
        if args.command == "train":
            train(
                train_data,
                development,
                config,
                args.output,
                stopping=stopping,
                hours=hours,
                resume=args.resume,
                device=args.device,
            )
            return
        if args.checkpoint is None:
            parser.error("Evaluation requires --checkpoint")
        model, payload = load_checkpoint(args.checkpoint, config, args.device)
        current = json.loads((args.features / "index.json").read_text())
        if payload["dataset"] != current:
            raise ValueError("Checkpoint and feature dataset differ")
        report = evaluate(model, development, config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
