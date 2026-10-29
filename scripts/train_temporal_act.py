#!/usr/bin/env python3
"""Bounded CUDA temporal ACT A1 learning with saved, resumable checkpoints."""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.assets.identity import RobotAssetRef  # noqa: E402
from alexdoor_xas.dataset.normalize import DatasetNormStats  # noqa: E402
from alexdoor_xas.dataset.sampling import collate_torch  # noqa: E402
from alexdoor_xas.policies.act.model import act_loss  # noqa: E402
from alexdoor_xas.policies.act.temporal import (  # noqa: E402
    TemporalActPolicy,
    fit_temporal_stats,
    save_temporal_policy,
    temporal_model,
    validate_training_sources,
)
from alexdoor_xas.policies.act.temporal_training import (  # noqa: E402
    TemporalTrainingSet,
    build_training_index,
)
from alexdoor_xas.policies.act.train import evaluate_l1  # noqa: E402
from alexdoor_xas.policies.common.data import to_device as _to_device  # noqa: E402
from alexdoor_xas.policies.common.training import (  # noqa: E402
    capture_rng_states,
    restore_rng_states,
    torch_save_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-handoff", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wall-seconds", type=float, default=900)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--diagnostic-batches", type=int, default=4)
    parser.add_argument(
        "--stop-file", type=Path, help="Save and stop at an update boundary when present"
    )
    parser.add_argument(
        "--focus-episode",
        action="append",
        default=[],
        help="Half-batch focus on an admitted full-cycle train episode; repeatable",
    )
    args = parser.parse_args()
    started = time.monotonic()
    if args.index_handoff:
        build_training_index(args.index_handoff, args.manifest)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required; no CPU fallback")
    if (
        args.output is None
        or min(args.updates, args.save_every, args.wall_seconds, args.diagnostic_batches) <= 0
    ):
        parser.error("Positive limits and --output required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    train = TemporalTrainingSet(
        args.manifest, data_root=args.data_root, focus_episode_ids=args.focus_episode
    )
    dev = TemporalTrainingSet(args.manifest, split="development", data_root=args.data_root)
    stats = (
        DatasetNormStats.from_dict(json.loads(args.normalization.read_text()))
        if args.normalization
        else fit_temporal_stats(train.episodes)
    )
    selected_ids = validate_training_sources(stats, [e.builder.episode_id for e in train.episodes])
    (args.output / "normalization.json").write_text(json.dumps(stats.to_dict(), indent=2))
    torch.manual_seed(args.seed)
    model = temporal_model(train.episodes[0].config, seed=args.seed).cuda()
    robot = RobotAssetRef.from_dict(train.episodes[0].metadata["robot_asset"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    step = 0
    if args.resume:
        # RNG byte states must remain on CPU; load_state_dict moves model/optimizer tensors.
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if previous["seed"] != args.seed or previous["stats"] != stats.to_dict():
            raise ValueError("Resume seed/normalization mismatch")
        previous_ids = validate_training_sources(stats, previous.get("training_episode_ids"))
        if not set(previous_ids) <= set(selected_ids):
            raise ValueError("Resume cannot discard previously selected training sources")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        restore_rng_states(previous["rng"])
        step = previous["step"]
    updates_this_run = 0
    recent_losses = deque(maxlen=256)
    log = (args.output / "progress.jsonl").open("x", buffering=1)
    result = dict(
        seed=args.seed,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        start_epoch_s=time.time(),
        initial_step=step,
        updates_this_run=0,
        recipe=train.recipe,
        focus_episode_ids=list(train.focus_episode_ids),
        batch_size=args.batch_size,
        training_episode_ids=list(selected_ids),
        normalization_fit_episode_ids=list(stats.train_episode_ids),
        final_test_opened=False,
        learned_success=False,
        diagnostic_batches=args.diagnostic_batches,
        stop_file=None if args.stop_file is None else str(args.stop_file),
    )
    (args.output / "run.json").write_text(json.dumps(result, indent=2))

    def save():
        checkpoint = args.output / f"checkpoint-{step:06d}.pt"
        save_temporal_policy(checkpoint, model, stats, robot, training_episode_ids=selected_ids)
        policy = TemporalActPolicy.from_checkpoint(
            checkpoint, config=train.episodes[0].config, runtime_asset=robot
        )
        sample = train.episodes[0].sample(train.records[0]["rows"][0])
        original = TemporalActPolicy(model, stats)
        original.reset(sample["episode_id"], sample["generation"])
        policy.reset(sample["episode_id"], sample["generation"])
        import numpy as np

        np.testing.assert_allclose(
            original.predict(sample), policy.predict(sample), atol=1e-7, rtol=0
        )
        torch_save_atomic(
            args.output / "resume.pt",
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=capture_rng_states(),
                step=step,
                seed=args.seed,
                stats=stats.to_dict(),
                training_episode_ids=list(selected_ids),
            ),
        )
        return str(checkpoint)

    def diagnose():
        # Compare fixed balanced causal draws without perturbing training RNG.
        rng = capture_rng_states()
        try:
            return dict(
                step=step,
                training_l1_diagnostic=evaluate_l1(
                    model,
                    (
                        train.batch(9300 + i, batch_size=8, stats=stats)
                        for i in range(args.diagnostic_batches)
                    ),
                    torch.device("cuda"),
                ),
                development_l1_diagnostic=evaluate_l1(
                    model,
                    (
                        dev.batch(9200 + i, batch_size=8, stats=stats)
                        for i in range(args.diagnostic_batches)
                    ),
                    torch.device("cuda"),
                ),
            )
        finally:
            restore_rng_states(rng)

    try:
        if args.diagnostic_batches != 4:
            log.write(json.dumps(dict(diagnose(), initial_diagnostic=True)) + "\n")
        while step < args.updates and time.monotonic() - started < args.wall_seconds - 60:
            if args.stop_file is not None and args.stop_file.exists():
                break
            begin = time.monotonic()
            batch = train.batch(args.seed * 1000000 + step, batch_size=args.batch_size, stats=stats)
            tensors = _to_device(collate_torch(batch), torch.device("cuda"))
            model.train()
            prediction, mu, logvar = model(tensors["obs"], tensors["actions"], tensors["is_pad"])
            losses = act_loss(prediction, tensors["actions"], tensors["is_pad"], mu, logvar, 10.0)
            if not torch.isfinite(losses["loss"]):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            updates_this_run += 1
            row = dict(
                step=step,
                duration_s=time.monotonic() - begin,
                **{k: float(v.detach()) for k, v in losses.items()},
            )
            recent_losses.append(row["loss"])
            if step % args.save_every == 0 or step == args.updates:
                row["training_loss_rolling"] = sum(recent_losses) / len(recent_losses)
                row["checkpoint"] = save()
                row.update(diagnose())
            log.write(json.dumps(row) + "\n")
            if step % 16 == 0:
                print(json.dumps(row), flush=True)
        result["checkpoint"] = save()
        result["stop_reason"] = (
            "monitor_stop"
            if args.stop_file is not None and args.stop_file.exists()
            else "update_limit"
            if step >= args.updates
            else "wall_limit"
        )
    except BaseException as error:
        result["stop_reason"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        result.update(
            final_step=step,
            updates_this_run=updates_this_run,
            whole_process_s=time.monotonic() - started,
            end_epoch_s=time.time(),
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        log.close()
        train.close()
        dev.close()


if __name__ == "__main__":
    main()
