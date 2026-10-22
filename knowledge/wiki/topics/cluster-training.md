# Official NVIDIA Cluster Training

## Runtime and application environments

The shared runtime is the unmodified NVIDIA
`nvcr.io/nvidia/isaac-lab:3.0.0-kitless` image, converted to SIF using NVIDIA's
[NGC/Singularity procedure](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html#running-singularity-containers).
The [NVIDIA installation guide](https://docs.nvidia.com/isaaclab/isaac-lab/installation)
documents this image and a Linux driver minimum of `580.65.06`.
[NVIDIA's HPC guide](https://docs.nvidia.com/isaaclab/features/running-on-hpc-clusters)
allows cluster-specific scheduler resources and storage bindings.

One image is shared through `/depot/euge/data/pacquadr/nvidia-robot-learning/`.
Application packages live in separate ordinary uv virtual environments. They do
not replace packages in the NVIDIA image. The workstation's Isaac Sim 6.0.1 /
Isaac Lab beta2 runtime and existing cluster environments are preserved.

| Component | Path below the shared root |
|---|---|
| NVIDIA image | `images/isaac-lab-3.0.0-kitless.sif` |
| AlexDoor Python | `apps/alexdoor-xas-20261022/bin/python` |
| SquadBot Python | `apps/squadbot-av-20261022/bin/python` |
| AlexDoor source | `sources/alexdoor-xas-f036e13` |
| SquadBot source | `sources/squadbot-av-c8b02af` |
| Installed application requirements | `requirements/` |

Source snapshots retain their Git history and exact commits. They were installed
editable in their respective environments, and both checkouts remained clean.
The local working trees were not switched, reset or edited for installation.

## Verified on 2026-10-22

| Check | Gilbreth A100 80 GB | Gautschi H100 80 GB |
|---|---|---|
| Node / driver | `gilbreth-k000` / `595.91.07` | `h002` / `595.71.05` |
| Official Newton Cartpole PPO | 16 environments, 5 iterations, saved checkpoints | Same bounded training passed |
| Official Newton Warp RGB-camera PPO | 16 environments, 5 iterations, saved checkpoints | Additional allocation unavailable; not run |
| Official OvPhysX zero-action Cartpole | 128 environments, 64 steps | Additional allocation unavailable; not run |
| AlexDoor training/resume and B1 model contracts | 12 GPU tests passed | 2 selected training/resume tests passed |
| SquadBot structured model/runtime contracts | 11 GPU tests passed | Same 11 GPU tests passed |
| SquadBot full sensory trainer on existing approved cache | One actual optimizer update and saved checkpoint | Not run |

CUDA device selection was verified in the NVIDIA runtime and both application
environments. Cartpole checkpoint inspection found changed actor/critic weights
and positive optimizer counters. These short checks establish runtime behavior,
not policy quality or completed robot-task training.

The sensory check used the existing `sensory-features-01` cache, complete
`train-00/C-left` and `train-00/M-left` episodes, seed 17, one update and a
train-only checkpoint check. Original masks, statistics, data and campaign rules
were preserved; the one-update rules existed only in this separate runtime
qualification invocation. The saved checkpoint is under
`qualification/a100-sensory-real-one-update/`. A subsequent GPU reload allocation
was unavailable, so strict sensory reload remains unverified.

Gautschi's initial allocation was unavailable; a later allocation completed its
Newton and application checks. Further H100 camera/OvPhysX/sensory and A100 sensory
reload requests also expired with busy nodes. These are unavailable checks, not
model failures. No such request remains pending. Other users' or chats' jobs were
not modified.

## Dependency isolation

Both application environments use Python 3.12.3, PyTorch `2.11.0+cu130` and
Torchvision `0.26.0+cu130`. AlexDoor installs its declared Diffusion extra
(`diffusers 0.41.0`); SquadBot installs its learning/sensory extras with
`transformers 4.57.6` and `protobuf 5.29.6`. Both environments pass `uv pip check`.
Application extras follow the existing project requirements; PyTorch/Torchvision
match the image's versions, and pytest supplies the existing checks. No project
requirements were changed.

This separation matters: the NVIDIA image itself contains Transformers 5.10.4
and Protobuf 7.36.0rc1 and lacks Diffusers and OpenCV. Changing those packages
inside the image would couple project requirements to the simulator runtime.
Its source `VERSION` is `3.0.0`; its Python distribution metadata reports
`isaaclab 28.0.0`. These upstream values were recorded verbatim, not repaired.

## Running a job

Use SSH aliases `gilbreth-purdue` or `gautschi-purdue` and account `euge`.
Resources used successfully were:

- Gilbreth: partition `a100-80gb`, QOS `normal`, one GPU, eight CPUs and 96 GB RAM.
- Gautschi: partition `ai`, QOS `normal`, one GPU and fourteen CPUs; site-default memory.

For example, on the Gilbreth login node, run this bounded NVIDIA example through
Slurm:

```bash
runtime=/depot/euge/data/pacquadr/nvidia-robot-learning
mkdir -p "$HOME/nvidia-cartpole-logs"
srun -A euge -p a100-80gb -q normal -N 1 -n 1 \
  --cpus-per-task=8 --gpus-per-node=1 --mem=64G --time=00:10:00 \
  singularity exec --nv \
  --bind "$HOME/nvidia-cartpole-logs:/workspace/isaaclab/logs" \
  --pwd /workspace/isaaclab \
  "$runtime/images/isaac-lab-3.0.0-kitless.sif" \
  isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
  --num_envs 16 presets=newton_mjwarp --max_iterations 5
```

For a project, keep the same image and replace `isaaclab ...` with that project's
Python executable and existing trainer arguments. Set the working directory to
the corresponding source snapshot. SquadBot's maintained entry points are
`tools/policy/train_sensory_policy.py` and
`tools/policy/train_structured_policy.py`; each requires its approved cache,
seed, configuration and a new output directory. Follow the scientific protocol
for full training budgets; the one-update qualification is not a new campaign.

## Simulation boundaries and evidence

NVIDIA's [Newton Warp renderer](https://docs.nvidia.com/isaaclab/concepts/renderers)
provides lightweight camera outputs. [OvPhysX](https://docs.nvidia.com/isaaclab/concepts/ov-phys-x)
provides kit-less PhysX. Neither makes the current Alex003 Isaac Sim, ZED RGB-D or
audio lifecycle code automatically portable. A project migration needs the
[official migration procedure](https://docs.nvidia.com/isaaclab/isaac-lab/migrating-to-3-0)
and task/contact/sensor validation. The `isaac-audio-sensors` SDK was inspected;
its live Isaac Sim integration was not migrated or qualified on these clusters.

Logs, package lists, checkpoints and the exact availability outcomes are under
`outputs/generated/nvidia-cluster-readiness-20261022/` locally and corresponding
qualification directories on each cluster. These local outputs and `TODO.md`
are ignored by Git. Local commits preserve the canonical documentation only.
