# Official SAM3 Compilation

2026-10-15; baseline `85b5719` on `PatrizioAcquadro/b1-observed-a3a4`, initially clean.
One experimental variant, disabled in the selected recipe: official video-builder
`compile=True`. The installed SAM3 revision exposes lazy component compilation in
`Sam3VideoInference._compile_model()`, reached by the causal base-class one-frame
propagation after an eager seed. The adapter raises compiler errors instead of
accepting the official setup's `suppress_errors=True` fallback. Component choices,
compiler modes and successful inference calculations remain official.

The [[tsdf-rebuild-double-buffer|TSDF baseline decision]] remains in force. Models,
weights, BF16/TF32, resolution, prompts, thresholds, persistent identity/session,
reconditioning OFF, bounded history, TAPIR, SuperPoint and P2P are unchanged.
No controller, policy, training or contact work is included.

## Preset evaluation

Persistent evidence and executable harnesses:
`outputs/evidence/b1/perception/sam3-compile-20261015/`.
`criteria.md` records the detailed gates and budgets before CUDA acquisition.
Compatibility and paired dense/sparse pilots precede any conditional complete
fixed-input and timed comparison. The shared GPU is reserved with both common
locks; foreign workloads are never interrupted and interference attempts remain.

Warmup uses a disposable causal state and only the first arrived RGB, with the
same prompt and thresholds. RNG state is restored and evaluated inference starts
with empty state, timestamps and IDs. Future observations and truth never enter
warmup. Startup, warmup/compilation and steady service are separate measurements.

The practical pilot benefit gate is 15% median / 10% p95 SAM3 improvement in both
sparse conditions without a dense regression above 10%. Mask/identity, bounded
compilation/memory and unchanged downstream P2P quality are independent gates;
internal bit equality is not required. Conditional complete runs score the three
frozen physical targets at 10/15/20 mm and 5 degrees, retain scheduled/processed
availability, publication ages, frequency, replacements, loss and gaps. Different
timed seeds/histories cannot establish a causal compiler effect on quality.
No requirement that this one change reach 150 ms.

## Current state

Implementation and preset protocol prepared; CUDA compatibility/pilot pending an
exclusive shared-GPU window. No performance or quality benefit is yet demonstrated.
