# Decision — Calibrated Position-Only Alex V2 Execution

> Applies to the retired B0 Alex V2 implementation only. The approved
> [[decisions/visuoproprioceptive-generalization-benchmark|B1 design]] supersedes
> this execution choice with seven-joint Purdue tool-pose control. Purdue control is implemented; do not reinterpret B0 results as seven-joint
> or orientation-controlled. The B0 source is historical at Git `9c16e3d`.

## Context

The Alex V2 benchmark needs a reproducible controller at the physical gripper contact point while preserving the A2/A3 representation contract.

## Decision

Use the fixed-base Alex V2 torso, the six calibrated right-arm joints, and position-only differential IK at the collision-derived tool point. Keep rotational A2/A3 components in data and adapter decisions, but do not command them to the robot.

The historical implementation used one task calibration and external Alex construction. Its calibration, manifests and B0 runtime were retired.

Accept task force only from exact-door raw PhysX contact selection. Geometric contact may be recorded for diagnosis but cannot replace sensed force.

## Consequences

- The historical B0 execution is intentionally translation-only.
- Calibration and contact checks were required before interpreting B0 results.
- The controller is specialized to the single-environment simulated Alex V2 benchmark.
- Simulator force thresholds and success do not establish physical-robot safety.

The retired calibration-authoring, generic executor, sensorless, and surrogate-robot paths are not part of this decision.

## Version Notes

- 2026-08-13 — Restated the active decision around one calibration, position-only tool-point IK, and exact-door contact sensing.
