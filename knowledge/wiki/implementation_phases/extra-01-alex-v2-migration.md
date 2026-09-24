# Extra 01 — Alex V2 Migration

> Historical phase record. The historical B0 contract is documented in [[topics/alex-v2-benchmark|Alex V2 Benchmark]].

## Objective

Replace provisional robot assumptions with the fixed-base IHMC Alex V2 door benchmark.

## Subphase E1.1 — Asset, Calibration, and Execution

#### Implementation

This work validated the Alex V2 asset and joint order, derived the collision tool point, introduced offset-point Jacobian control, and added door-panel force sensing.

Generic robot construction moved to the external Alex package. B0 task calibration, manifests and execution are historical at Git `9c16e3d`; Purdue replaces that runtime.

#### Key Decisions

- Requested rotation remains represented but is not actuated.
- The historical task used one calibration, with exact-panel contact filtering.

#### Problems / Limitations

- Calibration and force evidence are simulation-specific.
- The completed calibration-authoring and generic executor layers were removed.

## Artifacts

The calibration and D0–D4 layers were retired. Git retains their source; scientific results remain in the experiment pages.

## Files

No B0-specific implementation remains active.
