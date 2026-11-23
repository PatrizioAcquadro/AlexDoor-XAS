# TSDF Rebuild Double Buffer

## Question and method

Can two reusable CPU snapshot pairs retain successful/failure-state semantics and produce a useful
operational rebuild saving?

The recovered-source candidate uses a spare TSDF/color pair, publishing only complete arrays and
restoring scratch destinations in `finally`. Fixed-input, error-state, repeated-memory and four
timed-provider gates are independent. Local/full correctness milestones are `cc920dd` /`806976e`;
memory and adoption criteria are `a633202`.

## Result and decision

Exact local/full and repeated-memory gates pass. Captured rebuilds improve 35.2% light / 28.8%
nominal with bounded temporary memory.

Timed ordinary service/age p95 worsens 1.6%/1.2% light and 6.6%/5.8% nominal. Nominal correct 20 mm
target counts fall from 195/194/194 to 163/163/162 of 2,858 opening captures; native losses rise
0→27 and final-five-second correct counts fall 23→3 per target. Light quality improves but retains
52 native losses and almost no final correct support. The nominal buffered history has no post-seed
expansion, so its shorter maximum does not measure an optimized rebuild.

**Do not integrate the prototype.** Keep the maintained TSDF baseline: local savings and numerical
equivalence do not satisfy the predeclared operational benefit/quality requirement.

## Limits and evidence

Original worktree removal lost earlier raw local/interrupted prefixes; source/session extracts
survive in `recovery.json` /`original-tmp/`. Original reported 388/426-frame interruptions are not
preserved raw measurements. The later seven-cycle RustDesk-interrupted memory attempt is retained.
Timed seeds/history differ, so quality changes do not prove an arithmetic defect. No
hardware/contact or indefinite stability qualification follows.

Evidence: `outputs/evidence/b1/perception/tsdf-double-buffer-recovery-20261015/`;
`complete-summary.json`, `operational-summary.json`, `continuation-operational-detail.json`,
`decision.json`, recovered sources and per-run quality/reset records. Final decision: `85b5719`.
