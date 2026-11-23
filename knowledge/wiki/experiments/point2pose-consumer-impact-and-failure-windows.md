# Point2Pose Consumer Impact and Two Failure Windows

## Question and method

How do saved native tracking errors affect the observed zone, its normal and conditional downstream
targets, and where do failures arise?

Saved-data diagnosis `28626c1` compares original (`59abf31`) and bounded (`2b127b4`) histories: all
2,858 rows and five/six candidates per condition. No inference or policy run occurs. Evaluator truth
transports the immutable seed zone and saved targets; it never enters runtime decisions.

## Result and decision

Zone-point, oriented-normal and full-rotation errors are distinct. Conditional teacher-target
transport assumes the intended local target; it is neither actual A4 prediction error nor complete
A3 reference error.

| Primary condition/system | Accepted rows | Zone point p95 / max, mm | Normal p95 / max, degrees | Conditional teacher target p95 / max, mm |
|---|---:|---:|---:|---:|
| light baseline | 2,311 | 7.79 / 22.37 | 1.18 / 6.22 | 15.38 / 48.28 |
| light bounded | 2,791 | 15.43 / 45.17 | 1.20 / 5.17 | 16.66 / 23.18 |
| nominal baseline | 1,582 | 10.20 / 36.07 | 1.18 / 9.06 | 11.91 / 69.01 |
| nominal bounded | 2,282 | 9.28 / 178.61 | 0.40 / 29.81 | 7.28 / 116.23 |

At nominal 45.6667 s graph publication changes frontend 2.92 mm/0.58 degrees to 178.61 mm/32.73
degrees. Published map consistency can coexist with incorrect original material geometry. Light
deterioration at 52.1333 s exposes changed reference/correspondence support rather than an isolated
composition error. No better intermediate pose is substituted.

Keep baseline and bounded failures distinct. Investigate publication/support consistency and
material evidence during replacement before choosing a further intervention.

## Limits and evidence

The interrupted unbounded histories retain 1,506/2,239 missing rows. Relative alignment, apparent
next-frame recovery and conditional targets do not supply closed reference, physical hinge,
original-material identity, loaded contact, hardware calibration or 150 ms publication.

Evidence: `outputs/evidence/b1/perception/point2pose-consumer-diagnosis-01/`, saved evaluators,
original per-candidate metrics and `nominal-three-frames.png` / `light-three-frames.png`. Git:
`28626c1`; source boundary `3430808`.
