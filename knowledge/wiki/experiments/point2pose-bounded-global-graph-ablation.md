# Point2Pose Bounded Renewal Without the Global Graph

Full-sequence CUDA experiment on 2026-10-06 from clean `main @1d57f7b`; implementation `0e6185f`. One graph-off variant and exactly one fresh attempt per original light/nominal recording. All earlier baselines, failures and raw recordings remain preserved.

## Method and controls

The sole native configuration delta from bounded renewal (`2b127b4`) is `pipeline.params.use_key_frame_graph=true → false`. The default runtime remains graph-on. Models, automatic candidate/point selection, inlier criterion, all thresholds, 120-reference budget, replacement/promotion/compaction and SDF rules are unchanged. Native RANSAC starts at zero. Original initialization, masks, maps, IDs, tracks and poses match within each condition. All-candidate pose/decision prefixes are identical for nine light rows and 155 nominal rows, first diverging at 31.15/33.5833 s; subsequent graph/map/correspondence and shared RNG histories may diverge. This measures the total graph intervention along the sequence, rather than one isolated graph call.

Each CUDA process starts at the original 31 s seed and processes every original 60 Hz acquisition through 78.6167 s: 2,858 scheduled rows per condition, no reset, skipped frame, process retry or acquisition. Five light and six nominal candidates retain candidate-0 as primary. Both attempts complete. The recorded maximum is 63.7209 degrees; the 90.7-degree joint limit is unrecorded. Different lighting seeds prevent causal lighting attribution.

The existing evaluators score original-time accepted correct poses (≤1 cm/5 degrees), inaccurate acceptances, all finite/lost/rejected poses, measured-zone point/oriented normal, conditional exact saved teacher targets, strict historical material audits, references and memory. Error statistics exclude constructed seed zero; every scheduled row stays in availability denominators. The terminal observation has no next command. Expected rigid geometry/targets remain evaluator-only. Actual full A3 reference errors and A4 rollouts remain unavailable. See [[point2pose-consumer-impact-and-failure-windows|consumer metric definitions]].

## Complete primary outcomes

| Condition/system | Correct accepted poses | Inaccurate acceptances | Native-lost rows | Accepted point p95 / max, mm | Normal p95 / max, degrees | Conditional target p95 / max, mm |
|---|---:|---:|---:|---:|---:|---:|
| light / baseline | 2,276 | 35 | 531 | 7.79 / 22.37 | 1.18 / 6.22 | 15.38 / 48.28 |
| light / bounded | 2,250 | 541 | 65 | 15.43 / 45.17 | 1.20 / 5.17 | 16.66 / 23.18 |
| light / graph-off | 2,569 | 238 | 49 | 11.51 / 21.05 | 0.05 / 2.77 | 13.95 / 25.39 |
| nominal / baseline | 1,494 | 88 | 1,274 | 10.20 / 36.07 | 1.18 / 9.06 | 11.91 / 69.01 |
| nominal / bounded | 2,183 | 99 | 568 | 9.28 / 178.61 | 0.40 / 29.81 | 7.28 / 116.23 |
| nominal / graph-off | 2,004 | 58 | 794 | 8.15 / 27.52 | 0.05 / 10.71 | 10.54 / 80.78 |

| Condition/reference | Correct pose gains / losses at identical timestamps | Net correct available poses | Correct finite-pose gains / losses, including lost |
|---|---:|---:|---:|
| light / baseline | 505 / 212 | +293 | 56 / 241 |
| light / bounded | 455 / 136 | +319 | 461 / 128 |
| nominal / baseline | 669 / 159 | +510 | 235 / 343 |
| nominal / bounded | 65 / 244 | -179 | 91 / 258 |

| Condition/system, all finite non-seed poses | Point p95 / max, mm | Normal p95 / max, degrees | Full rotation p95 / max, degrees | Conditional target p95 / max, mm |
|---|---:|---:|---:|---:|
| light / baseline | 8.24 / 22.46 | 1.18 / 6.22 | 1.44 / 6.22 | 15.55 / 48.28 |
| light / bounded | 15.71 / 45.17 | 1.21 / 5.17 | 1.28 / 5.45 | 16.71 / 23.18 |
| light / graph-off | 11.59 / 21.05 | 0.05 / 2.78 | 0.54 / 2.80 | 13.97 / 25.39 |
| nominal / baseline | 34.39 / 49.83 | 2.65 / 9.06 | 2.65 / 9.07 | 16.94 / 69.01 |
| nominal / bounded | 19.02 / 178.61 | 2.07 / 29.81 | 2.42 / 32.73 | 11.96 / 116.23 |
| nominal / graph-off | 100.30 / 115.94 | 7.69 / 10.71 | 7.69 / 10.72 | 34.90 / 88.95 |

## Full-window memory and retained history

Memory is the sampled native-process GPU peak at 100 ms, separate from PyTorch step allocation and derived TSDF buffer size. It excludes other desktop processes; it is not an exact instantaneous device peak. CPU RSS and voxel surface fields were not recorded.

| Condition/system | Sampled process peak, GiB | PyTorch step peak, GiB | Maximum simultaneous active / historical references | Retained graph poses | Maximum TSDF buffers, GiB |
|---|---:|---:|---:|---:|---:|
| light / baseline | 6.58 | 4.17 | 420 / 420 | 14 | 0.55 |
| light / bounded | 7.33 | 5.15 | 512 / 2094 | 70 | 0.55 |
| light / graph-off | 7.78 | 5.27 | 520 / 2670 | 0 | 0.58 |
| nominal / baseline | 6.52 | 4.20 | 420 / 420 | 14 | 0.63 |
| nominal / bounded | 9.60 | 5.88 | 654 / 1830 | 60 | 1.85 |
| nominal / graph-off | 9.61 | 5.74 | 630 / 1890 | 0 | 1.83 |

Graph-off light: 89 retained all-candidate keyframes; 570 primary reference births and 478 retirements. No already-confirmed primary landmark changes; 540 pending-to-confirmed coordinate updates. Pending-to-confirmed coordinate updates remain native promotion; they are not graph optimization. Flag returns: 32; correct-pose returns: 18; strict historical/seed material-plus-pose returns: 0/0. These anchors assume rigid membership and do not certify ownership.

Graph-off nominal: 63 retained all-candidate keyframes; 420 primary reference births and 328 retirements. No already-confirmed primary landmark changes; 390 pending-to-confirmed coordinate updates. Pending-to-confirmed coordinate updates remain native promotion; they are not graph optimization. Flag returns: 50; correct-pose returns: 40; strict historical/seed material-plus-pose returns: 0/0. These anchors assume rigid membership and do not certify ownership.

## Fixed events and subsequent tails

Voxel counts refer to the primary TSDF; rebuild counters include every candidate.
Baseline historical counts were not recorded in these window traces.

### nominal at 45.6667 s

| System | Point / normal / conditional target, mm / degrees / mm | Accepted / lost | Active / historical | Published / independent material inliers | TSDF voxels / total rebuilds |
|---|---:|---|---:|---:|---:|
| baseline | 2.87 / 0.40 / 2.96 | False / True | 30 / not recorded | 3 / 1 | 7,352,721 / 0 |
| bounded | 178.61 / 29.81 / 116.23 | True / False | 117 / 180 | 10 / 6 | 21,086,856 / 10 |
| graph-off | 2.23 / 0.02 / 2.15 | True / False | 73 / 210 | 35 / 25 | 9,529,520 / 8 |

| System | Correct / inaccurate accepted poses from event through end | Lost rows from event | Longest whole-sequence inaccurate accepted run, frames | Terminal point / full rotation | Terminal lost interval |
|---|---:|---:|---:|---:|---|
| baseline | 763 / 86 | 1128 | 4 | 49.83 mm / 3.26 degrees | 266 frames; 74.2000–78.6167 s |
| bounded | 1305 / 98 | 568 | 4 | 22.06 mm / 3.14 degrees | 60 frames; 77.6333–78.6167 s |
| graph-off | 1125 / 58 | 794 | 6 | 115.94 mm / 7.95 degrees | 472 frames; 70.7667–78.6167 s |

### light at 52.1333 s

| System | Point / normal / conditional target, mm / degrees / mm | Accepted / lost | Active / historical | Published / independent material inliers | TSDF voxels / total rebuilds |
|---|---:|---|---:|---:|---:|
| baseline | 4.33 / 0.13 / 4.20 | True / False | 60 / not recorded | 11 / 10 | 11,734,020 / 3 |
| bounded | 12.37 / 1.14 / 13.11 | True / False | 86 / 360 | 13 / 4 | 12,558,924 / 9 |
| graph-off | 0.95 / 0.02 / 1.62 | True / False | 67 / 240 | 19 / 13 | 11,594,313 / 7 |

| System | Correct / inaccurate accepted poses from event through end | Lost rows from event | Longest whole-sequence inaccurate accepted run, frames | Terminal point / full rotation | Terminal lost interval |
|---|---:|---:|---:|---:|---|
| baseline | 1253 / 35 | 287 | 2 | 7.84 mm / 1.55 degrees | 8 frames; 78.5000–78.6167 s |
| bounded | 983 / 541 | 65 | 33 | 20.83 mm / 2.43 degrees | No terminal loss |
| graph-off | 1303 / 238 | 49 | 6 | 11.99 mm / 0.25 degrees | No terminal loss |

## All candidates

Each cell gives correct accepted / inaccurate accepted / native-lost rows. Candidate identities after renewal diverges are not cross-run material correspondences. Secondary rigid-leaf scores remain hypothetical; there is no truth-based reselection.

### light

| Candidate | Baseline | Bounded | Graph-off |
|---|---:|---:|---:|
| candidate-0 | 2276 / 35 / 531 | 2250 / 541 / 65 | 2569 / 238 / 49 |
| candidate-1 | 289 / 75 / 2486 | 426 / 897 / 1512 | 537 / 933 / 1384 |
| candidate-2 | 299 / 125 / 2433 | 364 / 720 / 1763 | 302 / 1070 / 1432 |
| candidate-3 | 2285 / 128 / 443 | 1872 / 160 / 825 | 1899 / 958 / 0 |
| candidate-4 | 2167 / 87 / 601 | 1740 / 1037 / 77 | 1520 / 1335 / 2 |

### nominal

| Candidate | Baseline | Bounded | Graph-off |
|---|---:|---:|---:|
| candidate-0 | 1494 / 88 / 1274 | 2183 / 99 / 568 | 2004 / 58 / 794 |
| candidate-1 | 1858 / 71 / 928 | 960 / 380 / 1487 | 977 / 366 / 1456 |
| candidate-2 | 287 / 43 / 2527 | 403 / 499 / 1955 | 373 / 735 / 1749 |
| candidate-3 | 1907 / 311 / 634 | 1660 / 604 / 593 | 1954 / 26 / 877 |
| candidate-4 | 0 / 0 / 2523 | 0 / 0 / 1873 | 0 / 0 / 1963 |
| candidate-6 | 0 / 0 / 2510 | 0 / 0 / 1931 | 0 / 0 / 1725 |

## Interpretation and decision

Keep graph-off as an experimental variant; do not adopt it as the common replacement. Primary light gains 319 correct available poses over bounded graph-on, reaching 89.89%, with fewer inaccurate acceptances (541→238) and losses (65→49). Its accepted point p95 remains 11.51 mm, and the final accepted point error is 11.99 mm. The longest inaccurate accepted run is six frames, versus 33 graph-on; repeated late errors remain. Primary nominal loses 179 correct available poses, reaching 70.12%; native losses increase 568→794 and terminal loss starts 6.87 s earlier, lasting 7.85 s. The retained final pose is 115.94 mm/7.95 degrees wrong and unavailable. Neither condition reaches 95% correct available poses.

The nominal spike is absent: at 45.6667 s, point/normal/conditional-target errors fall from 178.61 mm/29.81 degrees/116.23 mm to 2.23 mm/0.02 degrees/2.15 mm. In the three fixed neighboring frames, graph-off has no shared map changes or primary SDF integration; its primary grid remains 9,529,520 voxels and total rebuilds remain eight. Its supporting cohorts are chiefly born at 44.3833/44.6667 s, whereas bounded graph-on promotes different cohorts at the event. These are different full histories, not the same current correspondences with one optimizer call removed.

At light 52.1333 s, graph-off has 19 published inliers, 13 compatible with the independent historical anchors, versus 13/four graph-on. There is no same-frame graph, promotion, retirement or fusion in graph-off. Its nearby grid remains 11,594,313 voxels; supporting cohorts predate the event, chiefly 49.9833 s. The graph-on cohort born at 51.7 s and its landmark revisions do not recur on the same schedule. Later graph-off volume growth reaches 12,595,176 primary voxels; native fusion continues. The first inaccurate accepted graph-off light pose is at 59.15 s; removing the early deterioration does not remove the late tail.

Nominal's last primary birth is 56.2167 s and last retirement is 59.3667 s, leaving 92 active references. At 63.3667, 66.5333 and 70.75 s, renewal requests defer an unchanged 30-point batch because 92+30 exceeds 120. At the first loss (63.3833 s) and terminal-loss onset (70.7667 s), recomputed support falls to four pairs; native lost-state renewal remains suppressed. There is no same-frame retirement, promotion, map-coordinate change or fusion. The primary grid remains 11,547,500 voxels through the terminal frame. This exposes the existing capacity/support history; it does not prove that changing the budget, eviction or graph factor would fix it. All such controls remain unchanged.

No graph-off accepted candidate has mismatched inherited/published support or fewer than five published metric inliers. Nevertheless, inaccurate accepted poses remain and all 32 light/50 nominal flag returns fail the strict historical material-plus-pose audit. The secondary light audit also worsens: inaccurate acceptances rise from 3,355 to 4,534 across all candidates, despite the primary improvement. These totals describe hypothetical rigid-leaf scores, not aggregate robot availability or verified ownership. Candidate-0 remains fixed; no better secondary is selected.

Memory is not reduced overall: graph-off native-process peaks are 7.78/9.61 GiB versus 7.33/9.60 GiB graph-on. The active cap is verified in real CUDA storages, but retained keyframes, reference history and TSDF remain separate costs. Diagnostic request p95 is 711/894 ms; it is separate from offline accuracy and does not validate the unchanged 150 ms operational bound. One non-seed non-lost primary integration rejection per graph-off run is retained in addition to losses and inaccurate acceptances. Error/availability accounting never promotes a finite rejected/lost pose.

The next focused work is a saved-data diagnosis of the late nominal correspondence/cohort support and whole-batch capacity stall, together with the graph-on publication/landmark consistency at the original spike. Compare those mechanisms before another single controlled intervention. Do not relax gates or infer contact/A3/A4 readiness from this ablation.

## Evidence and limits

Saved outputs and experiment-specific copies of existing evaluation scripts are in `outputs/evidence/b1/perception/point2pose-bounded-no-graph-01/`: `controls.json`, `comparison.json`, `impact-summary.json`, `history.json`, per-candidate audits, and original full traces/resources/failures. Ignored payloads are not recoverable from Git. Exactly two selected RGB sheets show the fixed event, fixed later probe and original terminal frame for all three systems: `nominal-event-and-tail.png`, `light-event-and-tail.png`. `full-sequence-comparison.png` retains every finite pose and overlays accepted errors/losses across the complete original timeline. No better stage pose is substituted.

Graph-off disables global optimized landmark/keyframe-pose updates and their publication. Keyframe creation/promotion, native pending coordinate fusion, bounded replacement and TSDF integration remain enabled. Rebuilds still replay retained keyframes on observed extent growth, using their unoptimized poses. Size/bounds/rebuild counters and fusion logs describe that history; voxel values/surfaces are unavailable. Unchanged voxel count does not establish unchanged fused geometry. The experiment does not isolate a single missing factor, delayed publication, landmark Jacobian or replacement decision.

Single paired runs, one fixed camera, ideal depth, different seeds across lighting and a limited recorded angle do not establish hardware, occlusion/slip, population reliability, loaded contact, a closed A3 frame or actual A4 predictions. Strict material recovery and operational freshness remain independent. Offline compute latency includes diagnostics and concurrent CPU scoring; it never affects source-time accuracy or availability. The original 150 ms limit and 95%/1 cm/5 degree gates are unchanged. No training, campaign resume, provider adoption or push follows.
