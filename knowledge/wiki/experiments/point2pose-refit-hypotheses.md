# Point2Pose Refit Hypothesis Preservation

## Question and method

Does retaining a supported RANSAC seed when its own refit loses support improve the bounded
graph-off tracker?

Isolated candidate `96c0943` revalidates the seed at unchanged five-inlier/4 mm gates. One complete
light/nominal CUDA attempt processes all 2,858 original captures per condition with the original 31
s seed; no partial-batch combination or tolerance variant is tried.

## Result and decision

Light correct 10/15/20 mm counts regress from 2,569/2,776/2,805 to 2,183/2,547/2,564; native losses
rise 49→280, accepted position maximum 21.05→63.56 mm and longest correct absence 0.183→0.967 s.
Nominal improves to 2,215/2,285/2,308 correct poses and 546 losses versus 2,004/2,047/2,052 and 794,
but accepted 10 mm precision falls 97.19%→95.85%.

All 5,716 CUDA rows complete. Returned transforms match their recomputed residuals/support; a
mechanically consistent cluster may still be wrong. The original nominal tail contains both missed
supported hypotheses and later inadequate correspondence support. Jump refusals include correct and
incorrect candidates, so blanket removal is unjustified.

**Reject default/common adoption and combination with partial batches.** Preserve the nominal
benefit and light regression; do not widen thresholds or reduce required inliers. The implementation
is retired.

## Limits and evidence

Every positional correctness bound also requires 5-degree rotation; seed errors are excluded while
schedule denominators remain complete. Original motion ends at 63.7209 degrees, not 90.7. One
attempt per condition does not establish repeatability. Proposed current-support/material recovery
checks remain unimplemented and do not authorize another campaign.

Evidence: `outputs/evidence/b1/perception/point2pose-refit-hypotheses-01/`, especially
`comparison.json`, complete traces, memory/return audits and saved-tail enumeration. Git: `96c0943`
/`57ad483`, retirement `c01c3a1`. See [[point2pose-isolated-improvements]].
