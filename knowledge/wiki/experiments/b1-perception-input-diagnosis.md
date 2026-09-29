# B1 Perception — Input Dependence and Train Errors

## Scope and method

Read-only diagnosis of [[experiments/b1-perception-run-02|run-02]] at checkout
`68dc0ec`, using both existing checkpoints on the RTX 4090. No optimizer step,
model update, collection, sealed-test access or new training occurred. Source
and configuration remain unchanged. Evidence and reproducible diagnostic scripts
are in `outputs/b1/perception/run-02/input-diagnosis/`; checkpoint file hashes
and model tensors are verified unchanged.

The last checkpoint was replayed on all 25,006 train and 6,498 development causal
windows; per-door errors reproduce the preceding evaluation. Input experiments
use eight fixed progress positions in each of five phases of all 50 episodes:
2,000 windows per checkpoint, with manipulation results reported separately.
Two deterministic replacement-door assignments use train donors matched by
handedness, nominal/light condition, phase and phase progress. Individual RGB,
depth, proprioception or camera histories are replaced while other inputs stay
fixed. A second experiment exchanges early/late push histories within the same
episode: 304 train and 96 development windows per checkpoint.

These mixed observations are counterfactual sensitivity tests, not realistic
rollouts or proofs of what a separately trained modality-only model could do.
Equal phase sampling differs from full-trajectory gate weighting. Camera swaps
are uninformative because matched camera poses are identical; zero output change
does not establish camera-pose independence.

## Finding 1: complete height is not directly observed

Visual inspection shows a contact-oriented view that cuts off the upper door.
A geometric audit then uses each prepared panel's center, full width, height and
thickness, measured articulation, recorded optical camera pose and intrinsics.
For **all 31,654 cached frames**, the entire top face of the prepared panel bounds
is outside the viewing frustum. This is proved by a separating frustum half-space
containing none of its four corners, including the full panel thickness and
actual vertical offset; it is stronger than checking whether a single corner is
inside the image. See `check_panel_top.py` and `prepared-top-visibility.json`.

The observed stream supplies no direct measurement of that upper boundary.
Predicting full height within 1 cm on unseen door families consequently relies
on appearance/prior assumptions. This is a concrete observability limit, not
evidence that the current loss can recover the missing measurement with more
epochs. It also does not explain every contact error by itself.

The model retains only four frames, spanning 0.3 seconds. Adding an initial
inspection view would therefore also require retaining its static geometry
during manipulation; merely recording an earlier frame would not make it
available to the current sliding-window estimator.

## Finding 2: predictions depend mainly on RGB features

At the last checkpoint, replacing one modality with another door's matched
history changes train contact predictions as follows. Values are output-change
p95 across the fixed manipulation samples, with ranges across the two donor
assignments; these are **not accuracy or modality-importance percentages**.

| Replaced input | Contact prediction change p95 |
|---|---:|
| RGB features | 60.3–72.0 mm |
| Depth/XYZ and validity | 0.76–0.98 mm |
| Proprioception | 2.71–2.82 mm |

Changing RGB and depth together is close to changing RGB alone. Scaling metric
depth by 1% changes contact predictions by only 0.035 mm at train p95 and
0.067 mm at development p95. Similar RGB dominance is present in epoch 1.

The same-door temporal experiment controls for door identity and exercises much
larger motion differences. On train, the median true angle difference between
paired histories is 33.50 degrees. Replacing RGB alone changes predicted angle
by median 29.74 degrees; depth alone by 0.74 degrees; proprioception alone by
2.16 degrees. Thus a dominant proprioceptive trajectory shortcut is **not** the
leading explanation supported by these interventions. The model is weakly
sensitive to metric depth in the tested conditions. This does not establish that
RGB contains no useful geometry or that proprioception contributes nothing.

An input audit over 20 cached frames per episode finds no nonfinite RGB features,
geometry or proprioception. At three recorded manipulation instants per episode,
source/cache time, proprioception and camera pose agree; all 150 target contact
points project inside the original image. Being inside the image does not prove
visibility through the hand or panel occlusion. `recorded-views.png` provides
five-door visual evidence with diagnostic target overlays. These checks do not
repeat an exhaustive source/cache alignment audit.

## Finding 3: train failure is concentrated in composed contact position

On all 21,662 train manipulation windows, **every component except composed
contact position stays within its individual tolerance**. Contact position fails
on 22.71% of windows, leaving only 5/19 doors passing the per-door gates. Pooled
p95 is 12.44 mm; the per-door range remains 3.98–14.48 mm. This is substantially
more specific than saying that fourteen doors are generally unlearned.

Train pooled p95 errors are 4.11 mm for hinge origin, 6.83 mm for local contact
and 1.40 degrees for articulation. Their errors combine in the reconstructed
world contact. Its pooled signed mean residual is (-5.94, -1.94, +1.67) mm;
many failing doors show a persistent negative X offset. Push/hold p95 is about
12.5 mm versus 10.3 mm in contact. Paired nominal/light contact predictions differ
by only 0.80–2.31 mm at per-door p95, so these two lighting conditions do not
account for the dominant residual.

Privileged component substitutions isolate contributions. Replacing only local
contact with its target lowers pooled contact p95 to 6.58 mm and puts every
train door below 1 cm; replacing only hinge origin gives 9.70 mm pooled, and
only articulation gives 9.97 mm pooled. These use labels and are **not fixes or
deployable results**. Per-door residual demeaning also puts every door below
1 cm, supporting systematic bias rather than predominantly random failures;
no per-door correction was applied.

On 19 fixed nominal train batches, contact-position gradients and the sum of
other geometry gradients have positive cosine similarity throughout
(0.394–0.941, median 0.897). This local test does not support another blanket
loss reweighting as the immediate correction. It does not prove global
optimization quality or that learning-rate refinement will meet the gates.

Confidence provides little separation even on these train frames: its mean is
0.876 for accurate contacts and 0.864 for inaccurate contacts; empirical AUROC
is 0.570. All other train components pass, so contact accuracy also identifies
complete-state usability here. These are correlated recorded-frame diagnostics,
not an independent calibration validation. Development contact predictions are
all inaccurate and still have mean confidence 0.863.

## Fix priorities and acceptance

1. **Make required geometry observable.** Design a bounded inspection/view change
   that includes the upper boundary and relevant panel/frame extents. Preserve
   the manipulation setup and acceptance gates. Retain observed static geometry
   across the subsequent action; do not substitute asset metadata at inference.
2. **Give metric observations an explicit role.** Compare a small, separate
   depth/XYZ path using camera calibration with the current RGB-dominated fusion.
   Existing observations can support the first controlled comparison; full-height
   generalization still needs the visibility problem addressed. No backbone
   replacement or broad new collection has been selected by this diagnosis.
3. **Refine contact fitting on all train doors.** Test a short optimization
   refinement with a smaller learning rate and fixed per-door train geometry
   monitoring before changing loss weights. Keep all state checks, and prohibit
   asset-specific offsets or privileged component substitutions. This is a
   proposed experiment, not a proven remedy for the development failure.
4. **Verify confidence after geometric corrections.** Require measured acceptance
   of accurate states and rejection of inaccurate/unobservable states. The
   existing low-confidence missing-input examples do not establish this behavior
   for a visible but unfamiliar door. Do not lower the confidence threshold.
5. Only then run a separate bounded full experiment, checking original contact
   and complete-state development gates. Preserve all old attempts and keep the
   sealed test partition closed. Full-train fitting and held-out generalization
   remain separate acceptance results.

The diagnosis selects these priorities; it does not yet qualify an architecture,
camera trajectory, new dataset, confidence calibration or training recipe.
