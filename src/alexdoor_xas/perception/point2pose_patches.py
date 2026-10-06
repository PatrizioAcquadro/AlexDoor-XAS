"""Reproducible corrections to the pinned native source; no model or gate tuning."""

from pathlib import Path

NATIVE_FIXES = ("f2m_measured_loss", "sdf_returned_pose", "inlier_driven_renewal")

_REGISTER = "upstream/point2pose/modules/register/svd_cluster_ransac_register.py"
_FRONTEND = "upstream/point2pose/pipeline/components/front_end.py"
_CRITERION = "upstream/point2pose/modules/criterion/rotation_thres_and_min_num_criterion.py"
_REPLACEMENTS = {
    _REGISTER: (
        (
            '        selected_T = np.asarray(candidates[best_cluster_idx]["T"], '
            "dtype=np.float64)\n",
            '        selected_T = np.asarray(candidates[best_cluster_idx]["T"], '
            "dtype=np.float64)\n"
            "        cluster_T = selected_T.copy()\n",
        ),
        (
            '            candidates[best_cluster_idx]["T"] = refined_T\n',
            "            selected_T = refined_T\n"
            '            candidates[best_cluster_idx]["T"] = refined_T\n',
        ),
        (
            "            if selected_T is not None:\n"
            "                fallback_T = np.asarray(selected_T, dtype=np.float64)\n",
            "            if cluster_T is not None:\n"
            "                fallback_T = np.asarray(cluster_T, dtype=np.float64)\n",
        ),
    ),
    _FRONTEND: (
        (
            "                    if jump_rejected:\n"
            "                        obj.lost = True\n"
            "                    elif 0 < mean_res < self.reg_residual_thres:\n"
            "                        obj.lost = False\n",
            "                    obj.lost = bool(\n"
            "                        jump_rejected\n"
            "                        or not (0 <= mean_res < self.reg_residual_thres)\n"
            '                        or np.count_nonzero(stats_reg.get("inliers", []))\n'
            "                        < self.register._min_inliers\n"
            "                    )\n",
        ),
    ),
    _CRITERION: (
        (
            '        num_pts = reg_stats["correspond_curr3d"].shape[0]\n',
            '        num_pts = int(np.count_nonzero(reg_stats["inliers"]))\n',
        ),
        (
            '            f"[Criterion] num visible points: {num_pts} ',
            '            f"[Criterion] num final inliers: {num_pts} ',
        ),
    ),
}


def patch_tracking(root):
    """Validate every pinned edit before writing; preserve each pre-fix source."""
    pending = []
    for relative, replacements in _REPLACEMENTS.items():
        path = Path(root) / relative
        original = patched = path.read_text()
        for old, new in replacements:
            if patched.count(new) == 1:
                continue
            if patched.count(old) != 1:
                raise ValueError(f"Unexpected pinned Point2Pose source: {relative}")
            patched = patched.replace(old, new, 1)
        if patched != original:
            pending.append((path, original, patched))
    for path, original, patched in pending:
        backup = path.with_suffix(".py.before-tracking-fixes")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(patched)
    return NATIVE_FIXES
