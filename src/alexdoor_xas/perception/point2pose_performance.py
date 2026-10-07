"""Explicit offline controls from the pinned official live implementation."""


def configure_performance(config, root, controls, *, diagnostic_only):
    """Change only named compute components; never import demo admission relaxations."""
    if not controls:
        return
    if not diagnostic_only:
        raise ValueError("performance_comparison_requires_diagnostic_mode")
    if set(controls) - {
        "query_chunk_size",
        "sam2_small",
        "tapir_crop",
        "resolution",
        "num_pips_iter",
        "simplified_svd",
    }:
        raise ValueError("unknown_performance_control")
    tracker = config.tracker.params
    if "query_chunk_size" in controls:
        chunk = controls["query_chunk_size"]
        if type(chunk) is not int or chunk < 0:
            raise ValueError("invalid_query_chunk_size")
        tracker.query_chunk_size = chunk
    if controls.get("sam2_small"):
        config.segmenter.params.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        config.segmenter.params.checkpoint = str(root / "checkpoints/sam2.1_hiera_small.pt")
    if controls.get("tapir_crop"):
        config.tracker.type = "tapir_crop"
        tracker.update(
            crop_size="auto",
            crop_pad_factor=1.5,
            crop_min_size=64,
            crop_recenter_thres_px=8,
            crop_center_mode="bbox",
            crop_debug=False,
        )
    if "resolution" in controls:
        resolution = controls["resolution"]
        if type(resolution) is not int or resolution not in (256, 384, 480):
            raise ValueError("invalid_tapir_resolution")
        tracker.resize_height = tracker.resize_width = resolution
    if "num_pips_iter" in controls:
        iterations = controls["num_pips_iter"]
        if type(iterations) is not int or iterations not in (1, 2, 3, 4):
            raise ValueError("invalid_tapir_iterations")
        tracker.num_pips_iter = iterations
    if controls.get("simplified_svd"):
        config.register.type = "svd_residual_outlier"
        # The official live YAML uses MAD (not an effective fixed 4 mm gate).
        # Keep the benchmark's frontend/sampling/jump/depth/residual controls.
        config.register.params.update(
            max_iter=5,
            threshold_method="fixed",
            inlier_thres=0.004,
            min_inliers=5,
            enable_sdf_refine=False,
        )
