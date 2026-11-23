"""Operational Purdue configuration; common door placement belongs to Phase 4.1."""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass


@configclass
class DoorPushPurdueEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 3600.0
    action_space = 6
    observation_space = 18
    state_space = 0
    sim = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=2,
    )
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    num_rerenders_on_reset = 2
    reset_camera_frames: int = 8
    native_steel: bool = False
    native_renderer_cfg: object | None = None
    diagnostic_hinge_stop_deg: float | None = None
    observed_load_feedback: bool = False
    measure_surface_components: bool = False
    recover_force_only_gap: bool = False
    pose_tool_frame_reference: bool = False
    pose_held_surface: bool = False
    action_mode: str = "A2"
    cameras: bool = True
    urdf_cache_dir: str | None = None
    camera_mount_pitch_rad: float = 0.0
    synthetic_door: object | None = None
    prepared_door: object | None = None
    door_mass_scale: float = 1.0
    door_hinge_damping: float | None = None
    floor_pose: tuple = (0.0, 0.0, 0.0)
    max_joint_speed: float = 0.5
    # Optional pose-command reserve; hard position/admission limits stay separate.
    pose_target_margin_rad: tuple | None = None
    pose_target_domain_rad: tuple | None = None
    # Optional soft nullspace preference; physical and admission bounds stay separate.
    pose_centering_domain_rad: tuple | None = None
    # Optional coupled preference in measured support; actual admission stays separate.
    pose_posture_support_rad: tuple | None = None
    # Existing stopping/acceleration reserve, excluding the measured velocity term.
    pose_posture_stopping_reserve_rad: tuple | None = None
    # Optional outer pose feedback; inner actuator impedance remains unchanged.
    pose_feedback_time_constant_s: float | None = None
    pose_reference_feedforward: bool = False
    pose_normal_compliance: bool = False
    # Optional geometric transport during explicit normal-reference commands.
    pose_normal_transport: bool = False
    # Finite position impedance on the other five axes in normal-reference mode.
    pose_finite_tangent_impedance: bool = False
    # Legacy stop response; a separately validated recipe can supply new gains.
    # Canonical right-arm order, same damping ratios and physical effort limits.
    stop_stiffness: tuple = (320.0, 320.0, 160.0, 160.0, 20.0, 10.0, 50.0)
    stop_diagnostic_continuation_s: float = 0.0
    centering_gain: float = 0.1
    ik_damping: float = 0.05
    max_pos_delta_m: float = 0.01
    max_rot_delta_rad: float = 0.05
    # Safe commissioning pose; Phase 4.1 selects the benchmark ready/parked setup.
    initial_joints: dict = {
        "LEFT_SHOULDER_X": 0.35,
        "LEFT_ELBOW_Y": -1.0,
        "RIGHT_SHOULDER_X": -0.35,
        "RIGHT_ELBOW_Y": -1.0,
    }
