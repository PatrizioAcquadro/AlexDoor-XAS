#!/usr/bin/env bash
# Launch an AlexDoor Isaac entry point with the project's physics runtime.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_sim_dir="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
isaac_lab_dir="${ISAAC_LAB_DIR:-$HOME/IsaacLab}"
export PYTHONUNBUFFERED=1
profile="${ISAAC_PHYSICS_PROFILE:-}"
if [[ -z "$profile" ]]; then
    entry="${1:-}"
    case "${entry##*/}" in
        collect_perception.py|qualify_door.py) profile=newton_steel_warp ;;
        pilot_physics.py|pilot_workspace_survey.py|verify_pilot_dynamics.py|verify_loaded_stop.py) profile=isaacsim_physx ;;
        *) profile="$(PYTHONPATH="$root/src:${PYTHONPATH:-}" python3 -c 'from alexdoor_xas.simulation_profile import DEFAULT_PHYSICS_PROFILE; print(DEFAULT_PHYSICS_PROFILE)')" ;;
    esac
fi
export ISAAC_PHYSICS_PROFILE="$profile"
case "$profile" in
    newton_steel|isaacsim_physx)
        export PYTHONPATH="$root/src:$isaac_lab_dir/source/isaaclab:$isaac_lab_dir/source/isaaclab_physx:${PYTHONPATH:-}"
        ;;
esac
case "$profile" in
    newton_steel_native|newton_steel_warp)
        native_runtime="$root/outputs/generated/isaac-native-runtime"
        native_python="${ALEXDOOR_NATIVE_PYTHON:-$native_runtime/.venv/bin/python}"
        if [[ ! -x "$native_python" ]]; then
            printf 'Install the B1 runtime with scripts/install-isaac-native.sh or set ALEXDOOR_NATIVE_PYTHON.\n' >&2
            exit 2
        fi
        export ALEXDOOR_USD_CACHE="${ALEXDOOR_USD_CACHE:-$native_runtime/usd-cache}"
        # Keep runtime-generated files beside the installation on the same disk.
        export TMPDIR="${TMPDIR:-$native_runtime/tmp}"
        export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$native_runtime/cache}"
        export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/cuda}"
        export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$XDG_CACHE_HOME/warp}"
        mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$WARP_CACHE_PATH"
        export PYTHONPATH="$root/src:${ALEXDOOR_ALEX_ROOT:-$root/../Alex}/src:$native_runtime/opencv:${PYTHONPATH:-}"
        exec "$native_python" "$@"
        ;;
    newton_steel)
        export NEWTON_SITE="${NEWTON_SITE:-$root/outputs/generated/isaac-steel-runtime/site}"
        export PYTHONPATH="$isaac_lab_dir/source/isaaclab_newton:$PYTHONPATH"
        source "$root/src/isaac_physics_profile/launch.sh"
        "$isaac_sim_dir/python.sh" "$@"
        ;;
    isaacsim_physx)
        exec "$isaac_sim_dir/python.sh" "$@"
        ;;
    *)
        printf 'Unknown AlexDoor physics profile: %s\n' "$profile" >&2
        exit 2
        ;;
esac
