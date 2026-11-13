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
        *) profile="$(PYTHONPATH="$root/src:${PYTHONPATH:-}" python3 -c 'from alexdoor_xas.simulation_profile import DEFAULT_PHYSICS_PROFILE; print(DEFAULT_PHYSICS_PROFILE)')" ;;
    esac
fi
if [[ "$profile" != newton_steel_warp ]]; then
    export PYTHONPATH="$root/src:$isaac_lab_dir/source/isaaclab:$isaac_lab_dir/source/isaaclab_physx:${PYTHONPATH:-}"
fi
case "$profile" in
    newton_steel_warp)
        native_python="${ALEXDOOR_NATIVE_PYTHON:-$root/outputs/generated/isaac-native-runtime/.venv/bin/python}"
        if [[ ! -x "$native_python" ]]; then
            printf 'Set ALEXDOOR_NATIVE_PYTHON to the Isaac Lab native runtime Python (see the Purdue B1 wiki).\n' >&2
            exit 2
        fi
        export ISAAC_PHYSICS_PROFILE="$profile"
        export ALEXDOOR_USD_CACHE="${ALEXDOOR_USD_CACHE:-$root/outputs/generated/isaac-native-runtime/usd-cache}"
        export PYTHONPATH="$root/src:${ALEXDOOR_ALEX_ROOT:-$root/../Alex}/src:${PYTHONPATH:-}"
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
