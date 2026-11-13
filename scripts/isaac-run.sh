#!/usr/bin/env bash
# Launch an AlexDoor Isaac entry point with the project's physics runtime.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_sim_dir="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
isaac_lab_dir="${ISAAC_LAB_DIR:-$HOME/IsaacLab}"
export PYTHONPATH="$root/src:$isaac_lab_dir/source/isaaclab:$isaac_lab_dir/source/isaaclab_physx:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
profile="${ISAAC_PHYSICS_PROFILE:-$(python3 -c 'from alexdoor_xas.simulation_profile import DEFAULT_PHYSICS_PROFILE; print(DEFAULT_PHYSICS_PROFILE)')}"
case "$profile" in
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
