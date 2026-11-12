# Source before launching Kit to select matching Newton and Warp without replacing Kit USD.
: "${NEWTON_SITE:?Set NEWTON_SITE to the validated Newton/Warp/OpenUSD site}"
NEWTON_SITE="$(realpath "$NEWTON_SITE")"
export NEWTON_SITE
ISAAC_PHYSICS_WHEELS="$(mktemp -d /tmp/isaac-physics-wheels.XXXXXX)"
trap 'rm -r -- "$ISAAC_PHYSICS_WHEELS"' EXIT
for package in newton newton_usd_schemas warp; do
    test -d "$NEWTON_SITE/$package"
    ln -s -- "$NEWTON_SITE/$package" "$ISAAC_PHYSICS_WHEELS/$package"
done
export PYTHONPATH="$ISAAC_PHYSICS_WHEELS:${PYTHONPATH:-}"
