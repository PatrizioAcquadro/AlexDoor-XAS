#!/usr/bin/env bash
# Install the native AlexDoor B1 Python runtime from the pinned Isaac Lab lock.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="$root/outputs/generated/isaac-native-runtime"
lab="$runtime/IsaacLab"
upstream=https://github.com/isaac-sim/IsaacLab.git
revision=67fe12806afbdb8bf8e125866661dd201088a54d
upstream_lock=70b4f037be69b362c08d9e5499a95edc7eded7e323ad0915fc314cfc9ba3c631
corrected_lock=9de0c8935a0b0b581cfbd417dad064400941cd0b582a5c5e0ec704046d9fa087

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

[[ $# == 0 ]] || fail "Usage: scripts/install-isaac-native.sh"
python="$(command -v python3.12)" || fail "Install Python 3.12 first."
uv="$(command -v uv || true)"
uv="${uv:-$runtime/tools/uv}"
[[ -x "$uv" ]] || fail "Install uv and add it to PATH first."

mkdir -p "$runtime/tmp" "$runtime/cache/uv"
# Serialize installers that share the same checkout and virtual environment.
exec 9>"$runtime/install.lock"
flock -n 9 || fail "Another installer is using $runtime"

if [[ ! -e "$lab" ]]; then
    staging="$(mktemp -d "$runtime/IsaacLab.XXXXXX")"
    trap 'rm -rf -- "$staging"' EXIT
    git init -q "$staging"
    git -C "$staging" remote add origin "$upstream"
    git -C "$staging" fetch --depth=1 origin "$revision"
    # AlexDoor uses its own assets. Isaac Lab's LFS media isn't needed for B1.
    GIT_LFS_SKIP_SMUDGE=1 git -C "$staging" checkout -q --detach "$revision"
    mv -T -- "$staging" "$lab"
    trap - EXIT
fi

cd "$lab"
[[ "$(git rev-parse HEAD)" == "$revision" ]] || fail "Unexpected Isaac Lab revision in $lab"
git diff --quiet HEAD -- . ':!uv.lock' || fail "Isaac Lab has tracked changes outside uv.lock"
if [[ "$(sha256sum < uv.lock)" == "$upstream_lock  -" ]]; then
    git apply --unidiff-zero "$root/configs/isaac-native-numpy.patch"
fi
[[ "$(sha256sum < uv.lock)" == "$corrected_lock  -" ]] || fail "Unexpected Isaac Lab lockfile"

export UV_PROJECT_ENVIRONMENT="$runtime/.venv"
export UV_CACHE_DIR="$runtime/cache/uv"
export UV_PYTHON_PREFERENCE=only-system
export TMPDIR="$runtime/tmp"
if [[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" "$uv" sync --locked --active --extra ovrtx
else
    "$uv" sync --locked --python "$python" --extra ovrtx
fi
"$uv" pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
