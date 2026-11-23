#!/usr/bin/env bash
# Run paired windows serially on a single, exclusively held GPU.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
config="$(realpath "$1")"
output="$(realpath -m "$2")"
shift 2
if (( $# == 0 )); then
    set -- tracking-single tracking-batch block rgbd
fi
: "${BENCH_GPU_UUID:?Set BENCH_GPU_UUID to the RTX GPU UUID reported by nvidia-smi -L}"
if [[ -z "${BENCH_BACKENDS:-}" ]]; then
    BENCH_BACKENDS="$(python3 -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["backends"]))' "$config")"
fi
export BENCH_BACKENDS
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES="$BENCH_GPU_UUID" OMNICLIENT_HUB_MODE=disabled PYTHONUNBUFFERED=1
export launcher="$root/scripts/isaac-run.sh"
flock -n --close /tmp/codex-local-gpu0.lock flock -n --close /tmp/alexdoor-xas-gpu.lock bash -c '
    set -euo pipefail
    config="$1"; output="$2"; shift 2
    processes="$(nvidia-smi -i "$BENCH_GPU_UUID" --query-compute-apps=pid --format=csv,noheader)"
    test -z "$processes" || { printf "%s\n" "$processes" >&2; exit 2; }
    for case_name do
        for backend in $BENCH_BACKENDS; do
            case "$backend" in physx|steel) ;; *) printf "Invalid backend: %s\n" "$backend" >&2; exit 2 ;; esac
            run="$output/$case_name/$backend"
            mkdir -p "$(dirname "$run")"
            mkdir "$run"
            nvidia-smi -i "$BENCH_GPU_UUID" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits \
                --loop-ms=100 > "$run/vram.csv" &
            sample_pid=$!
            trap '\''kill "$sample_pid" 2>/dev/null || true'\'' EXIT
            status=0
            profile=isaacsim_physx
            if [[ "$backend" == steel ]]; then profile=newton_steel_native; fi
            ISAAC_PHYSICS_PROFILE="$profile" "$launcher" -m alexdoor_xas.benchmarks.run --config "$config" \
                --case "$case_name" --backend "$backend" --output "$run" > "$run/console.log" 2>&1 || status=$?
            kill "$sample_pid" 2>/dev/null || true
            wait "$sample_pid" 2>/dev/null || true
            trap - EXIT
            test -s "$run/result.json" || status=1
            printf "%s\n" "$status" > "$run/returncode.txt"
            printf "%s %s exit=%s\n" "$case_name" "$backend" "$status"
            (( status == 0 )) || exit "$status"
        done
    done
' bash "$config" "$output" "$@"
