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
: "${ISAAC_SIM_DIR:?Set ISAAC_SIM_DIR to the Isaac Sim installation}"
: "${NEWTON_SITE:?Set NEWTON_SITE to the isolated Newton wheel and dependencies}"
: "${BENCH_GPU_UUID:?Set BENCH_GPU_UUID to the RTX GPU UUID reported by nvidia-smi -L}"
if [[ -z "${BENCH_BACKENDS:-}" ]]; then
    BENCH_BACKENDS="$(python3 -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["backends"]))' "$config")"
fi
export BENCH_BACKENDS
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES="$BENCH_GPU_UUID" OMNICLIENT_HUB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
NEWTON_SITE="$(realpath "$NEWTON_SITE")"
export NEWTON_SITE
BENCH_WHEEL_VIEW="$(mktemp -d /tmp/isaac-benchmark-wheels.XXXXXX)"
export BENCH_WHEEL_VIEW
trap 'rm -r -- "$BENCH_WHEEL_VIEW"' EXIT
for package in newton newton_usd_schemas warp; do
    test -d "$NEWTON_SITE/$package"
    ln -s -- "$NEWTON_SITE/$package" "$BENCH_WHEEL_VIEW/$package"
done
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
            runtime_path="$PYTHONPATH"
            if [[ "$backend" == steel ]]; then runtime_path="$BENCH_WHEEL_VIEW:$runtime_path"; fi
            PYTHONPATH="$runtime_path" "$ISAAC_SIM_DIR/python.sh" -m alexdoor_xas.benchmarks.run --config "$config" \
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
