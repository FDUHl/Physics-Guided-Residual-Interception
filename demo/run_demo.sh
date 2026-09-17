#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash demo/run_demo.sh {v38|v47|v48|v49} [viewer|headless] [extra args...]"
  exit 2
fi

model="$1"
display_mode="${2:-viewer}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

case "$model" in
  v38)
    script="demo/demo_png_vy_vz_residual.py"
    checkpoint="checkpoints/v38_5d_mlp.pth"
    ;;
  v47)
    script="demo/demo_png_vy_lstm_pd_height.py"
    checkpoint="checkpoints/v47_5d_lstm.pth"
    ;;
  v48)
    script="demo/demo_png_vy_vz_residual.py"
    checkpoint="checkpoints/v48_3d_mlp.pth"
    ;;
  v49)
    script="demo/demo_png_vy_lstm_pd_height.py"
    checkpoint="checkpoints/v49_3d_lstm.pth"
    ;;
  *)
    echo "Unknown model: $model (expected v38, v47, v48, or v49)"
    exit 2
    ;;
esac

case "$display_mode" in
  viewer)
    display_arg="--viewer"
    ;;
  headless)
    display_arg="--headless"
    ;;
  *)
    echo "Unknown display mode: $display_mode (expected viewer or headless)"
    exit 2
    ;;
esac

mkdir -p demo_outputs

PYTHONUNBUFFERED=1 \
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/aerialgym_torch_extensions}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aerialgym_matplotlib}" \
python -u "$script" \
  --checkpoint "$checkpoint" \
  --mode residual \
  "$display_arg" \
  --use-warp \
  --device cuda:0 \
  --steps 300 \
  --print-every 50 \
  --deterministic \
  --robot-z 60 \
  --target-rear-bearing-deg 10 \
  --init-distance 30 \
  --init-target-pitch-deg 10 \
  --target-min-z 30 \
  --target-max-z 90 \
  --robot-max-velocity 12 \
  --forward-speed 12 \
  --max-vy 8 \
  --max-vz 3 \
  --max-yaw-rate 2.2 \
  --hold-kp 5.5 \
  --hold-kd 0.55 \
  --hold-max-vz 3 \
  --save-log "demo_outputs/${model}" \
  --save-plot "demo_outputs/${model}.png" \
  "$@"
