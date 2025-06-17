#!/usr/bin/env bash

condorgmm_TEST_MODE=${condorgmm_TEST_MODE:-0}

if [[ $condorgmm_TEST_MODE -ne 1 ]]; then
  set -euo pipefail
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPTS_DIR/common.sh"

# Prompt user for scene and frame_rate
read -p "Enter scenes: " scene
read -p "Enter frame rate: " frame_rate

if [[ -z $scene ]]; then
  error_exit "Scene is required."
fi

if [[ -z $frame_rate ]]; then
  error_exit "Frame rate is required."
fi

timestamp=$(date +%s)
experiment="$USER-evaluate-results-${timestamp}"
output_file="${experiment}.log"

info "Running evaluate on scenes $scene with frame rate $frame_rate to $output_file"

nohup python src/condorgmm/end_to_end.py \
  --experiment "$experiment" \
  --scene="$scene" \
  --FRAME_RATE="$frame_rate" \
  --use_gt_pose >"$output_file" &

info "(Running in the background...)"
