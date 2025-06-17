#!/usr/bin/env bash

condorgmm_TEST_MODE=${condorgmm_TEST_MODE:-0}

if [[ $condorgmm_TEST_MODE -ne 1 ]]; then
  set -euo pipefail
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPTS_DIR/common.sh"

TRACE_DIR="$PIXI_PROJECT_ROOT/tensorboard"

trace() {
  tensorboard \
    --logdir \
    $TRACE_DIR
  info "(Running tensorboard in the background with logdir $TRACE_DIR...)"
}

clean() {
  rm -rf $TRACE_DIR/*
}

parse-and-execute() {
  case "$1" in
  --trace)
    trace
    ;;
  --clean)
    clean
    ;;
  *)
    error_exit "Unknown command. Use --trace or --clean."
    ;;
  esac
}

if [[ $condorgmm_TEST_MODE -eq 1 ]]; then
  echo "Entering test mode..."
else
  parse-and-execute "$@"
fi
