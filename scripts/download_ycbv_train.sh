#!/usr/bin/env bash

condorgmm_TEST_MODE=${condorgmm_TEST_MODE:-0}

if [[ $condorgmm_TEST_MODE -ne 1 ]]; then
  set -euo pipefail
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPTS_DIR/common.sh"

SRC=https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main
PREFIX="assets/bop"

ZIP_FILE="ycbv_train_real.zip"
Z01_FILE="ycbv_train_real.z01"
ALL_ZIP_FILE="ycbv_train_real_all.zip"

ZIP_FILE_PATH="$PREFIX/$ZIP_FILE"
Z01_FILE_PATH="$PREFIX/$Z01_FILE"
ALL_ZIP_FILE_PATH="$PREFIX/$ALL_ZIP_FILE"

ZIP_URL="$SRC/$ZIP_FILE"
Z01_URL="$SRC/$Z01_FILE"

download() {
  info "Downloading $ZIP_URL"
  wget -q --show-progress --continue --timeout=60 "$ZIP_URL" -P "$PREFIX"

  info "Downloading $Z01_URL"
  wget -q --show-progress --continue --timeout=60 "$Z01_URL" -P "$PREFIX"
}

combine() {
  info "Combining $ZIP_FILE_PATH into $ALL_ZIP_FILE_PATH"
  zip -s0 "$ZIP_FILE_PATH" --out "$ALL_ZIP_FILE_PATH" >/dev/null 2>&1
}

check-integrity() {
  info "Checking zip file integrity..."
  if unzip -tqq "$ALL_ZIP_FILE_PATH" >/dev/null 2>&1; then
    info "Integrity check passed."
    return 0
  else
    error_exit "Integrity check failed for $ALL_ZIP_FILE_PATH."
    return 1
  fi
}

extract-combined() {
  info "Extracting $ALL_ZIP_FILE_PATH"
  unzip -qqo "$ALL_ZIP_FILE_PATH" -d "$PREFIX/ycbv" >/dev/null 2>&1
}

main() {
  info "Downloading files into $PREFIX"
  download
  combine
  if check-integrity; then
    extract-combined
  else
    error_exit "Integrity check failed. Aborting extraction."
  fi
  info "YCBV training data downloaded and verified in $PREFIX."
}

if [[ $condorgmm_TEST_MODE -eq 1 ]]; then
  echo "Entering test mode..."
else
  main "$@"
fi
