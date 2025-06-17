#!/usr/bin/env bash

set -euo pipefail

info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

error_exit() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
  exit 1
}

files=($(ls "$USER-evaluate-results"* 2>/dev/null))

if [ ${#files[@]} -eq 0 ]; then
  error_exit "No result files found with prefix: $USER-evaluate-results"
fi

if [ ${#files[@]} -eq 1 ]; then
  selected_file="${files[0]}"
  echo "Automatically selected: $selected_file"
else
  echo "Please select a results file:"
  select file in "${files[@]}"; do
    if [ -n "$file" ]; then
      selected_file="$file"
      break
    else
      echo "Invalid selection. Please try again."
    fi
  done
fi

if [ -z "$selected_file" ]; then
  error_exit "Usage: $0 <file>"
fi

metrics_folder=$(grep "Per-tracking-run Metrics Folder Name:" "$selected_file" | cut -d':' -f2- | xargs)

if [ -z "$metrics_folder" ]; then
  error_exit "Per-tracking-run Metrics Folder Name not found."
fi

echo "$metrics_folder"

python src/condorgmm/scripts/aggregate_metrics.py "$metrics_folder" || error_exit "Failed to compare."
