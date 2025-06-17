#!/usr/bin/env bash

TEST_DIR="$(dirname "$(realpath "$0")")"
cd "$TEST_DIR" || exit 1

for file in test-*; do
  if [ -f "$file" ]; then
    echo "Running test $file..."
    ./"$file"
  fi
done
