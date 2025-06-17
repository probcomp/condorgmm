#!/usr/bin/env bash

TEST_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$(realpath "$TEST_DIR/../../scripts")"
SOURCE_SCRIPT="$SCRIPT_DIR/storage.sh"

TEMP_MOUNT_POINT="/mnt/temp"
TEMP_DIR="/tmp/test-dir"
LINK_PATH="/tmp/test-link"

suite() {
  suite_addTest test-is-mounted
  suite_addTest test-create-directory
  suite_addTest test-symlink-exists
}

oneTimeSetUp() {
  export condorgmm_TEST_MODE=1
  source "$SOURCE_SCRIPT"
}

oneTimeTearDown() {
  export condorgmm_TEST_MODE=0
}

setUp() {
  if [ -d "$TEMP_DIR" ]; then
    rmdir "$TEMP_DIR"
  fi
  rm -f "$LINK_PATH"
}

test-is-mounted() {
  grep() {
    echo "$TEMP_MOUNT_POINT"
    return 0
  }
  is-mounted "$TEMP_MOUNT_POINT"
  $_ASSERT_TRUE_ $?

  grep() { return 1; }
  is-mounted "$TEMP_MOUNT_POINT"
  $_ASSERT_FALSE_ $?
}

test-create-directory() {
  # directory exists, should return false
  mkdir -p "$TEMP_DIR"
  create-directory "$TEMP_DIR"
  $_ASSERT_FALSE_ $?

  # directory doesn't exist, should return true
  [ -d "$TEMP_DIR" ] && rmdir "$TEMP_DIR"
  create-directory "$TEMP_DIR"
  $_ASSERT_TRUE_ $?
}

test-symlink-exists() {
  ln -s "/some/target" "$LINK_PATH"
  symlink-exists "$LINK_PATH"
  $_ASSERT_TRUE_ $?

  symlink-exists "/nope"
  $_ASSERT_FALSE_ $?
}

source "$TEST_DIR/shunit2"
