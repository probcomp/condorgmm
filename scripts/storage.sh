#!/usr/bin/env bash

condorgmm_TEST_MODE=${condorgmm_TEST_MODE:-0}

if [[ $condorgmm_TEST_MODE -ne 1 ]]; then
  set -euo pipefail
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPTS_DIR/common.sh"

parse-and-execute() {
  if [[ $# -lt 1 ]]; then
    error_exit "No parameters passed. Use --mount or --symbolic-links."
  fi

  case "$1" in
  --mount)
    mount
    ;;
  --symbolic-links)
    symbolic-links
    ;;
  *)
    error_exit "Unknown parameter $1. Use --mount or --symbolic-links."
    ;;
  esac
}

create-directory() {
  local dir_path="$1"

  if [ -d "$dir_path" ]; then
    info "Directory $dir_path already exists."
    return 1
  else
    mkdir -p "$dir_path" || error_exit "Failed to create directory $dir_path."
    info "Created directory $dir_path."
    return 0
  fi
}

DISK_NAME="/dev/nvme0n2"
MOUNT_POINT="/mnt/disks/condorgmm-data"

is-mounted() {
  local mp="$1"
  if grep -qs "$mp" /proc/mounts; then
    return 0
  else
    return 1
  fi
}

mount-disk() {
  info "Listing attached disks..."
  lsblk || error_exit "Failed to list attached disks."

  if lsblk | grep -q "$(basename $DISK_NAME)"; then
    info "Disk $DISK_NAME found."
  else
    error_exit "Disk $DISK_NAME not found."
  fi

  info "Creating filesystem on $DISK_NAME..."
  sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "$DISK_NAME" || error_exit "Failed to create filesystem."

  info "Creating mount point at $MOUNT_POINT..."
  sudo mkdir -p "$MOUNT_POINT" || error_exit "Failed to create mount point."

  info "Mounting $DISK_NAME to $MOUNT_POINT..."
  sudo mount -o discard,defaults "$DISK_NAME" "$MOUNT_POINT" || error_exit "Failed to mount disk."

  info "Updating /etc/fstab..."
  if grep -q "$DISK_NAME" /etc/fstab; then
    info "Entry for $DISK_NAME already exists in /etc/fstab."
  else
    sudo bash -c "echo '$DISK_NAME $MOUNT_POINT ext4 discard,defaults,nofail 0 2' >> /etc/fstab" || error_exit "Failed to update /etc/fstab."
  fi

  info "Verifying the mount..."
  df -h | grep "$DISK_NAME" || error_exit "Disk not mounted successfully."

  info "Setting ownership of $MOUNT_POINT to user $USER..."
  sudo chown -R "$USER":"$USER" "$MOUNT_POINT" || error_exit "Failed to set ownership of $MOUNT_POINT."

  info "Disk mounted."
}

mount() {
  info "Mounting..."

  if is-mounted "$MOUNT_POINT"; then
    info "Disk $DISK_NAME is already mounted at $MOUNT_POINT. "
    exit 0
  fi

  mount-disk
  create-directory "/mnt/disks/condorgmm-data/bop"
  create-directory "/mnt/disks/condorgmm-data/bop/ycbv"
  info "Disk mounted and data directories created."
}

SYMLINK_PATH="assets/bop"
TARGET_DIR="/mnt/disks/condorgmm-data/bop"

symlink-exists() {
  local link_path="$1"

  if [ -L "$link_path" ]; then
    return 0
  else
    return 1
  fi
}

symbolic-links() {
  info "Setting up symbolic links..."
  if symlink-exists "$TARGET_DIR" "$SYMLINK_PATH"; then
    info "Not creating symlink because $SYMLINK_PATH exist."
    return 0
  else
    if ! ln -s "$TARGET_DIR" "$SYMLINK_PATH" >/dev/null 2>&1; then
      info "Symbolic link $SYMLINK_PATH already exists"
    else
      info "Created symbolic link $SYMLINK_PATH -> $TARGET_DIR."
    fi
  fi
}

if [[ $condorgmm_TEST_MODE -eq 1 ]]; then
  echo "Entering test mode..."
else
  parse-and-execute "$@"
fi
