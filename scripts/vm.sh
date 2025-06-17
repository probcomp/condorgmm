#!/usr/bin/env bash

condorgmm_TEST_MODE=${condorgmm_TEST_MODE:-0}

if [[ $condorgmm_TEST_MODE -ne 1 ]]; then
  set -euo pipefail
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPTS_DIR/common.sh"

gcp-auth() {
  gcloud auth login \
    --project="$PROJECT" \
    --update-adc --force ||
    error_exit "Failed to authenticate gcloud."
}

create-user-vm() {
  INSTANCE="condorgmm-standard-$USER"
  info "Boostrapping new $INSTANCE..."

  read -p "Enter your GCP project: " PROJECT

  gcp-auth

  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" \
    --zone="us-west1-a" \
    --image-family="common-cu123-ubuntu-2204-py310" \
    --image-project="deeplearning-platform-release" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=400GB \
    --boot-disk-type=pd-standard \
    --machine-type=g2-standard-8 \
    --accelerator="type=nvidia-l4,count=1" \
    --metadata="install-nvidia-driver=True" \
    --create-disk="name=$INSTANCE-data,size=2048GB,type=pd-balanced,auto-delete=no" ||
    error_exit "Failed to create GCP instance."

  gcloud compute config-ssh --project "$PROJECT" || error_exit "Failed to configure SSH."

  info "Your VM $INSTANCE.us-west1-a.$PROJECT is ready"
}

create-runner-vm() {
  INSTANCE="condorgmm-github-runner"
  info "Creating new github runner VM $INSTANCE..."

  read -p "Enter the GCP project: " PROJECT

  gcp-auth

  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" \
    --zone="us-west1-a" \
    --image-family="common-cu123-ubuntu-2204-py310" \
    --image-project="deeplearning-platform-release" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=400GB \
    --boot-disk-type=pd-standard \
    --machine-type=g2-standard-8 \
    --accelerator="type=nvidia-l4,count=1" \
    --metadata="install-nvidia-driver=True" \
    --create-disk="name=$INSTANCE-data,size=2048GB,type=pd-balanced,auto-delete=no" ||
    error_exit "Failed to create GCP instance."

  gcloud compute config-ssh --project "$PROJECT" || error_exit "Failed to configure SSH."

  info "Runner VM $INSTANCE is ready."
}

parse-and-execute() {
  if [[ -z ${1-} ]]; then
    error_exit "No command provided. Use --create-user-vm or --create-runner-vm."
    return 1
  fi
  case "$1" in
  --create-user-vm)
    create-user-vm
    ;;
  --create-runner-vm)
    create-runner-vm
    ;;
  *)
    error_exit "Unknown command. Use --create-user-vm or --create-runner-vm."
    ;;
  esac
}

if [[ $condorgmm_TEST_MODE -eq 1 ]]; then
  echo "Entering test mode..."
else
  parse-and-execute "$@"
fi
