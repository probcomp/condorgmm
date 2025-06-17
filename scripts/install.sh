#!/usr/bin/env bash

# Installs the condorgmm global and project environment.

condorgmm_TEST_MODE=${condorgmm_TEST_MODE:-0}

if [[ $condorgmm_TEST_MODE -ne 1 ]]; then
  set -euo pipefail
fi

info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

error_exit() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
  exit 1
}

PIXI_HOME="$HOME/.pixi"
PIXI_BIN="$PIXI_HOME/bin"
BASHRC="$HOME/.bashrc"
export PATH="$PIXI_BIN:$PATH"

# Deactivates the "base" conda environment
deactivate-conda() {
  if command -v conda >/dev/null 2>&1; then
    info "Conda detected. Deactivating base environment and disabling auto-activation."
    conda init >/dev/null 2>&1 || error_exit "Failed to initialize Conda."
    conda deactivate >/dev/null 2>&1 || info "No active Conda environment to deactivate."
    conda config --set auto_activate_base false >/dev/null 2>&1 || error_exit "Failed to disable auto-activation of the Conda base environment."
    info "Successfully deactivated Conda base environment and disabled auto-activation."
  else
    info "Conda not found. Skipping deactivation step."
  fi
}

# Install pixi
install-pixi() {
  info "Installing pixi..."
  curl -fsSL https://pixi.sh/install.sh | bash || error_exit "Failed to install pixi."
  if ! grep -q 'eval "$(pixi completion --shell bash)"' "$BASHRC"; then
    echo 'eval "$(pixi completion --shell bash)"' >>"$BASHRC"
  fi
}

# Update global pixi environment
update-pixi-global() {
  info "Updating pixi global environment..."
  pixi global update || error_exit "Failed to update pixi."
}

# Authenticate gcloud
authenticate-gcloud() {
  if gcloud auth list --filter=status:ACTIVE --format="get(account)" 2>/dev/null | grep -q .; then
    info "gcloud is already authenticated."
  else
    info "Authenticating gcloud..."
    gcloud auth login --update-adc --force || error_exit "Failed to authenticate gcloud."
  fi
}

# Configure git
configure-git() {
  pixi global install git || error_exit "Failed to install git."

  info "Checking git configuration..."
  if git config --global user.name &>/dev/null; then
    GIT_USER_NAME=$(git config --global user.name)
    info "git username already configured: $GIT_USER_NAME"
  else
    read -p "Enter your git username: " GIT_USER_NAME
    git config --global user.name "$GIT_USER_NAME" || error_exit "Failed to set git username."
    info "git username configured: $GIT_USER_NAME"
  fi

  if git config --global user.email &>/dev/null; then
    GIT_USER_EMAIL=$(git config --global user.email)
    info "git email already configured: $GIT_USER_EMAIL"
  else
    read -p "Enter your git email: " GIT_USER_EMAIL
    git config --global user.email "$GIT_USER_EMAIL" || error_exit "Failed to set git email."
    info "git email configured: $GIT_USER_EMAIL"
  fi
}

# Configure GitHub CLI
install-github-cli() {
  info "Installing GitHub CLI..."
  pixi global install gh || error_exit "Failed to install GitHub CLI."
}

authenticate-github() {
  if gh auth status &>/dev/null; then
    info "GitHub CLI is already authenticated."
  else
    info "GitHub CLI is not authenticated."
    gh auth login --web || error_exit "Failed to authenticate GitHub CLI."
    info "Successfully authenticated GitHub CLI."
  fi
}

# Install lazygit
install-lazygit() {
  info "Installing lazygit..."
  pixi global install lazygit || error_exit "Failed to install lazygit."
}

# Clone condorgmm repository
clone-condorgmm-repo() {
  info "Cloning condorgmm repository..."
  read -p "Enter the branch name activate [main]: " BRANCH_NAME
  BRANCH_NAME=${BRANCH_NAME:-main}
  echo "Activating branch: $BRANCH_NAME"
  gh repo clone probcomp/condorgmm || error_exit "Failed to clone condorgmm repository."
  pushd condorgmm || error_exit "Failed to enter condorgmm directory."
  git checkout "$BRANCH_NAME" || error_exit "Failed to checkout branch $BRANCH_NAME"
  popd
}

# Install pre-commit and setup hooks
install-pre-commit() {
  info "Installing pre-commit..."
  pixi global install pre-commit || error_exit "Failed to install pre-commit."
  info "Setting up pre-commit hooks..."
  pre-commit install || error_exit "Failed to install pre-commit hooks."
}

# Update project dependencies
update-dependencies() {
  info "Updating project dependencies..."
  pixi clean && pixi install || error_exit "Failed to clean project and install dependencies."
  pixi update || error_exit "Failed to update project dependencies."
}

upgrade-system-packages() {
  info "Updating system packages..."
  if ! sudo apt update -y; then
    exit_error "Failed to update package lists"
  fi
  if ! sudo apt upgrade -y; then
    exit_error "Failed to upgrade packages"
  fi
}

remove-mesa-glvnd() {
  info "Removing existing Mesa and GL packages..."
  sudo apt remove -y --purge 'mesa*' || exit_error "Failed to remove mesa packages"
  sudo apt remove -y libglvnd0 libglvnd-dev || exit_error "Failed to remove libglvnd packages"
}

prompt_reboot() {
  read -p "A reboot is required to complete the installation. Reboot now? (y/n): " response
  if [[ $response == "y" || $response == "Y" ]]; then
    echo "Rebooting the system..."
    sudo reboot
  else
    info "Reboot canceled!"
    info "condorgmm environments installed (but you still need to reboot)."
    info "Remember to: 'source ~/.bashrc'"
  fi
}

install() {
  local flag="$1"

  info "Installing condorgmm global environment..."
  touch "$BASHRC" || error_exit "Failed to create or access .bashrc."
  deactivate-conda
  install-pixi
  update-pixi-global
  authenticate-gcloud
  configure-git
  install-github-cli
  authenticate-github
  install-lazygit
  if [[ $flag == "clone" ]]; then
    clone-condorgmm-repo cd condorgmm
  fi
  install-pre-commit
  update-dependencies
  upgrade-system-packages
  remove-mesa-glvnd
  prompt_reboot
}

parse-and-execute() {
  if [[ $# -eq 0 ]]; then
    install ""
    exit 0
  fi

  case "$1" in
  --clone)
    install "clone"
    exit 0
    ;;
  *)
    error_exit "Unknown parameter $1. Only --clone supported."
    ;;
  esac

  install
  exit 0
}

if [[ $condorgmm_TEST_MODE -eq 1 ]]; then
  echo "Entering test mode..."
else
  parse-and-execute "$@"
fi
