info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

error_exit() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
  exit 1
}
