#!/usr/bin/env bash
set -euo pipefail

export SRC=https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/

wget --recursive -nd --no-parent $SRC -P assets/ycbineoat

for file in assets/ycbineoat/*.tar.gz; do tar -xzf "$file" -C assets/ycbineoat; done
