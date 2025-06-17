#!/usr/bin/env bash

set -euo pipefail

export SRC=https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main

wget $SRC/ycbv_base.zip -P assets/bop     # Base archive with dataset info, camera parameters, etc.
wget $SRC/ycbv_models.zip -P assets/bop   # 3D object models.
wget $SRC/ycbv_test_all.zip -P assets/bop # All test images ("_bop19" for a subset used in the BOP Challenge 2019/2020).

unzip -qq assets/bop/ycbv_base.zip -d assets/bop
unzip -qq assets/bop/ycbv_models.zip -d assets/bop/ycbv
unzip -qq assets/bop/ycbv_test_all.zip -d assets/bop/ycbv
