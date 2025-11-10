#!/bin/bash

if [ -d "build" ]; then
    rm -r build
fi

mkdir build

PYTHON_ENV="$CONDA_PREFIX"
OUTPUT_DIR="../../basicsr/archs/vmamba_2d/v2dmamba_scan"

cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="$PYTHON_ENV" -DCUDA_ARCHS="89;75;80" -DOUTPUT_DIRECTORY="$OUTPUT_DIR" -B build

cmake --build build -- -j32