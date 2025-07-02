#!/bin/bash

if [ -d "build" ]; then
    rm -r build
fi

mkdir build
#cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR=/opt/conda -DCUDA_ARCHS="70;75;80" -DBOUNDARY_CHECK=1 -DNAN_SMEM_CHECK=1 -DNAN_GRAD_CHECK=1 -B build
#cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="/home/jzhang/Dev/anaconda3_2023/envs/vmamba" -DCUDA_ARCHS="70;75;80" -DBOUNDARY_CHECK=1 -B build
cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="/home/heshiwen/anaconda3/envs/2dmamba" -DCUDA_ARCHS="89;75;80" -DOUTPUT_DIRECTORY=../../v2dmamba_scan -B build

#cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="/home/jzhang/Dev/anaconda3_2023/envs/vmamba" -DCUDA_ARCHS="70;75;80" -DOUTPUT_DIRECTORY=../v2dmamba_scan -B build

cmake --build build -- -j32