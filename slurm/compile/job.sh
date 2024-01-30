#!/bin/bash

PROJECT_DIR=$1
BUILD_DIR="$PROJECT_DIR/build_native"

lscpu

module load cmake
module load gcc/13.1.0
module load openmpi
module unload cuda

rm -r $BUILD_DIR || true
mkdir -p $BUILD_DIR

cd $BUILD_DIR
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_FLAGS="-march=native -mtune=native" -DCMAKE_C_FLAGS="-march=native -mtune=native" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build . --target matrix-solver -- -j 52