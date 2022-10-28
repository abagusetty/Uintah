#!/bin/bash

module load cmake PrgEnv-amd amd/5.3.0 rocm/5.3.0 boost
module -t list

rm -rf build
mkdir build

cd build

cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DENABLE_HIP=ON -DENABLE_EXAMPLES=ON -DCMAKE_CXX_FLAGS="-std=c++17 -I/opt/rocm-5.3.0/include" -DCMAKE_EXE_LINKER_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64" ../src/

cmake --build . -j 32

cd StandAlone

cp sus ../../tmp/sus
cp compare_uda ../../tmp/compare_uda
