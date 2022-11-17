#!/bin/bash

module unload craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm
module unload cray-libsci
module load cmake
module load openblas/0.3.17
module -t list

rm -rf build
mkdir build

cd build

cmake -DCMAKE_C_COMPILER=/gpfs/alpine/gen243/proj-shared/holmenjk/crusher/llvm/build_5.2.0/bin/clang -DCMAKE_CXX_COMPILER=/gpfs/alpine/gen243/proj-shared/holmenjk/crusher/llvm/build_5.2.0/bin/clang++ -DENABLE_SYCL=ON -DENABLE_EXAMPLES=ON -DCMAKE_CXX_FLAGS="-std=c++17 -O3 -sycl-std=2020 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a -I/opt/cray/pe/mpich/8.1.17/ofi/cray/10.0/include" ../src/

cmake --build . -j 32

cd StandAlone

cp sus ../../tmp/sus
cp compare_uda ../../tmp/compare_uda
