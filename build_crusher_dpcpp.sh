#!/bin/bash

module load cmake libxml2
module load openblas/0.3.17
module -t list

rm -rf build_sycl
mkdir build_sycl

cd build_sycl

cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_SYCL=ON -DENABLE_EXAMPLES=OFF -DCMAKE_CXX_FLAGS="-std=c++17 -O3 -sycl-std=2020 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a -I/opt/cray/pe/mpich/8.1.17/ofi/cray/10.0/include" ../src/

cmake --build . -j 32

cd StandAlone

cp sus ../../tmp/sus
cp compare_uda ../../tmp/compare_uda
