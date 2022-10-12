#!/bin/bash

module load rocm/5.2.0 gcc cmake craype-accel-amd-gfx90a PrgEnv-gnu
module unload cray-libsci

rm -rf build
mkdir build

cd build

../src/configure \
  --enable-64bit \
  --enable-optimize="-std=c++17 -g -O2" \
  --enable-assertion-level=0 \
  --enable-examples \
  --with-hip="/opt/rocm-5.2.0" \
  --with-mpi="/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1/" \
  CC=cc \
  CXX=CC \
  CXXFLAGS="-Wno-deprecated -Wno-unused-local-typedefs -I/opt/rocm-5.2.0/include/ -D__HIP_PLATFORM_AMD__" \
  LDFLAGS='-ldl' \
  F77=ftn

make -j32 sus compare_uda

cd StandAlone

cp sus ../../tmp/sus
cp compare_uda ../../tmp/compare_uda
