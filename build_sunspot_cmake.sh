#!/bin/bash

module -t list

rm -rf build_sycl
#export MPI_HOME=${MPICH_DIR}
#echo $MPI_HOME

cmake -H. -Bbuild_sycl -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DENABLE_SYCL=ON -DENABLE_EXAMPLES=OFF ./src/
cd build_sycl
make -j32

#cp sus ../../tmp/sus
#cp compare_uda ../../tmp/compare_uda
