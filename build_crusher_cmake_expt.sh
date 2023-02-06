#!/bin/bash

module load cmake
module load rocm/5.4.0 cray-mpich/8.1.23 craype/2.7.19 libxml2/2.9.12
module load cce/15.0.0
module -t list

rm -rf build_hip
export MPI_HOME=${MPICH_DIR}
echo $MPI_HOME

cmake -H. -Bbuild_hip -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DENABLE_HIP=ON -DENABLE_EXAMPLES=ON ./src/
cd build_hip
make -j32

#cp sus ../../tmp/sus
#cp compare_uda ../../tmp/compare_uda
