if (HAVE_SYCL)
  target_sources( uintah PRIVATE GPUDataWarehouse.cc SYCLScheduler.cc )
endif (HAVE_SYCL)

if (HAVE_CUDA OR HAVE_HIP)
  target_sources( uintah PRIVATE GPUDataWarehouse.cu UnifiedScheduler.cc )
endif ()
