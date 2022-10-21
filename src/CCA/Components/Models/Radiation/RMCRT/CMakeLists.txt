set( SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/Radiometer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RayGPU.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/RMCRTCommon.cc
    )

# jcs sort out kokkos stuff
if( ENABLE_KOKKOS )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayKokkos.cc )
else()
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/Ray.cc )
endif()

# jcs need to sort out cuda stuff
if( ENABLE_CUDA )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayGPUKernel.cu )
endif()

if( ENABLE_HIP )
    list( APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/RayGPUKernel.cu )
    #set_source_files_properties(RayGPUKernel.cu PROPERTIES COMPILE_FLAGS -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --offload-arch=gfx90a -x hip)
endif()

set( MDL_SRC ${MDL_SRC} ${SRC} PARENT_SCOPE )
