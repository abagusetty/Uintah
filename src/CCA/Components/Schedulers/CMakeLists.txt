if( ENABLE_CUDA )
    set( GPU_SRC
            GPUDataWarehouse.cu
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
        )
endif( ENABLE_CUDA )

if( ENABLE_HIP )
    set( GPU_SRC
            GPUDataWarehouse.cu
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
        )

    # target_sources( CCA_Components_Schedulers PRIVATE ${GPU_SRC} )
    # target_link_libraries( CCA_Components_Schedulers PUBLIC hip::host )
    set_source_files_properties(${GPU_SRC} PROPERTIES COMPILE_FLAGS "-D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=$ENV{ROCM_PATH} --offload-arch=gfx90a -x hip")
endif( ENABLE_HIP )

if( ENABLE_SYCL )
    set( GPU_SRC
            GPUDataWarehouse.cc
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
            SYCLScheduler.cc
        )
endif( ENABLE_SYCL )

add_library( CCA_Components_Schedulers
        DependencyBatch.cc
        DependencyException.cc
        DetailedDependency.cc
        DetailedTask.cc
        DetailedTasks.cc
        DynamicMPIScheduler.cc
        KokkosOpenMPScheduler.cc
        MPIScheduler.cc
        MemoryLog.cc
        OnDemandDataWarehouse.cc
        Relocate.cc
        RuntimeStats.cpp
        SchedulerCommon.cc
        SchedulerFactory.cc
        SendState.cc
        TaskGraph.cc
        UnifiedScheduler.cc
        ${GPU_SRC}
    )

# jcs sort out this "DLINK_FILES" business

target_link_libraries( CCA_Components_Schedulers PRIVATE
        CCA_Ports
        CCA_Components_ProblemSpecification
        Core_Containers
        Core_Exceptions
        Core_Geometry
        Core_Grid
        Core_Malloc
        Core_OS
        Core_Parallel
        Core_Util
        Threads::Threads
    )

if( ENABLE_CUDA )
    target_link_libraries( CCA_Components_Schedulers PRIVATE CUDA::cudart )
endif( ENABLE_CUDA )

if( ENABLE_HIP )
    target_link_libraries( CCA_Components_Schedulers PRIVATE hip::host )
endif( ENABLE_HIP )


target_include_directories( CCA_Components_Schedulers
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
