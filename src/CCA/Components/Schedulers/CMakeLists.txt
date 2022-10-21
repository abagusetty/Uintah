if( HAVE_CUDA )
    set( GPU_SRC
            GPUDataWarehouse.cu
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
            GPUMemoryPool.cc
        )
endif( HAVE_CUDA )

if( HAVE_HIP )
    set( GPU_SRC
            GPUDataWarehouse.cu
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
            GPUMemoryPool.cc
        )
endif( HAVE_HIP )

if( HAVE_SYCL )
    set( GPU_SRC
            GPUDataWarehouse.cc
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
            GPUMemoryPool.cc
            SYCLScheduler.cc
        )
endif( HAVE_SYCL )

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

target_link_libraries( CCA_Components_Schedulers
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

target_include_directories( CCA_Components_Schedulers
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
