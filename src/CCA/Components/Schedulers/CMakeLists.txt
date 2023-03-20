if( ENABLE_CUDA OR ENABLE_HIP )
    set( GPU_SRC
            GPUDataWarehouse.cu
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cpp
        )
endif()

if( ENABLE_SYCL )
    #add_compile_options(-lze_loader)
    set( GPU_SRC
            GPUDataWarehouse.cc
            GPUGridVariableGhosts.cc
            GPUGridVariableInfo.cc
        )
endif( ENABLE_SYCL )

if( ENABLE_CUDA )
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CUDA)
    include_directories( $ENV{MPI_HOME}/include )
endif( ENABLE_CUDA )

if( ENABLE_HIP )
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CXX)
    include_directories( $ENV{MPI_HOME}/include )
endif( ENABLE_HIP )


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
        Threads::Threads )

if( ENABLE_CUDA )
    target_link_libraries( CCA_Components_Schedulers PRIVATE CUDA::cudart )
endif( ENABLE_CUDA )
if( ENABLE_HIP )
    target_compile_options( CCA_Components_Schedulers PRIVATE -fgpu-rdc )
    target_link_libraries( CCA_Components_Schedulers PRIVATE hip::device )
    target_link_options( CCA_Components_Schedulers PRIVATE "$<$<LINK_LANGUAGE:CXX>:-fgpu-rdc;--hip-link>" )
endif( ENABLE_HIP )


target_include_directories( CCA_Components_Schedulers
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
