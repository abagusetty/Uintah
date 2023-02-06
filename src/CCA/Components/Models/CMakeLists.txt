set( LIB_DEPS
        CCA_Ports
        CCA_Components_Schedulers
        Core_Disclosure
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_IO
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

add_subdirectory( FluidsBased )
add_subdirectory( HEChem )
add_subdirectory( MultiMatlExchange )
add_subdirectory( Radiation )
add_subdirectory( SolidReactionModel )

# abb: for now only Radiation/RMCRT is GPU enabled
if( ENABLE_CUDA )
    find_package(curand REQUIRED)
    include_directories( $ENV{MPI_HOME}/include )
endif( ENABLE_CUDA )

if( ENABLE_HIP )
    set_source_files_properties(${MDL_RADIATION_SRC} PROPERTIES LANGUAGE CXX)
    find_package(hiprand REQUIRED)
    include_directories( $ENV{MPI_HOME}/include )
endif( ENABLE_HIP )

message(STATUS "MKL_RADIATION_SRC :${MDL_RADIATION_SRC}")
add_library( CCA_Components_Models
               ModelFactory.cc
               ${MDL_FLUIDSBASED_SRC}
               ${MDL_HECHEM_SRC}
               ${MDL_MMEXCH_SRC}
               ${MDL_RADIATION_SRC}
               ${MKL_SRM_SRC} )

if( ENABLE_ICE )
    list( APPEND LIB_DEPS
            CCA_Components_ICE_Core
            CCA_Components_ICE_CustomBCs
            CCA_Components_ICE_Materials
        )
endif()
if( ENABLE_MPM_ICE )
    list( APPEND LIB_DEPS
            CCA_Components_MPMICE_Core
            CCA_Components_SwitchingCriteria
        )
endif()

target_link_libraries( CCA_Components_Models PRIVATE ${LIB_DEPS} )

if( ENABLE_CUDA )
    target_link_libraries( CCA_Components_Models PRIVATE CUDA::cudart CUDA::curand )
endif( ENABLE_CUDA )
if( ENABLE_HIP )
    target_compile_options( CCA_Components_Models PRIVATE -fgpu-rdc )
    target_link_options( CCA_Components_Models PRIVATE "$<$<LINK_LANGUAGE:CXX>:-fgpu-rdc;--hip-link>" )
    target_link_libraries( CCA_Components_Models PRIVATE hip::device hip::hiprand )
endif( ENABLE_HIP )

target_include_directories( CCA_Components_Models
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )
