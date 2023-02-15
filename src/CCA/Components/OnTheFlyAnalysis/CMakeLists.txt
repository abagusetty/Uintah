set ( OTF_SRC
        AnalysisModuleFactory.cc
        AnalysisModule.cc
        controlVolume.cc
        controlVolFluxes.cc
        FileInfoVar.cc
        lineExtract.cc
        meanTurbFluxes.cc
        MinMax.cc
        momentumAnalysis.cc
        planeAverage.cc
        planeExtract.cc
        spatialAvg.cc
        statistics.cc
	turbulentFluxes.cc
    )

# Collect list of library dependencies
set( LIB_DEPS
        CCA_Ports
        Core_Disclosure
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
        Core_Math
        Core_OS
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

if( ENABLE_ICE )
    list( APPEND OTF_SRC vorticity.cc )
    list( APPEND LIB_DEPS CCA_Components_ICE_Core CCA_Components_ICE_Materials )
endif()
if( ENABLE_MPM )
    list( APPEND OTF_SRC flatPlate_heatFlux.cc particleExtract.cc )
    list( APPEND LIB_DEPS CCA_Components_MPM_Core CCA_Components_MPM_Materials )
endif()
if( ENABLE_MPM_ICE )
    list( APPEND OTF_SRC 1stLawThermo.cc )
endif()

add_library( CCA_Components_OnTheFlyAnalysis ${OTF_SRC} )
message(STATUS "libdeps ${LIB_DEPS}")
target_link_libraries( CCA_Components_OnTheFlyAnalysis PUBLIC ${LIB_DEPS} )

target_include_directories( CCA_Components_OnTheFlyAnalysis
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )
