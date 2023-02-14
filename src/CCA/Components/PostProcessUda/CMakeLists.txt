add_library( CCA_Components_PostProcessUda
        PostProcessUda.cc
        Module.cc
        ModuleFactory.cc
    )

target_link_libraries( CCA_Components_PostProcessUda
    PUBLIC
        CCA_Components_Application
        CCA_Components_OnTheFlyAnalysis
        CCA_Components_ProblemSpecification
        CCA_Ports
        Core_Containers
        Core_DataArchive
        Core_Disclosure
        Core_Exceptions
        Core_Grid
        Core_OS
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Components_PostProcessUda
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
