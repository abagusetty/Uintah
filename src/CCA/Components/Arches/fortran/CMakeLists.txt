set( F_FILE_PREFIXES
  arrass
  cellg
  fixval
  fixval_trans
  prescoef_var
  pressrcpred
  pressrcpred_var
  uvelcoef
  uvelcoef_central
  uvelcoef_upwind
  uvelcoef_mixed
  uvelcoef_hybrid
  uvelsrc
  vvelcoef
  vvelcoef_central
  vvelcoef_upwind
  vvelcoef_mixed
  vvelcoef_hybrid
  vvelsrc
  wallbc
  wvelcoef
  wvelcoef_central
  wvelcoef_upwind
  wvelcoef_mixed
  wvelcoef_hybrid
  wvelsrc
    )

# This will generate a header with appropriate name-mangling macros for C-Fortran compatibility
FortranCInterface_HEADER( FortranNameMangle.h
    SYMBOL_NAMESPACE F_
    SYMBOLS ${F_FILE_PREFIXES}
    )

# These fortran functions don't have C++ linkage, so we don't need name mangling on them:
list( APPEND F_FILE_PREFIXES
    fixval
    fixval_trans
    )

foreach( name ${F_FILE_PREFIXES} )
  list( APPEND F_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${name}.F )
#  list( APPEND INC_FILES ${name}_fort.h )
endforeach()

# advertise these files upward to be included in the ARCHES_SRC list of files to compile
set( ARCHES_SRC ${ARCHES_SRC} ${F_FILES} PARENT_SCOPE )
