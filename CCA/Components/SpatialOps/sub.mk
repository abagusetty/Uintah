# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/SpatialOps

SRCS     += $(SRCDIR)/SpatialOps.cc \
      $(SRCDIR)/Fields.cc \
    $(SRCDIR)/EqnFactory.cc \
    $(SRCDIR)/ExplicitTimeInt.cc \
    $(SRCDIR)/BoundaryCond.cc \
    $(SRCDIR)/EqnBase.cc \
    $(SRCDIR)/ScalarEqn.cc \
    $(SRCDIR)/SourceTermFactory.cc \
    $(SRCDIR)/SourceTermBase.cc \
    $(SRCDIR)/ModelFactory.cc \
    $(SRCDIR)/ModelBase.cc \
    $(SRCDIR)/ConstSrcTerm.cc \
    $(SRCDIR)/SpatialOpsMaterials.cc

# SUBDIRS := $(SRCDIR)/fortran 
# include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Core/ProblemSpec   \
        Core/GeometryPiece \
        Core/Grid          \
        Core/Util          \
        Core/Disclosure    \
        Core/Exceptions    \
        CCA/Components/OnTheFlyAnalysis \
        CCA/Ports     \
        Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

