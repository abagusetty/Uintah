# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

SRCDIR  := StandAlone/tools/dumpfields
PROGRAM := StandAlone/tools/dumpfields/dumpfields

SRCS    := \
	$(SRCDIR)/dumpfields.cc \
	\
	$(SRCDIR)/utils.cc \
	$(SRCDIR)/Args.cc \
	$(SRCDIR)/FieldSelection.cc \
	\
	$(SRCDIR)/FieldDiags.cc \
	$(SRCDIR)/ScalarDiags.cc \
	$(SRCDIR)/VectorDiags.cc \
	$(SRCDIR)/TensorDiags.cc \
	\
	$(SRCDIR)/FieldDumper.cc \
	$(SRCDIR)/TextDumper.cc \
	$(SRCDIR)/EnsightDumper.cc \
	$(SRCDIR)/InfoDumper.cc \
	$(SRCDIR)/HistogramDumper.cc 

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := \
    Core_DataArchive                    \
    Core_Grid                           \
    Core_ProblemSpec                    \
    Core_GeometryPiece                  \
    CCA_Components_ProblemSpecification \
    CCA_Ports                           \
    Core_Parallel                       \
    Core_Math                           \
    Core_Disclosure                     \
    Core_Util                           \
    Core_Thread                         \
    Core_Persistent                     \
    Core_Exceptions                     \
    Core_Containers                     \
    Core_Malloc                         \
    Core_IO                             \
    Core_OS                             

else # Non-static build

  ifeq ($(LARGESOS),yes)
    PSELIBS := Packages/Uintah
  else
    PSELIBS := \
        Core/Exceptions    \
        Core/Grid          \
        Core/Util          \
        Core/Math          \
        Core/Parallel      \
        Core/Disclosure    \
        Core/ProblemSpec   \
        Core/Disclosure    \
        Core/DataArchive   \
	CCA/Ports          \
        CCA/Components/ProblemSpecification \
        Core/Exceptions  \
        Core/Persistent  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
  endif
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := \
        $(TEEM_LIBRARY) \
        $(XML2_LIBRARY) \
        $(Z_LIBRARY) \
        $(THREAD_LIBRARY) \
        $(F_LIBRARY) \
        $(PETSC_LIBRARY) \
        $(HYPRE_LIBRARY) \
        $(BLAS_LIBRARY) \
        $(LAPACK_LIBRARY) \
        $(MPI_LIBRARY) \
        $(X_LIBRARY) \
        $(M_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(F_LIBRARY) \
          $(TEEM_LIBRARY) $(PNG_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

