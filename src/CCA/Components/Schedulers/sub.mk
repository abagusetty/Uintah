#
#  The MIT License
#
#  Copyright (c) 1997-2020 The University of Utah
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
#
#
# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := CCA/Components/Schedulers

SRCS += \
        $(SRCDIR)/DependencyBatch.cc          \
        $(SRCDIR)/DependencyException.cc      \
        $(SRCDIR)/DetailedDependency.cc       \
        $(SRCDIR)/DetailedTask.cc             \
        $(SRCDIR)/DetailedTasks.cc            \
        $(SRCDIR)/DynamicMPIScheduler.cc      \
        $(SRCDIR)/KokkosScheduler.cc          \
        $(SRCDIR)/KokkosOpenMPScheduler.cc    \
        $(SRCDIR)/MemoryLog.cc                \
        $(SRCDIR)/MPIScheduler.cc             \
        $(SRCDIR)/OnDemandDataWarehouse.cc    \
        $(SRCDIR)/Relocate.cc                 \
        $(SRCDIR)/RuntimeStats.cc             \
        $(SRCDIR)/SchedulerCommon.cc          \
        $(SRCDIR)/SchedulerFactory.cc         \
        $(SRCDIR)/SendState.cc                \
        $(SRCDIR)/TaskGraph.cc                \
        $(SRCDIR)/UnifiedScheduler.cc

ifeq ($(HAVE_CUDA),yes)
  SRCS += $(SRCDIR)/GPUDataWarehouse.cu       \
          $(SRCDIR)/GPUGridVariableInfo.cc    \
          $(SRCDIR)/GPUGridVariableGhosts.cc  \
          $(SRCDIR)/GPUMemoryPool.cc

  DLINK_FILES += CCA/Components/Schedulers/GPUDataWarehouse.o
endif

ifeq ($(HAVE_SYCL),yes)
  SRCS += $(SRCDIR)/GPUDataWarehouse.cpp      \
          $(SRCDIR)/GPUGridVariableInfo.cc    \
          $(SRCDIR)/GPUGridVariableGhosts.cc  \
          $(SRCDIR)/GPUMemoryPool.cc

  DLINK_FILES += CCA/Components/Schedulers/GPUDataWarehouse.o
endif

PSELIBS := \
        CCA/Components/ProblemSpecification \
        CCA/Ports        \
        Core/Containers  \
        Core/Disclosure  \
        Core/Exceptions  \
        Core/Geometry    \
        Core/Grid        \
        Core/Math        \
        Core/OS          \
        Core/Parallel    \
        Core/ProblemSpec \
        Core/Util

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(CUDA_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
