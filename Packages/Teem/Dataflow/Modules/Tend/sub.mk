#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Teem/Dataflow/Modules/Tend


SRCS     += \
	$(SRCDIR)/TendAnscale.cc\
	$(SRCDIR)/TendAnvol.cc\
	$(SRCDIR)/TendBmat.cc\
	$(SRCDIR)/TendEpireg.cc\
	$(SRCDIR)/TendEstim.cc\
	$(SRCDIR)/TendEval.cc\
	$(SRCDIR)/TendEvalClamp.cc\
	$(SRCDIR)/TendEvalPow.cc\
	$(SRCDIR)/TendEvec.cc\
	$(SRCDIR)/TendEvecRGB.cc\
	$(SRCDIR)/TendExpand.cc\
	$(SRCDIR)/TendMake.cc\
	$(SRCDIR)/TendNorm.cc\
	$(SRCDIR)/TendPoint.cc\
	$(SRCDIR)/TendSatin.cc\
	$(SRCDIR)/TendShrink.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/Teem/Core/Datatypes Core/Datatypes \
	Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/GeomInterface Core/Datatypes Core/Geometry \
        Core/TkExtensions Packages/Teem/Core/Datatypes \
	Packages/Teem/Dataflow/Ports

LIBS := $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
TEEM_MODULES := $(TEEM_MODULES) $(LIBNAME)
endif
