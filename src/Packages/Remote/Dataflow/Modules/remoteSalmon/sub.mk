# Makefile fragment for this subdirectory

include $(SRCTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Remote/Dataflow/Modules/remoteSalmon

SRCS     += \
	$(SRCDIR)/remoteSalmon.cc \
	$(SRCDIR)/HeightSimp.cc \
	$(SRCDIR)/RenderModel.cc \
	$(SRCDIR)/SimpMesh.cc\
	$(SRCDIR)/OpenGLServer.cc \
	$(SRCDIR)/socketServer.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Core/Datatypes Dataflow/Comm \
	Core/Persistent Core/Exceptions Core/Geometry \
	Core/Geom Core/Thread Core/Containers \
	Dataflow/GuiInterface Dataflow/TkExtensions Core/Util \
	Dataflow/TkExtensions Core/Datatypes \
	Dataflow/Modules/Salmon SCICore/OS Remote/Tools

LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/smallso_epilogue.mk

