# Makefile fragment for this subdirectory

# rtrt
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/rtrt.cc

ifeq ($(findstring -n32, $(C_FLAGS)),-n32)
#ifneq ($(USE_SOUND),no)
   AUDIOFILE_LIBRARY := -L/home/sci/dav
   SOUNDDIR := Packages/rtrt/Sound
   SOUNDLIBS := -laudio $(AUDIOFILE_LIBRARY) -laudiofile
endif

PROGRAM := Packages/rtrt/StandAlone/rtrt
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	$(SOUNDDIR) \
	Packages/rtrt/visinfo \
	Core/Thread \
	Core/Persistent \
	Core/Geometry \
	Core/Exceptions

endif
LIBS := -L/usr/sci/local/lib64 -loogl $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBS) $(X11_LIBS) -lXi -lXmu $(FASTM_LIBRARY) -lm $(THREAD_LIBS) $(PERFEX_LIBRARY) $(SOUNDLIBS)

include $(SCIRUN_SCRIPTS)/program.mk

# multi_rtrt
SRCS := $(SRCDIR)/multi_rtrt.cc
PROGRAM := Packages/rtrt/StandAlone/mrtrt
include $(SCIRUN_SCRIPTS)/program.mk

#nrrd2brick
SRCS := $(SRCDIR)/nrrd2brick.cc
LIBS := $(FASTM_LIBRARY) -lnrrd -lbiff -lair -lm $(THREAD_LIBS) $(X11_LIBS) -lXi -lXmu 
PROGRAM := Packages/rtrt/StandAlone/nrrd2brick
include $(SCIRUN_SCRIPTS)/program.mk

# visinfo
SRCDIR := Packages/rtrt/visinfo

SRCS := $(SRCDIR)/findvis.c

PROGRAM := Packages/rtrt/StandAlone/findvis
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/visinfo

endif
LIBS := $(GL_LIBS)

include $(SCIRUN_SCRIPTS)/program.mk

# gl
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/gl.cc

PROGRAM := Packages/rtrt/StandAlone/gl
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	Core/Thread \
	Core/Exceptions

endif
LIBS := $(GL_LIBS) $(FASTM_LIBRARY) -lm $(THREAD_LIBS) $(PERFEX_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

# mkbc
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/mkbc.cc

PROGRAM := Packages/rtrt/StandAlone/mkbc
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	Core/Persistent \
	Core/Thread \
	Core/Exceptions

endif
LIBS := $(GL_LIBS) -lfastm -lm -lfetchop -lperfex

include $(SCIRUN_SCRIPTS)/program.mk

# test2
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/test2.cc

PROGRAM := Packages/rtrt/StandAlone/test2
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	Core/Thread \
	Core/Persistent \
	Core/Exceptions

endif
LIBS := $(GL_LIBS) $(FASTM_LIBRARY) -lm -lXmu $(THREAD_LIBS) $(PERFEX_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

# glthread
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/glthread.cc

PROGRAM := Packages/rtrt/StandAlone/glthread
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	Core/Thread \
	Core/Persistent \
	Core/Exceptions

endif
LIBS := $(GL_LIBS) $(FASTM_LIBRARY) -lm -lXmu $(THREAD_LIBS) $(PERFEX_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk


SUBDIRS := \
	$(SRCDIR)/utils \
	$(SRCDIR)/scenes \

include $(SCIRUN_SCRIPTS)/recurse.mk

# Convenience target:
.PHONY: rtrt
rtrt: prereqs Packages/rtrt/StandAlone/rtrt scenes
.PHONY: scenes
scenes: $(SCENES)
.PHONY: librtrt
librtrt: lib/libPackages_rtrt_Core.so
