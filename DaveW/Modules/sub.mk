#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/Modules

SUBDIRS := \
	$(SRCDIR)/CS684\
	$(SRCDIR)/EEG\
	$(SRCDIR)/EGI\
	$(SRCDIR)/FDM\
	$(SRCDIR)/FEM\
	$(SRCDIR)/ISL\
	$(SRCDIR)/MEG\
	$(SRCDIR)/Path\
	$(SRCDIR)/Readers\
	$(SRCDIR)/Tensor\
	$(SRCDIR)/Writers\
#[INSERT NEW SUBDIRS HERE]

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.3  2000/06/07 20:54:54  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:02  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:27  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
