#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Kurt/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/VolVis.tcl $(SRCDIR)/PadField.tcl
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Kurt/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.3  2000/03/21 17:33:25  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:37  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:29  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
