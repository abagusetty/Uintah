#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Contact

SRCS     += $(SRCDIR)/NullContact.cc $(SRCDIR)/SingleVelContact.cc \
            $(SRCDIR)/FrictionContact.cc $(SRCDIR)/ContactFactory.cc \
	    $(SRCDIR)/Contact.cc $(SRCDIR)/RigidBodyContact.cc

#
# $Log$
# Revision 1.6  2001/01/11 03:31:31  guilkey
# Created new contact model for rigid bodies.
#
# Revision 1.5  2000/05/30 20:19:10  sparker
# Changed new to scinew to help track down memory leaks
# Changed region to patch
#
# Revision 1.4  2000/04/27 21:28:58  jas
# Contact is now created using a factory.
#
# Revision 1.3  2000/04/27 20:00:26  guilkey
# Finished implementing the SingleVelContact class.  Also created
# FrictionContact class which Scott will be filling in to perform
# frictional type contact.
#
# Revision 1.2  2000/03/21 02:14:48  dav
# updated SingleVel to SingleVelContact
#
# Revision 1.1  2000/03/17 09:29:36  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
