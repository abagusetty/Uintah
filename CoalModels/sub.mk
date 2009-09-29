# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/CoalModels

SRCS += \
  $(SRCDIR)/ModelFactory.cc \
  $(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/PartVel.cc \
  $(SRCDIR)/KobayashiSarofimDevol.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/DragModel.cc
