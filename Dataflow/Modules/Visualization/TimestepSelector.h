/****************************************
CLASS
    TimestepSelector


OVERVIEW TEXT
    This module receives a DataArchive and selects the visualized timestep.
    Or Animates the data.



KEYWORDS
    

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 2000

    Copyright (C) 1999 SCI Group

LOG
    Created June 26, 2000
****************************************/
#ifndef TIMESTEPSELECTOR_H
#define TIMESTEPSELECTOR_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/ArchivePort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/TclInterface/TCLvar.h> 
#include <string>
#include <vector>


namespace Uintah {
using namespace Uintah::Datatypes;
using namespace SCIRun;

class TimestepSelector : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  TimestepSelector(const clString& id); 

  // GROUP: Destructors
  //////////
  virtual ~TimestepSelector(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  
private:

  TCLstring tcl_status;

  TCLint animate;

  TCLint time;
  TCLdouble timeval;

  ArchiveIPort *in;
  ArchiveOPort *out;
  
  ArchiveHandle archiveH;
  void setVars(ArchiveHandle ar);

}; //class 
} // End namespace Uintah



#endif
