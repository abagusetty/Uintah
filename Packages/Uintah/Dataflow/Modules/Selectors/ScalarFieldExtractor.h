/****************************************
CLASS
    ScalarFieldExtractor

    

OVERVIEW TEXT
    This module receives a DataArchive object.  The user
    interface is dynamically created based information provided by the
    DataArchive.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June, 2000

    Copyright (C) 2000 SCI Group

LOG
    Created June 27, 2000
****************************************/
#ifndef SCALARFIELDEXTRACTOR_H
#define SCALARFIELDEXTRACTOR_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Dataflow/Modules/Selectors/FieldExtractor.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h> 
#include <string>
#include <vector>


namespace Uintah {
using namespace SCIRun;

class ScalarFieldExtractor : public FieldExtractor { 
  
public: 

  // GROUP: Constructors
  //////////
  ScalarFieldExtractor(const string& id); 

  // GROUP: Destructors
  //////////
  virtual ~ScalarFieldExtractor(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  virtual void get_vars(vector< string >&,
			vector< const TypeDescription *>&);
  
private:

  GuiString tcl_status;

  GuiString sVar;
  GuiInt sMatNum;

  const TypeDescription *type;

  ArchiveIPort *in;
  FieldOPort *sfout;
  
  std::string positionName;

}; //class 

} // End namespace Uintah



#endif
