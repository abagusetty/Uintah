#include "BurnFactory.h"
#include "NullBurn.h"
#include "SimpleBurn.h"
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

using namespace Uintah;

Burn* BurnFactory::create(ProblemSpecP& ps)
{
    ProblemSpecP child = ps->findBlock("burn");
    if(!child)
      throw ProblemSetupException("Cannot find burn_model tag");
    std::string burn_type;
    if(!child->getAttribute("type",burn_type))
      throw ProblemSetupException("No type for burn_model"); 
    
    if (burn_type == "null")
      return(scinew NullBurn(child));
    
    else if (burn_type == "simple")
      return(scinew SimpleBurn(child));
    
    else 
      throw ProblemSetupException("Unknown Burn Type R ("+burn_type+")");
}

