#ifndef Uintah_CombinedFactory_h
#define Uintah_CombinedFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class PropertyBase;

  class CombinedFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static PropertyBase* create(ProblemSpecP& ps);
  };

} // End namespace Uintah

#endif /* Uintah_CombinedFactory_h */

