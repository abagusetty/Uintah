#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>


namespace Uintah {

/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   EquationOfState.h

   Biswajit Banerjee
   Department of Mechanical Enegineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 McMurtry Container Dynamics Group

KEYWORDS
   Damage_Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class EquationOfState {
      public:
	 
	 EquationOfState();
	 virtual ~EquationOfState();
	 
	 //////////
	 // Calculate the pressure using a equation of state
	 virtual double computePressure(const MPMMaterial* matl,
				        const double& bulk,
				        const double& shear,
                                        const Matrix3& deformGrad,
                                        const Matrix3& rateOfDeformation,
                                        const Matrix3& stress,
                                        const double& temperature,
                                        const double& density,
                                        const double& delT) = 0;

      };
} // End namespace Uintah
      


#endif  // __EQUATION_OF_STATE_H__

