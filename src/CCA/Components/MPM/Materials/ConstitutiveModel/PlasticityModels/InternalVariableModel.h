/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __INTERNAL_VARIABLE_MODEL_H__
#define __INTERNAL_VARIABLE_MODEL_H__

#include "PlasticityState.h"
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Math/Matrix3.h>
#include <vector>
#include <map>


namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  InternalVariableModel
    \brief  Abstract base class for the evolution of internal variables
            for plasticity models
  */
  ///////////////////////////////////////////////////////////////////////////

  class InternalVariableModel {

  private:

  public:
         
    InternalVariableModel();
    virtual ~InternalVariableModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Initial computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) {};

    // Computes and requires for internal evolution variables
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) {};

    virtual void allocateCMDataAddRequires(Task* task, 
                                           const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb){};

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   std::map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw){};

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to){};

    virtual void initializeInternalVariable(ParticleSubset* pset,
                                            DataWarehouse* new_dw){};

    virtual void getInternalVariable(ParticleSubset* pset,
                                     DataWarehouse* old_dw,
                                     constParticleVariableBase& intvar){};

    virtual void allocateAndPutInternalVariable(ParticleSubset* pset,
                                                DataWarehouse* new_dw,
                                                ParticleVariableBase& intvar){}; 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw,
                                     constParticleVariableBase& intvar){}; 

    ///////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the internal variable and return new value  */
    virtual double computeInternalVariable(const PlasticityState* state) const = 0;

    ///////////////////////////////////////////////////////////////////////////
    // Compute derivative of internal variable with respect to volumetric
    // elastic strain
    virtual double computeVolStrainDerivOfInternalVariable(const PlasticityState* state) const = 0;

  };
} // End namespace Uintah
      


#endif  // __INTERNAL_VARIABLE_MODEL_H__

