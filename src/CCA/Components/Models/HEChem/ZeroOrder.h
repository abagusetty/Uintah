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


#ifndef Packages_Uintah_CCA_Components_Models_ZeroOrder_h
#define Packages_Uintah_CCA_Components_Models_ZeroOrder_h

#include <CCA/Components/Models/HEChem/HEChemModel.h>

#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;
/**************************************

CLASS
   ZeroOrder
  

GENERAL INFORMATION

   ZeroOrder.h

   Joseph Peterson
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   zero order 

DESCRIPTION
   A zero order pressure dependent rate model based on the rate equation:
   dF/dt = G*P^b
 
WARNING

****************************************/

  class ZeroOrder : public HEChemModel {
  public:
    ZeroOrder(const ProcessorGroup* myworld,
              const MaterialManagerP& materialManager,
              const ProblemSpecP& params,
              const ProblemSpecP& prob_spec);

    virtual ~ZeroOrder();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);
      
    virtual void scheduleInitialize(SchedulerP&,
                                    const LevelP& level);

    virtual void scheduleRestartInitialize(SchedulerP&,
                                           const LevelP& level){};
      
    virtual void scheduleComputeStableTimeStep(SchedulerP&,
                                               const LevelP& level);
      
    virtual void scheduleComputeModelSources(SchedulerP&,
                                             const LevelP& level);        
                                             
  private:    
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse*,
                             DataWarehouse* new_dw);

    ZeroOrder(const ZeroOrder&);
    ZeroOrder& operator=(const ZeroOrder&);

    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* delFLabel;   // diagnostic labels
    const VarLabel* totalMassBurnedLabel;
    const VarLabel* totalHeatReleasedLabel;

    ProblemSpecP d_params;
    ProblemSpecP d_prob_spec;
    const Material* matl0;
    const Material* matl1;

    ICELabel* Ilb;
    MaterialSet* mymatls;
    
    // flags for the conservation test
    struct saveConservedVars{
      bool onOff;
      bool mass;
      bool energy;
    };
    saveConservedVars* d_saveConservedVars;

    std::string fromMaterial, toMaterial;
    double d_G;
    double d_b;
    double d_E0;
    double d_rho0;
    double d_threshold_pressure;
    double d_threshold_volFrac;
  };
}

#endif
