/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef Packages_Uintah_CCA_Components_Examples_RMCRT_Test_h
#define Packages_Uintah_CCA_Components_Examples_RMCRT_Test_h

#include <CCA/Components/Examples/uintahshare.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/DebugStream.h>

using SCIRun::DebugStream;

namespace Uintah
{
  class SimpleMaterial;
  class ExamplesLabel;
  class VarLabel;
  class GeometryObject;
/**************************************

CLASS
   RMCRT_Test
   
   RMCRT_Test simulation

GENERAL INFORMATION

   RMCRT_Test.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

  
   Copyright (C) 2011 SCI Group

KEYWORDS
   Regridder

DESCRIPTION
   Component for testing RMCRT concepts on multiple levels
  
WARNING
  
****************************************/

  class UINTAHSHARE RMCRT_Test: public UintahParallelComponent, public SimulationInterface {
  public:
    RMCRT_Test ( const ProcessorGroup* myworld );
    virtual ~RMCRT_Test ( void );

    // Interface inherited from Simulation Interface
    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP& state );
    virtual void scheduleInitialize            ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleTimeAdvance           ( const LevelP& level, SchedulerP& scheduler);
    virtual void scheduleErrorEstimate         ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleInitialErrorEstimate  ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleCoarsen               ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleRefine                ( const PatchSet* patches, SchedulerP& scheduler );
    virtual void scheduleRefineInterface       ( const LevelP& level, SchedulerP& scheduler, bool needCoarseOld, bool needCoarseNew);

  private:
    void initialize ( const ProcessorGroup*,
                      const PatchSubset* patches, 
                      const MaterialSubset* matls,
                       DataWarehouse* old_dw, 
                      DataWarehouse* new_dw );

    void computeStableTimestep ( const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw );
    
    void scheduleShootRays_onCoarseLevel(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls);
    
    void shootRays_onCoarseLevel ( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw );
                             
                             
    void scheduleShootRays_multiLevel(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);
    
    void shootRays_multiLevel( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

    void scheduleRefine_Q(SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls);

    void refine_Q(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse*,
                  DataWarehouse* new_dw);
                  
    void scheduleCoarsen_Q( const LevelP& level, 
                            SchedulerP& scheduler );
                  
    void coarsen_Q ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*, 
                     DataWarehouse* new_dw);
                                                    
    void schedulePseudoCFD(SchedulerP& sched,
                           const PatchSet* patches,
                           const MaterialSet* matls);

    void pseudoCFD ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);       

    void errorEstimate ( const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse*, 
                         DataWarehouse* new_dw, 
                         bool initial);
                         
    void printSchedule(const PatchSet* patches,
                       DebugStream& dbg,
                       const string& where);
  
    void printSchedule(const LevelP& level,
                       DebugStream& dbg,
                       const string& where);

    void printTask(const PatchSubset* patches,
                   const Patch* patch,
                   DebugStream& dbg,
                   const string& where);

    void printTask(const Patch* patch,
                   DebugStream& dbg,
                   const string& where);

   protected:
    const ProcessorGroup* d_myworld;
    


    ExamplesLabel*   d_examplesLabel;
    SimulationStateP d_sharedState;
    SimpleMaterial*  d_material;

    VarLabel* d_colorLabel;
    VarLabel* d_sumColorDiffLabel;

    SCIRun::Vector d_gridMax;
    SCIRun::Vector d_gridMin;
    
    Ghost::GhostType d_gn;
    Ghost::GhostType d_gac;
    

    // Fake cylinder
    SCIRun::Vector d_centerOfBall;
    SCIRun::Vector d_centerOfDomain;
    double         d_radiusOfBall;
    double         d_radiusOfOrbit;
    double         d_angularVelocity;
    int            d_matl;
    bool           d_CoarseLevelRMCRTMethod;
    bool           d_multiLevelRMCRTMethod;
    
    int d_orderOfInterpolation;         // Order of interpolation for interior fine patch
    std::vector<GeometryObject*> d_refine_geom_objs;
  };

} // namespace Uintah

#endif
