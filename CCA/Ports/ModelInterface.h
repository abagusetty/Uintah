#ifndef UINTAH_HOMEBREW_ModelInterface_H
#define UINTAH_HOMEBREW_ModelInterface_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>

namespace Uintah {
/**************************************

CLASS
   ModelInterface
   
   Short description...

GENERAL INFORMATION

   ModelInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Model of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Model_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class DataWarehouse;
  class Material;
  class ProcessorGroup;
  class VarLabel;
  class ModelSetup {
  public:
  };
  class ModelInfo {
  public:
    ModelInfo(const VarLabel* delt, const VarLabel* mass_source,
	      const VarLabel* momentum_source, const VarLabel* energy_source,
	      const VarLabel* density, const VarLabel* velocity,
	      const VarLabel* temperature, const VarLabel* pressure)
      : delT_Label(delt), mass_source_CCLabel(mass_source),
      momentum_source_CCLabel(momentum_source),
      energy_source_CCLabel(energy_source),
      density_CCLabel(density), velocity_CCLabel(velocity),
      temperature_CCLabel(temperature), pressure_CCLabel(pressure)
      {
      }
    const VarLabel* delT_Label;

    const VarLabel* mass_source_CCLabel;
    const VarLabel* momentum_source_CCLabel;
    const VarLabel* energy_source_CCLabel;

    const VarLabel* density_CCLabel;
    const VarLabel* velocity_CCLabel;
    const VarLabel* temperature_CCLabel;
    const VarLabel* pressure_CCLabel;
  private:
    ModelInfo(const ModelInfo&);
    ModelInfo& operator=(const ModelInfo&);
  };

   class ModelInterface : public UintahParallelPort {
   public:
     ModelInterface(const ProcessorGroup* d_myworld);
     virtual ~ModelInterface();
      
     //////////
     // Insert Documentation Here:
     virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			       ModelSetup& setup) = 0;
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleInitialize(const LevelP& level,
				     SchedulerP&) = 0;
     //////////
     // Insert Documentation Here:
     virtual void restartInitialize() {}
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleComputeStableTimestep(SchedulerP& sched,
						const LevelP& level,
						const ModelInfo*) = 0;
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleMassExchange(SchedulerP&,
				       const LevelP& level,
				       const ModelInfo*) = 0;
     virtual void scheduleMomentumAndEnergyExchange(SchedulerP&,
						    const LevelP& level,
						    const ModelInfo*) = 0;

   private:
     const ProcessorGroup* d_myworld;

     ModelInterface(const ModelInterface&);
     ModelInterface& operator=(const ModelInterface&);
   };
} // End namespace Uintah
   


#endif
