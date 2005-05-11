//----- RadiationDriver.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_DORadiationModel.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationDriver.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_RadiationModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
RadiationDriver::RadiationDriver(const ProcessorGroup* myworld,
				 ProblemSpecP& params):
  ModelInterface (myworld), params(params)
{
  d_perproc_patches = 0;
  d_DORadiation = 0;
  d_radCounter = -1; //to decide how often radiation calc is done
  d_radCalcFreq = 0; 

  d_cellInfoLabel = VarLabel::create("radCellInformation",
				     PerPatch<Models_CellInformationP>::getTypeDescription());
  shgamma_CCLabel = VarLabel::create("shgamma", CCVariable<double>::getTypeDescription());
  abskg_CCLabel = VarLabel::create("abskg", CCVariable<double>::getTypeDescription());
  esrcg_CCLabel = VarLabel::create("esrcg", CCVariable<double>::getTypeDescription());

  qfluxE_CCLabel = VarLabel::create("qfluxE", CCVariable<double>::getTypeDescription());
  qfluxW_CCLabel = VarLabel::create("qfluxW", CCVariable<double>::getTypeDescription());
  qfluxN_CCLabel = VarLabel::create("qfluxN", CCVariable<double>::getTypeDescription());
  qfluxS_CCLabel = VarLabel::create("qfluxS", CCVariable<double>::getTypeDescription());
  qfluxT_CCLabel = VarLabel::create("qfluxT", CCVariable<double>::getTypeDescription());
  qfluxB_CCLabel = VarLabel::create("qfluxB", CCVariable<double>::getTypeDescription());

  co2_CCLabel = VarLabel::create("CO2", CCVariable<double>::getTypeDescription());
  h2o_CCLabel = VarLabel::create("H2O", CCVariable<double>::getTypeDescription());
  mixfrac_CCLabel = VarLabel::create("scalar-f", CCVariable<double>::getTypeDescription());
  density_CCLabel = VarLabel::create("density", CCVariable<double>::getTypeDescription());
  temp_CCLabel = VarLabel::create("Temp", CCVariable<double>::getTypeDescription());
  tempCopy_CCLabel = VarLabel::create("TempCopy", CCVariable<double>::getTypeDescription());
  sootVF_CCLabel = VarLabel::create("sootVF", CCVariable<double>::getTypeDescription());
  sootVFCopy_CCLabel = VarLabel::create("sootVFCopy", CCVariable<double>::getTypeDescription());

  radiationSrc_CCLabel = VarLabel::create("radiationSrc", CCVariable<double>::getTypeDescription());
}

//****************************************************************************
// Destructor
//****************************************************************************
RadiationDriver::~RadiationDriver()
{
  delete d_DORadiation;
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  VarLabel::destroy(d_cellInfoLabel);

  VarLabel::destroy(shgamma_CCLabel);  
  VarLabel::destroy(abskg_CCLabel);  
  VarLabel::destroy(esrcg_CCLabel);  

  VarLabel::destroy(qfluxE_CCLabel);  
  VarLabel::destroy(qfluxW_CCLabel);  
  VarLabel::destroy(qfluxN_CCLabel);  
  VarLabel::destroy(qfluxS_CCLabel);  
  VarLabel::destroy(qfluxT_CCLabel);  
  VarLabel::destroy(qfluxB_CCLabel);

  VarLabel::destroy(co2_CCLabel);
  VarLabel::destroy(h2o_CCLabel);
  VarLabel::destroy(mixfrac_CCLabel);
  VarLabel::destroy(density_CCLabel);
  VarLabel::destroy(temp_CCLabel);
  VarLabel::destroy(tempCopy_CCLabel);
  VarLabel::destroy(sootVF_CCLabel);
  VarLabel::destroy(sootVFCopy_CCLabel);

  VarLabel::destroy(radiationSrc_CCLabel);
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
RadiationDriver::problemSetup(GridP& grid,
			      SimulationStateP& sharedState,
			      ModelSetup* setup)
{
  d_sharedState = sharedState;

  ProblemSpecP db = params->findBlock("RadiationModel");
  db->getWithDefault("radiationCalcFreq",d_radCalcFreq,5);
  d_DORadiation = scinew Models_DORadiationModel(myworld);
  d_DORadiation->problemSetup(db);
}

//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void RadiationDriver::scheduleInitialize(SchedulerP& sched,
					 const LevelP& level,
					 const ModelInfo*)
					 
{
  Task* t = scinew Task("RadiationDriver::initialize", this, 
                        &RadiationDriver::initialize);

  const PatchSet* patches= level->eachPatch();
  const MaterialSet* matls = d_sharedState->allICEMaterials();

  t->computes(qfluxE_CCLabel);
  t->computes(qfluxW_CCLabel);
  t->computes(qfluxN_CCLabel);
  t->computes(qfluxS_CCLabel);
  t->computes(qfluxT_CCLabel);
  t->computes(qfluxB_CCLabel);
  t->computes(radiationSrc_CCLabel);
  
  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actually initialize variables at first time step
//****************************************************************************

void
RadiationDriver::initialize(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse*,
			    DataWarehouse* new_dw)
{
  for (int p=0; p<patches->size();p++){
    const Patch* patch = patches->get(p);

    RadiationVariables vars;

    int iceIndex = 0;
    int matlIndex = d_sharedState->getICEMaterial(iceIndex)->getDWIndex();

    new_dw->allocateAndPut(vars.qfluxe, qfluxE_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vars.qfluxw, qfluxW_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vars.qfluxn, qfluxN_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vars.qfluxs, qfluxS_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vars.qfluxt, qfluxT_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vars.qfluxb, qfluxB_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vars.src, radiationSrc_CCLabel, matlIndex, patch);

    vars.qfluxe.initialize(0.0);
    vars.qfluxw.initialize(0.0);
    vars.qfluxn.initialize(0.0);
    vars.qfluxs.initialize(0.0);
    vars.qfluxt.initialize(0.0);
    vars.qfluxb.initialize(0.0);
    vars.src.initialize(0.0);
    
  }
}

//****************************************************************************
// dummy
//****************************************************************************
void RadiationDriver::scheduleComputeStableTimestep(SchedulerP&,
						    const LevelP&,
						    const ModelInfo*)
{
  // not applicable
}

//****************************************************************************
// schedule the computation of radiation fluxes and the source term
//****************************************************************************

void
RadiationDriver::scheduleComputeModelSources(SchedulerP& sched,
					     const LevelP& level,
					     const ModelInfo* mi)
{

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();

  string taskname = "RadiationDriver::buildLinearMatrix";

  Task* t = scinew Task("RadiationDriver::buildLinearMatrix", this, 
                        &RadiationDriver::buildLinearMatrix);

  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level);
  d_perproc_patches->addReference();

  sched->addTask(t, d_perproc_patches, ice_matls);

  scheduleCopyValues(level, sched, patches, ice_matls);
  scheduleComputeProps(level, sched, patches, ice_matls);    
  scheduleBoundaryCondition(level, sched, patches, ice_matls);    
  scheduleIntensitySolve(level, sched, patches, ice_matls, mi);

}

//****************************************************************************
// Initialize linear solver matrix for petsc/hypre
//****************************************************************************

void
RadiationDriver::buildLinearMatrix(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*,
				   DataWarehouse*)
{
  d_DORadiation->d_linearSolver->matrixCreate(d_perproc_patches, patches);
}

//****************************************************************************
// schedule copy of values from previous time step; if the radiation is to
// be updated using the radcounter, then we perform radiation calculations,
// else we just use the values from previous time step
//****************************************************************************

void
RadiationDriver::scheduleCopyValues(const LevelP& level,
				    SchedulerP& sched,
				    const PatchSet* patches,
				    const MaterialSet* matls)
{
  Task* t = scinew Task("RadiationDriver::copyValues",
		      this, &RadiationDriver::copyValues);
  int zeroGhostCells = 0;
  
  t->requires(Task::OldDW, qfluxE_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, qfluxW_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, qfluxN_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, qfluxS_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, qfluxT_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, qfluxB_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, radiationSrc_CCLabel, Ghost::None, zeroGhostCells);

  t->computes(qfluxE_CCLabel);
  t->computes(qfluxW_CCLabel);
  t->computes(qfluxN_CCLabel);
  t->computes(qfluxS_CCLabel);
  t->computes(qfluxT_CCLabel);
  t->computes(qfluxB_CCLabel);
  t->computes(radiationSrc_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual copy of old values of fluxes and source term
//****************************************************************************

void
RadiationDriver::copyValues(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int iceIndex = 0;
    int matlIndex = d_sharedState->getICEMaterial(iceIndex)->getDWIndex();
    int zeroGhostCells = 0;

    constCCVariable<double> oldFluxE;
    constCCVariable<double> oldFluxW;
    constCCVariable<double> oldFluxN;
    constCCVariable<double> oldFluxS;
    constCCVariable<double> oldFluxT;
    constCCVariable<double> oldFluxB;
    constCCVariable<double> oldRadiationSrc;

    old_dw->get(oldFluxE, qfluxE_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
    old_dw->get(oldFluxW, qfluxW_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
    old_dw->get(oldFluxN, qfluxN_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
    old_dw->get(oldFluxS, qfluxS_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
    old_dw->get(oldFluxT, qfluxT_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
    old_dw->get(oldFluxB, qfluxB_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
    old_dw->get(oldRadiationSrc, radiationSrc_CCLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);

    CCVariable<double> fluxE;
    CCVariable<double> fluxW;
    CCVariable<double> fluxN;
    CCVariable<double> fluxS;
    CCVariable<double> fluxT;
    CCVariable<double> fluxB;
    CCVariable<double> radiationSrc;

    new_dw->allocateAndPut(fluxE, qfluxE_CCLabel, matlIndex, patch);    
    new_dw->allocateAndPut(fluxW, qfluxW_CCLabel, matlIndex, patch);    
    new_dw->allocateAndPut(fluxN, qfluxN_CCLabel, matlIndex, patch);    
    new_dw->allocateAndPut(fluxS, qfluxS_CCLabel, matlIndex, patch);    
    new_dw->allocateAndPut(fluxT, qfluxT_CCLabel, matlIndex, patch);    
    new_dw->allocateAndPut(fluxB, qfluxB_CCLabel, matlIndex, patch);    
    new_dw->allocateAndPut(radiationSrc, radiationSrc_CCLabel, matlIndex, patch);    

    fluxE.copyData(oldFluxE);
    fluxW.copyData(oldFluxW);
    fluxN.copyData(oldFluxN);
    fluxS.copyData(oldFluxS);
    fluxT.copyData(oldFluxT);
    fluxB.copyData(oldFluxB);
    radiationSrc.copyData(oldRadiationSrc);
    
  }
}

//****************************************************************************
// schedule computation of radiative properties
//****************************************************************************

void
RadiationDriver::scheduleComputeProps(const LevelP& level,
				       SchedulerP& sched,
				       const PatchSet* patches,
				       const MaterialSet* matls)
{
  Task* t=scinew Task("RadiationDriver::computeProps",
		      this, &RadiationDriver::computeProps);
  int zeroGhostCells = 0;
  
  t->requires(Task::NewDW, co2_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, h2o_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, temp_CCLabel, Ghost::None, zeroGhostCells);
  // Below is for later, when we get sootVF from reaction table.  But
  // currently we compute sootVF from mixture fraction and temperature inside
  // the properties function
  //  t->requires(Task::NewDW, sootVF_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, mixfrac_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, density_CCLabel, Ghost::None, zeroGhostCells);

  t->computes(tempCopy_CCLabel);
  t->computes(sootVFCopy_CCLabel);
  t->computes(abskg_CCLabel);
  t->computes(esrcg_CCLabel);
  t->computes(shgamma_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual compute of properties
//****************************************************************************

void
RadiationDriver::computeProps(const ProcessorGroup* pc,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int iceIndex = 0;
    int matlIndex = d_sharedState->getICEMaterial(iceIndex)->getDWIndex();
    int zeroGhostCells = 0;
    
    RadiationVariables radVars;
    RadiationConstVariables constRadVars;
    
    PerPatch<Models_CellInformationP> cellInfoP;
    if (new_dw->exists(d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew Models_CellInformation(patch));
      new_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    }
    Models_CellInformation* cellinfo = cellInfoP.get().get_rep();

    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();

    //    if (d_radCounter%d_radCalcFreq == 0) {
      new_dw->get(constRadVars.temperature, temp_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
      new_dw->get(constRadVars.co2, co2_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
      new_dw->get(constRadVars.h2o, h2o_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);

      new_dw->allocateAndPut(radVars.temperature, tempCopy_CCLabel, 
			     matlIndex, patch);
      radVars.temperature.copyData(constRadVars.temperature);

      // We will use constRadVars.sootVF when we get it from the table.
      // For now, calculate sootVF in radcoef.F
      // As long as the test routines are embedded in the properties and
      // boundary conditions, we will need radVars.sootVF, because those
      // test routines modify the soot properties (bad design).
      // So until that is fixed, the idea is to get constRadVars.sootVF
      // and copy it to radVars.sootVF, and use radVars.sootVF in our
      // calculations.  Instead, we now compute sootVF from mixture fraction
      // and temperature inside the properties function
      // 
      //      new_dw->get(constRadVars.sootVF, sootVF_CCLabel, matlIndex, patch,
      //		  Ghost::None, zeroGhostCells);
      old_dw->get(constRadVars.mixfrac, mixfrac_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
      new_dw->get(constRadVars.density, density_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
      new_dw->allocateAndPut(radVars.sootVF, sootVFCopy_CCLabel, 
			     matlIndex, patch);
      //      radVars.sootVF.copyData(constRadVars.sootVF);

      new_dw->allocateAndPut(radVars.ABSKG, abskg_CCLabel, 
			     matlIndex, patch);
      new_dw->allocateAndPut(radVars.ESRCG, esrcg_CCLabel, 
			     matlIndex, patch);
      new_dw->allocateAndPut(radVars.shgamma, shgamma_CCLabel, 
			     matlIndex, patch);

      d_DORadiation->computeRadiationProps(pc, patch, cellinfo,
					   &radVars, &constRadVars);
      //    }
  }
}

//****************************************************************************
// schedule computation of radiative boundary condition
//****************************************************************************
void 
RadiationDriver::scheduleBoundaryCondition(const LevelP& level,
					   SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls)
{
  Task* t=scinew Task("RadiationDriver::boundaryCondition",
		      this, &RadiationDriver::boundaryCondition);

  t->modifies(tempCopy_CCLabel);
  t->modifies(abskg_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual boundary condition for radiation
//****************************************************************************

void
RadiationDriver::boundaryCondition(const ProcessorGroup* pc,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* ,
				   DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int iceIndex = 0;
    int matlIndex = d_sharedState->getICEMaterial(iceIndex)->getDWIndex();
    
    RadiationVariables radVars;
    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();

    //    if (d_radCounter%d_radCalcFreq == 0) {

      new_dw->getModifiable(radVars.temperature, tempCopy_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.ABSKG, abskg_CCLabel, matlIndex, patch);

      d_DORadiation->boundaryCondition(pc, patch, &radVars);

      //    }
  }
}

//****************************************************************************
// schedule solve for radiative intensities, radiative source, and heat fluxes
//****************************************************************************

void
RadiationDriver::scheduleIntensitySolve(const LevelP& level,
					SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls,
					const ModelInfo* mi)
{
  Task* t=scinew Task("RadiationDriver::intensitySolve",
		      this, &RadiationDriver::intensitySolve, mi);
  int zeroGhostCells = 0;

  t->requires(Task::NewDW, co2_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, h2o_CCLabel, Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, sootVFCopy_CCLabel, Ghost::None, zeroGhostCells);

  t->modifies(tempCopy_CCLabel);
  t->modifies(abskg_CCLabel);
  t->modifies(esrcg_CCLabel);
  t->modifies(shgamma_CCLabel);

  t->modifies(qfluxE_CCLabel);
  t->modifies(qfluxW_CCLabel);
  t->modifies(qfluxN_CCLabel);
  t->modifies(qfluxS_CCLabel);
  t->modifies(qfluxT_CCLabel);
  t->modifies(qfluxB_CCLabel);
  t->modifies(radiationSrc_CCLabel);

  t->modifies(mi->energy_source_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual solve for radiative intensities, radiative source and heat fluxes
//****************************************************************************

void
RadiationDriver::intensitySolve(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* ,
				DataWarehouse* new_dw,
				const ModelInfo* mi)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    int iceIndex = 0;
    int matlIndex = d_sharedState->getICEMaterial(iceIndex)->getDWIndex();
    int zeroGhostCells = 0;

    RadiationVariables radVars;
    RadiationConstVariables constRadVars;
    CCVariable<double> energySource;

    IntVector domLo = patch->getCellLowIndex();
    IntVector domHi = patch->getCellHighIndex();
    CCVariable<double> zeroSource;
    zeroSource.allocate(domLo,domHi);
    zeroSource.initialize(0.0);

    PerPatch<Models_CellInformationP> cellInfoP;
    if (new_dw->exists(d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew Models_CellInformation(patch));
      new_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    }
    Models_CellInformation* cellinfo = cellInfoP.get().get_rep();

    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();

    //    if (d_radCounter%d_radCalcFreq == 0) {

      new_dw->get(constRadVars.co2, co2_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
      new_dw->get(constRadVars.h2o, h2o_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
      new_dw->get(constRadVars.sootVF, sootVFCopy_CCLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);

      new_dw->getModifiable(radVars.temperature, tempCopy_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.ABSKG, abskg_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.ESRCG, esrcg_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.shgamma, shgamma_CCLabel, matlIndex, patch);

      new_dw->getModifiable(radVars.qfluxe, qfluxE_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.qfluxw, qfluxW_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.qfluxn, qfluxN_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.qfluxs, qfluxS_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.qfluxt, qfluxT_CCLabel, matlIndex, patch);
      new_dw->getModifiable(radVars.qfluxb, qfluxB_CCLabel, matlIndex, patch);

      new_dw->getModifiable(radVars.src, radiationSrc_CCLabel, matlIndex, patch);
      new_dw->getModifiable(energySource, mi->energy_source_CCLabel, matlIndex, patch);

      d_DORadiation->intensitysolve(pc, patch, cellinfo, &radVars, &constRadVars);

      //    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      //      energySource[c] += radVars.src[c];
      energySource[c] += zeroSource[c];
    }
  }
}
//______________________________________________________________________
void RadiationDriver::scheduleModifyThermoTransportProperties(SchedulerP&,
							      const LevelP&,
							      const MaterialSet*)
{
  // not applicable  
}
void RadiationDriver::computeSpecificHeat(CCVariable<double>&,
					  const Patch*,
					  DataWarehouse*,
					  const int)
{
  // not applicable
}
//______________________________________________________________________
//
void RadiationDriver::scheduleErrorEstimate(const LevelP&,
					    SchedulerP&)
{
  // This may not be ever implemented, since it is known that the radiation
  // calculations work fine with a coarse mesh.
}
//__________________________________
void RadiationDriver::scheduleTestConservation(SchedulerP&,
					       const PatchSet*,
					       const ModelInfo*)
{
  // do nothing; there is a conservation test in Models_DORadiationModel;
  // perhaps at a later time, I will move it here.
}
