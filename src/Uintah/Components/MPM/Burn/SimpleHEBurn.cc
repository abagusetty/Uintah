/* REFERENCED */
static char *id="@(#) $Id$";

// SimpleHEBurn.cc
//
// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "SimpleHEBurn.h"
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah::MPM;

SimpleHEBurn::SimpleHEBurn(ProblemSpecP& ps)
{
  // Constructor

  ps->require("a",a);
  ps->require("b",b); 
  d_burnable = true;
  
  std::cerr << "a = " << a << " b = " << b << std::endl;

}

SimpleHEBurn::~SimpleHEBurn()
{
  // Destructor

}

void SimpleHEBurn::initializeBurnModelData(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouseP& new_dw)
{
  //   const MPMLabel* lb = MPMLabel::getLabels();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<int> pIsIgnited;
   new_dw->allocate(pIsIgnited, lb->pIsIgnitedLabel, pset);
   CCVariable<double> burnedMass;
   new_dw->allocate(burnedMass, lb->cBurnedMassLabel, matl->getDWIndex(), patch);


   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
        particleIndex idx = *iter;
	pIsIgnited[idx]=0;
   }

   new_dw->put(pIsIgnited, lb->pIsIgnitedLabel);
   new_dw->put(burnedMass, lb->cBurnedMassLabel, matl->getDWIndex(), patch);
}

bool SimpleHEBurn::getBurns() const
{
  return d_burnable;
}

void SimpleHEBurn::addCheckIfComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const Patch* patch,
                                                 DataWarehouseP& old_dw,
                                                 DataWarehouseP& new_dw) const
{

  //  const MPMLabel* lb = MPMLabel::getLabels();

  task->requires(old_dw, lb->pIsIgnitedLabel, matl->getDWIndex(),
				patch, Ghost::None);

//  task->requires(new_dw, lb->pTemperatureRateLabel, matl->getDWIndex(),
//				patch, Ghost::None);

  task->requires(old_dw, lb->pSurfLabel, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(old_dw, lb->cBurnedMassLabel,matl->getDWIndex(), 
				patch, Ghost::None);

  task->computes(new_dw, lb->pIsIgnitedLabel_preReloc,matl->getDWIndex(),patch);

  task->computes(new_dw, lb->cBurnedMassLabel,matl->getDWIndex(),patch);
  task->computes(new_dw, lb->pSurfLabel_preReloc,matl->getDWIndex(),patch);

  task->requires(old_dw, lb->delTLabel);

}

void SimpleHEBurn::addMassRateComputesAndRequires(Task* task,
                                                  const MPMMaterial* matl,
                                                  const Patch* patch,
                                                  DataWarehouseP& old_dw,
                                                  DataWarehouseP& new_dw) const
{
  //  const MPMLabel* lb = MPMLabel::getLabels();

  task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(new_dw, lb->pIsIgnitedLabel_preReloc, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(old_dw, lb->delTLabel);

  task->requires(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(),
                                patch, Ghost::None);

  task->computes(new_dw, lb->pMassLabel_preReloc,matl->getDWIndex(),patch);

  task->computes(new_dw, lb->pVolumeLabel_preReloc,matl->getDWIndex(),patch);
}

void SimpleHEBurn::checkIfIgnited(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{

   //  double heatFluxToParticle;

  int matlindex = matl->getDWIndex();

  //  const MPMLabel* lb = MPMLabel::getLabels();
  // Create array for the particle's "IsIgnited" flag
  ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<int> pIsIgnited;
  old_dw->get(pIsIgnited, lb->pIsIgnitedLabel, pset);

  ParticleVariable<int> pissurf;
  old_dw->get(pissurf, lb->pSurfLabel, pset);

//  ParticleVariable<double> pTemperatureRate;
//  new_dw->get(pTemperatureRate, lb->pTemperatureRateLabel,
//                                matlindex, patch,Ghost::None,0);

  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  //  double specificHeat = matl->getSpecificHeat();

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

//     heatFluxToParticle = pmass[idx]*pTemperatureRate[idx]*delT
//							*specificHeat;

     //Insert some conditional on heatFluxToParticle here
     if(pissurf[idx] == 1){
	pIsIgnited[idx] = 1;
     }
     else {
	pIsIgnited[idx] = 0;
     }
  }

  new_dw->put(pIsIgnited,lb->pIsIgnitedLabel_preReloc);
  new_dw->put(pissurf,lb->pSurfLabel_preReloc);
}
 
void SimpleHEBurn::computeMassRate(const Patch* patch,
				   const MPMMaterial* matl,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw)
{

  int matlindex = matl->getDWIndex();
  //  const MPMLabel* lb = MPMLabel::getLabels();

  ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume,lb->pVolumeDeformedLabel, pset);
  ParticleVariable<Point> px;
  old_dw->get(px, lb->pXLabel, pset);

  ParticleVariable<int> pIsIgnited;
  new_dw->get(pIsIgnited, lb->pIsIgnitedLabel_preReloc, pset);

  CCVariable<double> burnedMass;
  old_dw->get(burnedMass, lb->cBurnedMassLabel,
			matlindex, patch,Ghost::None,0);

  ParticleSubset* remove_subset =
	new ParticleSubset(pset->getParticleSet(), false, matlindex, patch);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     if(pIsIgnited[idx]==1){
	IntVector ci;
	if(patch->findCell(px[idx],ci)){
	   pmass[idx]   = pmass[idx] - .5*a;
	   pvolume[idx] = pmass[idx]/1000.0;
	   burnedMass[ci] += .5*a;
	
	   if(pmass[idx]<=0.0){
	     pmass[idx]=0.0;
	     burnedMass[ci] -= .5*a;
	     remove_subset->addParticle(idx);
	   }
	}
     }
     else {
	// Do nothing to the particle mass and volume
     }
  }

  new_dw->put(pmass,lb->pMassLabel_preReloc);
  new_dw->put(pvolume,lb->pVolumeLabel_preReloc);
  new_dw->put(burnedMass,lb->cBurnedMassLabel, matlindex, patch);

  new_dw->deleteParticles(remove_subset);
}
 
// $Log$
// Revision 1.13  2000/07/05 23:43:32  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.12  2000/06/23 18:05:59  guilkey
// Used a different way of creating the delete_subset for the particles
// to be removed.
//
// Revision 1.11  2000/06/21 20:51:40  guilkey
// Implemented the removal of particles that are completely consumed.
// The function that does the removal doesn't yet work.
//
// Revision 1.10  2000/06/19 23:52:14  guilkey
// Added boolean d_burns so that certain stuff only gets done
// if a burn model is present.  Not to worry, the if's on this
// are not inside of inner loops.
//
// Revision 1.9  2000/06/16 23:23:38  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.8  2000/06/15 21:57:02  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.7  2000/06/15 18:08:35  guilkey
// A few changes to the SimpleHEBurn model.
//
// Revision 1.6  2000/06/13 23:05:35  guilkey
// Added some stuff to SimpleBurn model.
//
// Revision 1.5  2000/06/08 17:37:07  guilkey
// Fixed small error.
//
// Revision 1.4  2000/06/08 17:09:11  guilkey
// Changed an old_dw to a new_dw.
//
// Revision 1.3  2000/06/08 16:49:45  guilkey
// Added more stuff to the burn models.  Most infrastructure is now
// in place to change the mass and volume, we just need a little bit of science.
//
// Revision 1.2  2000/06/06 18:04:02  guilkey
// Added more stuff for the burn models.  Much to do still.
//
// Revision 1.1  2000/06/02 22:48:26  jas
// Added infrastructure for Burn models.
//
