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

#include <CCA/Components/MPM/MPMCommon.h> 
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
using namespace std;
using namespace Uintah;

static DebugStream cout_doing("MPM", false);

MPMCommon::MPMCommon(const ProcessorGroup* myworld,
                     MaterialManagerP materialManager) :
  ApplicationCommon(myworld, materialManager)
{
  lb = scinew MPMLabel();
}

MPMCommon::~MPMCommon()
{
  delete lb;
}

//______________________________________________________________________
//
void MPMCommon::materialProblemSetup(const ProblemSpecP& prob_spec, 
                                     MPMFlags* flags, 
                                     bool isRestart)
{
  d_flags = flags;

  //! so all components can know how many particle ghost cells to ask for
  d_flags->d_particle_ghost_type  = particle_ghost_type;
  d_flags->d_particle_ghost_layer = particle_ghost_layer;
  
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps     = prob_spec->findBlockWithOutAttribute( "MaterialProperties" );
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock( "MPM" );
  for( ProblemSpecP ps = mpm_mat_ps->findBlock( "material" ); ps != nullptr; ps = ps->findNextBlock( "material" ) ) {
    string index( "" );
    ps->getAttribute( "index",index );
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, m_materialManager, flags,isRestart);

    mat->registerParticleState( d_particleState,
                                d_particleState_preReloc );
    
    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      m_materialManager->registerMaterial( "MPM", mat,index_val);
    }
    else{
      m_materialManager->registerMaterial( "MPM", mat);
    }
  }
}
//______________________________________________________________________
//
void MPMCommon::scheduleUpdateStress_DamageErosionModels(SchedulerP   & sched,
                                                     const PatchSet * patches,
                                                     const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"MPMCommon::scheduleUpdateStress_DamageErosionModels");
  
  Task* t = scinew Task("MPM::updateStress_DamageErosionModels", this, 
                        &MPMCommon::updateStress_DamageErosionModels);

  t->requires(Task::OldDW, lb->simulationTimeLabel);
  
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    
    DamageModel* dm = mpm_matl->getDamageModel();
    dm->addComputesAndRequires(t, mpm_matl);
    
    ErosionModel* em = mpm_matl->getErosionModel();
    em->addComputesAndRequires(t, mpm_matl);
  }
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void MPMCommon::updateStress_DamageErosionModels(const ProcessorGroup *,
                                                 const PatchSubset    * patches,
                                                 const MaterialSubset * ,
                                                 DataWarehouse        * old_dw,
                                                 DataWarehouse        * new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
   
    printTask(patches, patch,cout_doing,
              "Doing updateStress_DamageModel");

    int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
    for(int m = 0; m < numMPMMatls; m++){
    
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      
      DamageModel* dm = mpm_matl->getDamageModel();
      dm->computeSomething( pset, mpm_matl, patch, old_dw, new_dw );
      
      ErosionModel* em = mpm_matl->getErosionModel();
      em->updateStress_Erosion( pset, old_dw, new_dw );
    }
  }
}

//______________________________________________________________________
//  Utilities
//    Put a std::vector of reduction variables into the new_dw
template< class T>
void MPMCommon::put_sum_vartype( std::vector<T>  reductionVars,
                                 const VarLabel * label,
                                 DataWarehouse  * new_dw )
{
  unsigned int numMatls = reductionVars.size();

//  if( numMatls > 1){    // ignore for single matl problems

    for (unsigned int m = 0; m < numMatls ; m++ ) {
      MPMMaterial* matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      int dwi = matl->getDWIndex();

      typedef ReductionVariable<T, Reductions::Sum<T> > sumVartype;

      new_dw->put( sumVartype(reductionVars[m]),  label, nullptr, dwi);
    }
//  }
}

//______________________________________________________________________
//    get a std::vector of reduction variables from the DW
template< class T>
std::vector<T> MPMCommon::get_sum_vartype( unsigned int    numMPMMatls,
                                           const VarLabel * label,
                                           DataWarehouse  * new_dw )
{
  std::vector<T>  reductionVars;
//  if( numMPMMatls > 1){    // ignore for single matl problems

    for (unsigned int m = 0; m < numMPMMatls ; m++ ) {

      MPMMaterial* matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      int dwi = matl->getDWIndex();

      typedef ReductionVariable<T, Reductions::Sum<T> > sumVartype;
      sumVartype reductionVar;
      
      new_dw->get( reductionVar, label, nullptr, dwi);
      reductionVars.push_back(reductionVar);
    }
//  }
  return reductionVars;
}
//__________________________________
//
// Explicit template instantiations:

template void MPMCommon::put_sum_vartype( std::vector<double> rv ,const VarLabel * l, DataWarehouse  * dw);
template void MPMCommon::put_sum_vartype( std::vector<Vector> rv ,const VarLabel * l, DataWarehouse  * dw);

template std::vector<double> MPMCommon::get_sum_vartype(unsigned int i, const VarLabel * l, DataWarehouse  * dw);
template std::vector<Vector> MPMCommon::get_sum_vartype(unsigned int i, const VarLabel * l, DataWarehouse  * dw);
