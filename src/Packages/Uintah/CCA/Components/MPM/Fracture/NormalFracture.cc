#include "NormalFracture.h"

#include "ParticlesNeighbor.h"
#include "Connectivity.h"
#include "CrackFace.h"
#include "CellsNeighbor.h"

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <stdlib.h>
#include <list>

namespace Uintah {
using namespace SCIRun;

void
NormalFracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw)
{
}

void NormalFracture::computeBoundaryContact(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  int matlindex = mpm_matl->getDWIndex();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    //patchAndGhost data
    ParticleVariable<Point>  pX_pg;
    ParticleVariable<double> pVolume_pg;
    ParticleVariable<int>    pIsBroken_pg;
    ParticleVariable<Vector> pCrackNormal_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);

    //patchOnly data
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

    //cout<<"computeFracture:computeBoundaryContact: "<< pset_p->numParticles()<<endl;

    ParticleVariable<Point>  pX_p;
    new_dw->get(pX_p, lb->pXXLabel, pset_p);

    //particle index exchange from patch to patch+ghost
    vector<int> pIdxEx( pset_p->numParticles() );
    fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);

    Lattice lattice(pX_pg);

    //Allocate new data
    ParticleVariable<Vector> pTouchNormal_p_new;
    new_dw->allocate(pTouchNormal_p_new, lb->pTouchNormalLabel, pset_p);
  
    for(ParticleSubset::iterator iter = pset_p->begin();
      iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = pIdxEx[pIdx_p];

      const Point& particlePoint = pX_pg[pIdx_pg];
      double size0 = pow(pVolume_pg[pIdx_pg],1./3.);
    
      ParticlesNeighbor particles;
      lattice.getParticlesNeighbor(particlePoint, particles);
      int particlesNumber = particles.size();

      int touchFacetsNum = 0;
      pTouchNormal_p_new[pIdx_p] = Vector(0.,0.,0.);

      //other side
      for(int j=0; j<particlesNumber; j++) {
        int idx_pg = particles[j];
        if( pIdx_pg == idx_pg ) continue;
        if(pIsBroken_pg[idx_pg] > 0) {
          double size1 = pow(pVolume_pg[idx_pg],1./3.);
          const Vector& n1 = pCrackNormal_pg[idx_pg];
      
          Vector dis = particlePoint - pX_pg[idx_pg];
          double vDis = Dot( dis, n1 );
	  
          if( vDis>0 && vDis<(size0+size1)/2 ) {
            double hDis = (dis - n1 * vDis).length();
            if(hDis < size1/2) {
	      pTouchNormal_p_new[pIdx_p] -= n1;
  	      touchFacetsNum ++;
	    }
          }
	}
      }

      //self side
      const Vector& n0 = pCrackNormal_pg[pIdx_pg];
      for(int j=0; j<particlesNumber; j++) {
        int idx_pg = particles[j];
        if( pIdx_pg == idx_pg ) continue;
 
        double size1 = pow(pVolume_pg[idx_pg],1./3.);
        Vector dis = pX_pg[idx_pg] - particlePoint;

        double vDis = Dot( dis, n0 );
        if( vDis>0 && vDis<(size0+size1)/2 ) {
          double hDis = (dis - n0 * vDis).length();
          if(hDis < size0/2) {
	    pTouchNormal_p_new[pIdx_p] += n0;
  	    touchFacetsNum ++;
          }
        }
      }

      if(touchFacetsNum>0) {
        pTouchNormal_p_new[pIdx_p].normalize();
        //cout<<"HAVE crack contact!"<<endl;
      }
    }

    new_dw->put(pTouchNormal_p_new, lb->pTouchNormalLabel);
  }
}

void NormalFracture::computeConnectivity(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  static Vector zero(0.,0.,0.);

  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    int matlindex = mpm_matl->getDWIndex();
    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    ParticleVariable<Point>  pX_pg;
    ParticleVariable<double> pVolume_pg;
    ParticleVariable<int>    pIsBroken_pg;
    ParticleVariable<Vector> pCrackNormal_pg;
    ParticleVariable<Vector> pTouchNormal_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
    new_dw->get(pTouchNormal_pg, lb->pTouchNormalLabel, pset_pg);

    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

    //cout<<"computeConnectivity:computeBoundaryContact: "<< pset_p->numParticles()<<endl;

    ParticleVariable<Point>  pX_p;
    new_dw->get(pX_p, lb->pXXLabel, pset_p);

    vector<int> pIdxEx( pset_p->numParticles() );
    fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);

    ParticleVariable<int>       pConnectivity_p_new;
    ParticleVariable<Vector>    pContactNormal_p_new;
    new_dw->allocate(pConnectivity_p_new, lb->pConnectivityLabel, pset_p);
    new_dw->allocate(pContactNormal_p_new, lb->pContactNormalLabel, pset_p);

    Lattice lattice(pX_pg);
    ParticlesNeighbor particles;
    IntVector cellIdx;
    IntVector nodeIdx[8];

    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
      pContactNormal_p_new[pIdx_p] = zero;
    
      patch->findCell(pX_p[pIdx_p],cellIdx);
      particles.clear();
      particles.buildIn(cellIdx,lattice);
      int particlesNumber = particles.size();

      //Connectivity Info
      patch->findNodesFromCell(cellIdx,nodeIdx);
    
      int conn[8];
      for(int k=0;k<8;++k) {
        const Point& A = pX_pg[pIdx_pg];
        Point B = patch->nodePosition(nodeIdx[k]);

        conn[k] = 1;
    
        for(int i=0; i<particlesNumber; i++) {
          int pidx_pg = particles[i];

          if(pidx_pg == pIdx_pg) {
            if( pTouchNormal_pg[pidx_pg].length2() > 0.5 ) {
	        conn[k] = 2;
	        pContactNormal_p_new[pIdx_p] = pTouchNormal_pg[pidx_pg];
	        break;
	    }

	    if(pIsBroken_pg[pidx_pg]>0) {
	      if( Dot(B-A,pCrackNormal_pg[pidx_pg]) > 0 ) {
	        conn[k] = 0;
		break;
	      }
	    }
	  }
	
	  else {
            double r = pow(pVolume_pg[pidx_pg],0.3333333333)/2;
            double r2 = 2*r*r;
	
            if( pTouchNormal_pg[pidx_pg].length2() > 0.5 ) {
              Point O = pX_pg[pidx_pg] + pTouchNormal_pg[pidx_pg] * r;
              if( !particles.visible(A,B,O,pTouchNormal_pg[pidx_pg],r2) ) {
	        conn[k] = 2;
	        pContactNormal_p_new[pIdx_p] = pTouchNormal_pg[pidx_pg];
	        break;
	      }
	    }
            if(conn[k]==1) {
	      if(pIsBroken_pg[pidx_pg]>0) {
	        Point O = pX_pg[pidx_pg] + pCrackNormal_pg[pidx_pg] * r;
	        if( !particles.visible(A,B,O,pCrackNormal_pg[pidx_pg],r2) ) {
	          conn[k] = 0;
		  break;
	        }
	      }
  	    }
	  }
        }
      }

      Connectivity connectivity(conn);
      pConnectivity_p_new[pIdx_p] = connectivity.flag();
    }
  
    new_dw->put(pConnectivity_p_new, lb->pConnectivityLabel);
    new_dw->put(pContactNormal_p_new, lb->pContactNormalLabel);
  }
}

void NormalFracture::computeFracture(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  ParticleVariable<Point>  pX_pg;
  ParticleVariable<int>    pIsBroken_pg;
  ParticleVariable<Vector> pCrackNormal_pg;
  ParticleVariable<double> pVolume_pg;

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
  old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
  old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);

  //patchOnly data
  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  //cout<<"computeFracture:numParticles: "<< pset_p->numParticles()<<endl;
  
  ParticleVariable<Point>  pX_p;
  //ParticleVariable<double> pStrainEnergy_p;
  ParticleVariable<double> pToughness_p;
  ParticleVariable<Vector> pRotationRate_p;
  ParticleVariable<int> pConnectivity_p;
  ParticleVariable<Matrix3> pStress_p;

  new_dw->get(pX_p, lb->pXXLabel, pset_p);
  //new_dw->get(pStrainEnergy_p, lb->pStrainEnergyLabel, pset_p);
  old_dw->get(pToughness_p, lb->pToughnessLabel, pset_p);
  new_dw->get(pRotationRate_p, lb->pRotationRateLabel, pset_p);
  new_dw->get(pConnectivity_p, lb->pConnectivityLabel, pset_p);
  new_dw->get(pStress_p, lb->pStressAfterStrainRateLabel, pset_p);

  //particle index exchange from patch to patch+ghost
  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);
  Lattice lattice(pX_pg);

  NCVariable<Matrix3> gStress;
  new_dw->get(gStress, lb->gStressForSavingLabel,
    matlindex, patch, Ghost::AroundCells, 1);

  ParticleVariable<int> pIsBroken_p_new;
  ParticleVariable<Vector> pCrackNormal_p_new;
  ParticleVariable<Matrix3> pStress_p_new;
  
  new_dw->allocate(pIsBroken_p_new, lb->pIsBrokenLabel, pset_p);
  new_dw->allocate(pCrackNormal_p_new, lb->pCrackNormalLabel, pset_p);
  new_dw->allocate(pStress_p_new, lb->pStressLabel, pset_p);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  const Vector dx = patch->dCell();
  double cellLength = dx.x();
  
  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx_p = *iter;
    particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
    pIsBroken_p_new[pIdx_p] = pIsBroken_pg[pIdx_pg];
    pCrackNormal_p_new[pIdx_p] = pCrackNormal_pg[pIdx_pg];
    
    if( pIsBroken_pg[pIdx_pg] > 0 ) {
      pCrackNormal_p_new[pIdx_p] += Cross( pRotationRate_p[pIdx_p] * delT, 
	                            pCrackNormal_pg[pIdx_pg] );
      pCrackNormal_p_new[pIdx_p].normalize();
    }
    
    pStress_p_new[pIdx_p] = pStress_p[pIdx_p];

    if(pIsBroken_pg[pIdx_pg] > 0) continue;
    
    //check toughness
    const Matrix3& stress = pStress_p[pIdx_p];
    double stre = sqrt( stress(1,1) * stress(1,1) +
                        stress(2,2) * stress(2,2) +
			stress(3,3) * stress(3,3) );
    if(stre * cellLength*2 < pToughness_p[pIdx_p]) continue;

    //crack direction
    Vector nx,ny;
    double sigmay,sigmax;
    getFirstAndSecondEigenvalue(stress, ny, sigmay, nx, sigmax);
    if(sigmay<=0) continue;

/*
    IntVector ni[8];
    Vector d_S[8];
    patch->findCellAndShapeDerivatives(pX_p[pIdx_p], ni, d_S);

    Connectivity connectivity(pConnectivity_p[pIdx_p]);
    int conn[8];
    connectivity.getInfo(conn);
    Connectivity::modifyShapeDerivatives(conn,d_S,Connectivity::connect);

    Vector dSn (0.,0.,0.);
    for(int k = 0; k < 8; k++) {
      if(conn[k]==Connectivity::connect) {
	double gStressNN = Dot(ny, gStress[ni[k]]*ny);
	for (int i = 1; i<=3; i++) {
	  dSn[i] += gStressNN * d_S[k][i] / dx[i];
	}
      }
    }
    if( Dot(dSn,ny)<0 ) ny=-ny;
*/

    //energy release rate
    static double Gmax=0;
    double G;
    ParticlesNeighbor particles;
    
    lattice.getParticlesNeighbor(pX_p[pIdx_p], particles);
    if( !particles.computeEnergyReleaseRate(
         pIdx_p,nx,ny,sigmay,pX_pg,pVolume_pg,G) ) continue;

    if(G>Gmax) {
      Gmax=G;
      cout<<"Max energy release rate: "<<Gmax<<endl;
    }
    
    if( G < pToughness_p[pIdx_p] ) continue;

    //geometry acceptance
    int particlesNumber = particles.size();
    double psize = pow( pVolume_pg[pIdx_pg], 1./3. );

    bool accept = false;
    for(int p=0; p<particlesNumber; p++) {
      int idx_pg = particles[p];

      Vector dis = pX_pg[idx_pg] - pX_pg[pIdx_pg];
      double vdis = Dot(ny,dis);
      if( fabs(vdis) > psize/2 ) continue;
      Vector Vdis = ny * vdis;
      if( (dis-Vdis).length() > psize*1.5 ) continue;
      
      if( pIsBroken_pg[idx_pg]>0) {
        if( Dot(pCrackNormal_pg[idx_pg],ny) > 0.9 ) {
          if( fabs(Dot(dis,pCrackNormal_pg[idx_pg])) < psize/2 ) {
            accept = true;
            break;
          }
        }
      }
    }
    if(!accept) continue;

    cout<<"crack! "<<"nx="<<nx<<" ny="<<ny<<endl;

    //stress release
    for(int i=1;i<=3;++i)
    for(int j=1;j<=3;++j) {
      pStress_p_new[pIdx_p](i,j) -= ny[i] * sigmay * ny[j];
    }
    
    pCrackNormal_p_new[pIdx_p] = ny;
    pIsBroken_p_new[pIdx_p]++;
  }

  new_dw->put(pIsBroken_p_new, lb->pIsBrokenLabel_preReloc);
  new_dw->put(pCrackNormal_p_new, lb->pCrackNormalLabel_preReloc);
  new_dw->put(pStress_p_new, lb->pStressAfterFractureReleaseLabel);
  new_dw->put(pToughness_p, lb->pToughnessLabel_preReloc);
  }
}

NormalFracture::
NormalFracture(ProblemSpecP& ps) : Fracture(ps)
{
}

NormalFracture::~NormalFracture()
{
}

//for debugging
bool
NormalFracture::
isDebugParticle(const Point& p)
{
  double cellsize=0.1;
  double particlesize = cellsize/2;
  Point p0 = Point(cellsize/4,cellsize/4,cellsize/4);
  Vector d = p - p0;
  
  if( fabs(d.x()) < particlesize/2 &&
      fabs(d.y()) < particlesize/2 &&
      fabs(d.z()) < particlesize/2 ) return true;
  else return false;
}

} //namespace Uintah
