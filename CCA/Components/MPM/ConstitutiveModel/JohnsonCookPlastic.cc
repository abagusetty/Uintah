#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCookPlastic.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;


JohnsonCookPlastic::JohnsonCookPlastic(ProblemSpecP& ps)
{
  ps->require("A",d_CM.A);
  ps->require("B",d_CM.B);
  ps->require("C",d_CM.C);
  ps->require("n",d_CM.n);
  ps->require("m",d_CM.m);
  d_CM.TRoom = 294;
  d_CM.TMelt = 1594;
}
	 
JohnsonCookPlastic::~JohnsonCookPlastic()
{
}
	 
void 
JohnsonCookPlastic::addInitialComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet*) const
{
}

void 
JohnsonCookPlastic::addComputesAndRequires(Task* ,
				    const MPMMaterial* ,
				    const PatchSet*) const
{
}

void 
JohnsonCookPlastic::addParticleState(std::vector<const VarLabel*>& ,
				     std::vector<const VarLabel*>& )
{
}

void 
JohnsonCookPlastic::allocateCMDataAddRequires(Task* ,
					      const MPMMaterial* ,
					      const PatchSet* ,
					      MPMLabel* ) const
{
}

void JohnsonCookPlastic::allocateCMDataAdd(DataWarehouse* ,
					   ParticleSubset* ,
					   map<const VarLabel*, 
                                           ParticleVariableBase*>* ,
					   ParticleSubset* ,
					   DataWarehouse* )
{
}


void 
JohnsonCookPlastic::initializeInternalVars(ParticleSubset* ,
				           DataWarehouse* )
{
}

void 
JohnsonCookPlastic::getInternalVars(ParticleSubset* ,
                                    DataWarehouse* ) 
{
}

void 
JohnsonCookPlastic::allocateAndPutInternalVars(ParticleSubset* ,
                                               DataWarehouse* ) 
{
}

void 
JohnsonCookPlastic::allocateAndPutRigid(ParticleSubset* ,
                                        DataWarehouse* ) 
{
}

void
JohnsonCookPlastic::updateElastic(const particleIndex )
{
}

void
JohnsonCookPlastic::updatePlastic(const particleIndex , const double& )
{
}

double 
JohnsonCookPlastic::computeFlowStress(const PlasticityState* state,
				      const double& delT,
				      const double& tolerance,
				      const MPMMaterial* matl,
				      const particleIndex idx)
{
  double epdot = state->plasticStrainRate;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double Tr = matl->getRoomTemperature();
  double Tm = state->meltingTemp;

  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_CM.C);
  else
    strainRatePart = 1.0 + d_CM.C*log(epdot);
  ASSERT(T < Tm);
  d_CM.TRoom = Tr;  d_CM.TMelt = Tm;
  double m = d_CM.m;
  double Tstar = (T-Tr)/(Tm-Tr);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));
  return (strainPart*strainRatePart*tempPart);
}

/*! In this case, \f$\dot{\epsilon_p}\f$ is the time derivative of 
    \f$\epsilon_p\f$.  Hence, the evolution law of the internal variables
    \f$ \dot{q_\alpha} = \gamma h_\alpha \f$ requires 
    \f$\gamma = \dot{\epsilon_p}\f$ and \f$ h_\alpha = 1\f$. */
void 
JohnsonCookPlastic::computeTangentModulus(const Matrix3& stress,
                                          const PlasticityState* state,
				          const double& ,
                                          const MPMMaterial* matl,
                                          const particleIndex idx,
				          TangentModulusTensor& Ce,
				          TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = stress - one*(stress.Trace()/3.0);

  // Calculate the equivalent stress
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Get f_q = dsigma/dep (h = 1, therefore f_q.h = f_q)
  double f_q = evalDerivativeWRTPlasticStrain(state, idx);

  // Form the elastic-plastic tangent modulus
  Matrix3 Cr, rC;
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      Cr(ii1,jj1) = 0.0;
      rC(ii1,jj1) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cr(ii1,jj1) += Ce(ii,jj,kk,ll)*rr(kk1,ll+1);
          rC(ii1,jj1) += rr(kk1,ll+1)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii,jj)*rr(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
                             Cr(ii1,jj1)*rC(kk1,ll+1)/(-f_q + rCr);
	}  
      }  
    }  
  }  
}

void
JohnsonCookPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
						const particleIndex idx,
						Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}


double
JohnsonCookPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                                   const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double Tm = state->meltingTemp;

  // Calculate strain rate part
  double strainRatePart = 1.0;
  if (epdot < 1.0) strainRatePart = pow((1.0 + epdot),d_CM.C);
  else strainRatePart = 1.0 + d_CM.C*log(epdot);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T-d_CM.TRoom)/(Tm-d_CM.TRoom);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double D = strainRatePart*tempPart;

  double deriv = d_CM.B*d_CM.n*D*pow(ep,d_CM.n-1)/ep;
  return deriv;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
JohnsonCookPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
JohnsonCookPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
JohnsonCookPlastic::evalDerivativeWRTTemperature(const PlasticityState* state,
                                                 const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double Tm = state->meltingTemp;

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate strain rate part
  double strainRatePart = 1.0;
  if (epdot < 1.0) strainRatePart = pow((1.0 + epdot),d_CM.C);
  else strainRatePart = 1.0 + d_CM.C*log(epdot);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T-d_CM.TRoom)/(Tm-d_CM.TRoom);

  double F = strainPart*strainRatePart;
  double deriv = - m*F*pow(Tstar,m)/(T-d_CM.TRoom);
  return deriv;
}

double
JohnsonCookPlastic::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                                const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double Tm = state->meltingTemp;

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T-d_CM.TRoom)/(Tm-d_CM.TRoom);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double E = strainPart*tempPart;

  double deriv = E*d_CM.C/epdot;
  return deriv;

}


