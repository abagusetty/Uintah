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

#include "CamClayYield.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <cmath>

using namespace Uintah;
using namespace std;

CamClayYield::CamClayYield(ProblemSpecP& ps)
{
  ps->require("M",d_M);
}
         
CamClayYield::CamClayYield(const CamClayYield* yc)
{
  d_M = yc->d_M; 
}
         
CamClayYield::~CamClayYield()
{
}

void CamClayYield::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP yield_ps = ps->appendChild("plastic_yield_condition");
  yield_ps->setAttribute("type","camclay_yield_function");
  yield_ps->appendElement("M",d_M);

}
         
//--------------------------------------------------------------
// Evaluate yield condition (q = state->q
//                           p = state->p
//                           p_c = state->p_c)
//--------------------------------------------------------------
double 
CamClayYield::evalYieldCondition(const PlasticityState* state)
{
  double p = state->p;
  double q = state->q;
  double p_c = state->p_c;
  return q*q/(d_M*d_M) + p*(p - p_c);
}

//--------------------------------------------------------------
// Evaluate yield condition max (q = state->q
//                               p = state->p
//                               p_c = state->p_c)
//--------------------------------------------------------------
double 
CamClayYield::evalYieldConditionMax(const PlasticityState* state)
{
  double p_c = state->p_c;
  double qmax = fabs(0.5*d_M*p_c);
  return qmax*qmax/(d_M*d_M);
}

//--------------------------------------------------------------
// Derivatives needed by return algorithms and Newton iterations

//--------------------------------------------------------------
// Compute df/dp  where p = volumetric stress = 1/3 Tr(sigma)
//   df/dp = 2p - p_c
//--------------------------------------------------------------
double 
CamClayYield::computeVolStressDerivOfYieldFunction(const PlasticityState* state)
{
  // std::cout << " p = " << state->p << " pc = " << state->p_c << " dfdp = " << 2*state->p-state->p_c << endl;
  return (2.0*state->p - state->p_c);
}

//--------------------------------------------------------------
// Compute df/dq  
//   df/dq = 2q/M^2
//--------------------------------------------------------------
double 
CamClayYield::computeDevStressDerivOfYieldFunction(const PlasticityState* state)
{
  return 2.0*state->q/(d_M*d_M);
}

//--------------------------------------------------------------
// Compute d/depse_v(df/dp)
//   df/dp = 2p(epse_v, epse_s) - p_c(epse_v)
//   d/depse_v(df/dp) = 2dp/depse_v - dp_c/depse_v
//
// Requires:  Equation of state and internal variable
//--------------------------------------------------------------
double
CamClayYield::computeVolStrainDerivOfDfDp(const PlasticityState* state,
                                               const MPMEquationOfState* eos,
                                               const ShearModulusModel* ,
                                               const InternalVariableModel* intvar)
{
  double dpdepsev = eos->computeDpDepse_v(state);
  double dpcdepsev = intvar->computeVolStrainDerivOfInternalVariable(state);
  return 2.0*dpdepsev - dpcdepsev;
}

//--------------------------------------------------------------
// Compute d/depse_s(df/dp)
//   df/dp = 2p(epse_v, epse_s) - p_c(epse_v)
//   d/depse_s(df/dp) = 2dp/depse_s 
//
// Requires:  Equation of state 
//--------------------------------------------------------------
double
CamClayYield::computeDevStrainDerivOfDfDp(const PlasticityState* state,
                                                   const MPMEquationOfState* eos,
                                                   const ShearModulusModel* ,
                                                   const InternalVariableModel* )
{
  double dpdepses = eos->computeDpDepse_s(state);
  return 2.0*dpdepses;
}

//--------------------------------------------------------------
// Compute d/depse_v(df/dq)
//   df/dq = 2q(epse_v, epse_s)/M^2
//   d/depse_v(df/dq) = 2/M^2 dq/depse_v
//
// Requires:  Shear modulus model
//--------------------------------------------------------------
double
CamClayYield::computeVolStrainDerivOfDfDq(const PlasticityState* state,
                                               const MPMEquationOfState* ,
                                               const ShearModulusModel* shear,
                                               const InternalVariableModel* )
{
  double dqdepsev = shear->computeDqDepse_v(state);
  return (2.0*dqdepsev)/(d_M*d_M);
}

//--------------------------------------------------------------
// Compute d/depse_s(df/dq)
//   df/dq = 2q(epse_v, epse_s)/M^2
//   d/depse_s(df/dq) = 2/M^2 dq/depse_s
//
// Requires:  Shear modulus model
//--------------------------------------------------------------
double
CamClayYield::computeDevStrainDerivOfDfDq(const PlasticityState* state,
                                                   const MPMEquationOfState* ,
                                                   const ShearModulusModel* shear,
                                                   const InternalVariableModel* )
{
  double dqdepses = shear->computeDqDepse_s(state);
  return (2.0*dqdepses)/(d_M*d_M);
}

//--------------------------------------------------------------
// Compute df/depse_v
//   df/depse_v = df/dq dq/depse_v + df/dp dp/depse_v - p dp_c/depse_v
//
// Requires:  Equation of state, shear modulus model, internal variable model
//--------------------------------------------------------------
double
CamClayYield::computeVolStrainDerivOfYieldFunction(const PlasticityState* state,
                                                            const MPMEquationOfState* eos,
                                                            const ShearModulusModel* shear,
                                                            const InternalVariableModel* intvar)
{
  double dfdq = computeDevStressDerivOfYieldFunction(state);
  double dfdp = computeVolStressDerivOfYieldFunction(state);
  double dqdepsev = shear->computeDqDepse_v(state);
  double dpdepsev = eos->computeDpDepse_v(state);
  double dpcdepsev = intvar->computeVolStrainDerivOfInternalVariable(state);
  double dfdepsev = dfdq*dqdepsev + dfdp*dpdepsev - state->p*dpcdepsev;

  return dfdepsev;
}

//--------------------------------------------------------------
// Compute df/depse_s
//   df/depse_s = df/dq dq/depse_s + df/dp dp/depse_s 
//
// Requires:  Equation of state, shear modulus model
//--------------------------------------------------------------
double
CamClayYield::computeDevStrainDerivOfYieldFunction(const PlasticityState* state,
                                                            const MPMEquationOfState* eos,
                                                            const ShearModulusModel* shear,
                                                            const InternalVariableModel* )
{
  double dfdq = computeDevStressDerivOfYieldFunction(state);
  double dfdp = computeVolStressDerivOfYieldFunction(state);
  double dqdepses = shear->computeDqDepse_s(state);
  double dpdepses = eos->computeDpDepse_s(state);
  double dfdepses = dfdq*dqdepses + dfdp*dpdepses;

  return dfdepses;
}

//--------------------------------------------------------------
// Other yield condition functions

// Evaluate yield condition (s = deviatoric stress
//                           p = state->p
//                           p_c = state->p_c)
double 
CamClayYield::evalYieldCondition(const Matrix3& ,
                                      const PlasticityState* state)
{
  double p = state->p;
  double q = state->q;
  double pc = state->p_c;
  double dummy = 0.0;
  return evalYieldCondition(p, q, pc, 0.0, dummy);
}

double 
CamClayYield::evalYieldCondition(const double p,
                                      const double q,
                                      const double p_c,
                                      const double,
                                      double& )
{
  return q*q/(d_M*d_M) + p*(p - p_c);
}

//--------------------------------------------------------------
// Other derivatives 

// Compute df/dsigma
//    df/dsigma = (2p - p_c)/3 I + sqrt(3/2) 2q/M^2 s/||s||
//              = 1/3 df/dp I + sqrt(3/2) df/dq s/||s||
//              = 1/3 df/dp I + df/ds
// where
//    s = sigma - 1/3 tr(sigma) I
void 
CamClayYield::evalDerivOfYieldFunction(const Matrix3& sig,
                                                const double p_c,
                                                const double ,
                                                Matrix3& derivative)
{
  Matrix3 One; One.Identity();
  double p = sig.Trace()/3.0;
  Matrix3 sigDev = sig - One*p;
  double df_dp = 2.0*p - p_c;
  Matrix3 df_ds(0.0);
  evalDevDerivOfYieldFunction(sigDev, 0.0, 0.0, df_ds);
  derivative = One*(df_dp/3.0) + df_ds;
  return;
}

// Compute df/ds  where s = deviatoric stress
//    df/ds = sqrt(3/2) df/dq s/||s|| = sqrt(3/2) 2q/M^2 n
void 
CamClayYield::evalDevDerivOfYieldFunction(const Matrix3& sigDev,
                                                   const double ,
                                                   const double ,
                                                   Matrix3& derivative)
{
  double sigDevNorm = sigDev.Norm();
  Matrix3 n = sigDev/sigDevNorm;
  double q_scaled = 3.0*sigDevNorm;
  derivative = n*(q_scaled/d_M*d_M);
  return;
}

/*! Derivative with respect to the Cauchy stress (\f$\sigma \f$) */
//   p_c = state->p_c
void 
CamClayYield::eval_df_dsigma(const Matrix3& sig,
                                      const PlasticityState* state,
                                      Matrix3& df_dsigma)
{
  evalDerivOfYieldFunction(sig, state->p_c, 0.0, df_dsigma);
  return;
}

/*! Derivative with respect to the \f$xi\f$ where \f$\xi = s \f$  
    where \f$s\f$ is deviatoric part of Cauchy stress */
void 
CamClayYield::eval_df_dxi(const Matrix3& sigDev,
                                   const PlasticityState* ,
                                   Matrix3& df_ds)
{
  evalDevDerivOfYieldFunction(sigDev, 0.0, 0.0, df_ds);
  return;
}

/* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
void 
CamClayYield::eval_df_ds_df_dbeta(const Matrix3& sigDev,
                                           const PlasticityState*,
                                           Matrix3& df_ds,
                                           Matrix3& df_dbeta)
{
  evalDevDerivOfYieldFunction(sigDev, 0.0, 0.0, df_ds);
  Matrix3 zero(0.0);
  df_dbeta = zero; 
  return;
}

/*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$) */
double 
CamClayYield::eval_df_dep(const Matrix3& ,
                                   const double& dsigy_dep,
                                   const PlasticityState* )
{
  cout << "CamClayYield: eval_df_dep not implemented yet " << endl;
  return 0.0;
}

/*! Derivative with respect to the porosity (\f$\epsilon^p \f$) */
double 
CamClayYield::eval_df_dphi(const Matrix3& ,
                                    const PlasticityState* )
{
  cout << "CamClayYield: eval_df_dphi not implemented yet " << endl;
  return 0.0;
}

/*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
double 
CamClayYield::eval_h_alpha(const Matrix3& ,
                                    const PlasticityState* )
{
  cout << "CamClayYield: eval_h_alpha not implemented yet " << endl;
  return 1.0;
}

/*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
double 
CamClayYield::eval_h_phi(const Matrix3& ,
                                  const double& ,
                                  const PlasticityState* )
{
  cout << "CamClayYield: eval_h_phi not implemented yet " << endl;
  return 0.0;
}

//--------------------------------------------------------------
// Tangent moduli
void 
CamClayYield::computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
                                                     const Matrix3& sigma, 
                                                     double sigY,
                                                     double dsigYdep,
                                                     double porosity,
                                                     double ,
                                                     TangentModulusTensor& Cep)
{
  cout << "CamClayYield: computeElasPlasTangentModulus not implemented yet " << endl;
  return;
}

void 
CamClayYield::computeTangentModulus(const TangentModulusTensor& Ce,
                                             const Matrix3& f_sigma, 
                                             double f_q1,
                                             double h_q1,
                                             TangentModulusTensor& Cep)
{
  cout << "CamClayYield: computeTangentModulus not implemented yet " << endl;
  return;
}


