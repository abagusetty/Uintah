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

#include "BorjaShear.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <cmath>
//#include <iostream>

using namespace Uintah;
using namespace std;

// Construct a shear modulus model.  
BorjaShear::BorjaShear(ProblemSpecP& ps )
{
  ps->require("mu0",d_mu0);
  ps->require("alpha",d_alpha);
  ps->require("p0",d_p0);
  ps->require("epse_v0",d_epse_v0);
  ps->require("kappatilde",d_kappatilde);
}

// Construct a copy of a shear modulus model.  
BorjaShear::BorjaShear(const BorjaShear* smm)
{
  d_mu0 = smm->d_mu0;
  d_alpha = smm->d_alpha;
  d_p0 = smm->d_p0;
  d_epse_v0 = smm->d_epse_v0;
  d_kappatilde = smm->d_kappatilde;
}

// Destructor of shear modulus model.  
BorjaShear::~BorjaShear()
{
}

void BorjaShear::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("elastic_shear_modulus_model");
  shear_ps->setAttribute("type","borja_shear_modulus");

  shear_ps->appendElement("mu0",d_mu0);
  shear_ps->appendElement("alpha",d_alpha);
  shear_ps->appendElement("p0",d_p0);
  shear_ps->appendElement("epse_v0",d_epse_v0);
  shear_ps->appendElement("kappatilde",d_kappatilde);
}

         
// Compute the shear modulus
double 
BorjaShear::computeInitialShearModulus()
{
  double mu_vol = evalShearModulus(0.0);
  return (d_mu0 - mu_vol);
}

double 
BorjaShear::computeShearModulus(const PlasticityState* state) 
{
  double mu_vol = evalShearModulus(state->epse_v);
  return (d_mu0 - mu_vol);
}

double 
BorjaShear::computeShearModulus(const PlasticityState* state) const
{
  double mu_vol = evalShearModulus(state->epse_v);
  return (d_mu0 - mu_vol);
}

// Compute the shear strain energy
// W = 3/2 mu epse_s^2
double
BorjaShear::computeStrainEnergy(const PlasticityState* state)
{
  double mu_vol = evalShearModulus(state->epse_v);
  double W = 1.5*(d_mu0 - mu_vol)*(state->epse_s*state->epse_s);
  return W;
}

/* Compute q = 3 mu epse_s
         where mu = shear modulus
               epse_s = sqrt{2/3} ||ee||
               ee = deviatoric part of elastic strain = epse - 1/3 epse_v I
               epse = total elastic strain
               epse_v = tr(epse) */
double 
BorjaShear::computeQ(const PlasticityState* state) const
{
  return evalQ(state->epse_v, state->epse_s);
}

/* Compute dq/depse_s */
double 
BorjaShear::computeDqDepse_s(const PlasticityState* state) const
{
  return evalDqDepse_s(state->epse_v, state->epse_s);
}

/* Compute dq/depse_v */
double 
BorjaShear::computeDqDepse_v(const PlasticityState* state) const
{
  return evalDqDepse_v(state->epse_v, state->epse_s);
}

// Private methods below:

//  Shear modulus computation (only pressure contribution)
double 
BorjaShear::evalShearModulus(const double& epse_v) const
{
  double mu_vol = d_alpha*d_p0*exp(-(epse_v - d_epse_v0)/d_kappatilde);
  return mu_vol;
}

//  Shear stress magnitude computation
double 
BorjaShear::evalQ(const double& epse_v, const double& epse_s) const
{
  double mu_vol = evalShearModulus(epse_v);
  double q = 3.0*(d_mu0 - mu_vol)*epse_s;

  return q;
}

//  volumetric derivative computation
double 
BorjaShear::evalDqDepse_v(const double& epse_v, const double& epse_s) const
{
  double mu_vol = evalShearModulus(epse_v);
  double dmu_depse_v = mu_vol/d_kappatilde;
  double dq_depse_v = 3.0*dmu_depse_v*epse_s;
  return dq_depse_v;
}

//  deviatoric derivative computation
double 
BorjaShear::evalDqDepse_s(const double& epse_v, const double& epse_s) const
{
  double mu_vol = evalShearModulus(epse_v);
  double dq_depse_s = 3.0*(d_mu0 - mu_vol);
  return dq_depse_s;
}

