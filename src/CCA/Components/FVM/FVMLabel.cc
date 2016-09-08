/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/FVM/FVMLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

using namespace Uintah;

FVMLabel::FVMLabel()
{

  ccESPotential       = VarLabel::create("cc.esPotential",
	                      CCVariable<double>::getTypeDescription());
  ccESPotentialMatrix = VarLabel::create("cc.esPotentialMatrix",
	                      CCVariable<Stencil7>::getTypeDescription());
  ccRHS_ESPotential   = VarLabel::create("cc.rhsEsPotential",
	                      CCVariable<double>::getTypeDescription());
  ccConductivity      = VarLabel::create("cc.Conductivity",
                        CCVariable<double>::getTypeDescription());
  ccGridConductivity  = VarLabel::create("cc.GridConductivity",
                          CCVariable<double>::getTypeDescription());
  fcxConductivity     = VarLabel::create("fcx.Conductivity",
                        SFCXVariable<double>::getTypeDescription());
  fcyConductivity     = VarLabel::create("fcy.Conductivity",
                        SFCYVariable<double>::getTypeDescription());
  fczConductivity     = VarLabel::create("fcz.Conductivity",
                        SFCZVariable<double>::getTypeDescription());
}

FVMLabel::~FVMLabel()
{
  VarLabel::destroy(ccESPotential);
  VarLabel::destroy(ccESPotentialMatrix);
  VarLabel::destroy(ccRHS_ESPotential);
  VarLabel::destroy(ccConductivity);
  VarLabel::destroy(ccGridConductivity);
  VarLabel::destroy(fcxConductivity);
  VarLabel::destroy(fcyConductivity);
  VarLabel::destroy(fczConductivity);
}

