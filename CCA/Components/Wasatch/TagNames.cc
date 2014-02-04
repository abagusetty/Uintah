/*
 * The MIT License
 *
 * Copyright (c) 2012-2014 The University of Utah
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

//-- Wasatch includes --//
#include "TagNames.h"

namespace Wasatch{
  
  //------------------------------------------------------------------
  
  TagNames::TagNames() :
  
  time    ( "time",     Expr::STATE_NONE ),
  timestep( "timestep", Expr::STATE_NONE ),
  stableTimestep( "StableDT", Expr::STATE_NONE ),
  
  celltype("CellType", Expr::STATE_NONE),
  
  xsvolcoord( "XSVOL", Expr::STATE_NONE ),
  ysvolcoord( "YSVOL", Expr::STATE_NONE ),
  zsvolcoord( "ZSVOL", Expr::STATE_NONE ),
  xxvolcoord( "XXVOL", Expr::STATE_NONE ),
  yxvolcoord( "YXVOL", Expr::STATE_NONE ),
  zxvolcoord( "ZXVOL", Expr::STATE_NONE ),
  xyvolcoord( "XYVOL", Expr::STATE_NONE ),
  yyvolcoord( "YYVOL", Expr::STATE_NONE ),
  zyvolcoord( "ZYVOL", Expr::STATE_NONE ),
  xzvolcoord( "XZVOL", Expr::STATE_NONE ),
  yzvolcoord( "YZVOL", Expr::STATE_NONE ),
  zzvolcoord( "ZZVOL", Expr::STATE_NONE ),
  
  // energy related variables
  temperature        ( "Temperature"       , Expr::STATE_NONE ),
  absorption         ( "AbsCoef"           , Expr::STATE_NONE ),
  radiationsource    ( "RadiationSource"   , Expr::STATE_NONE ),
  radvolq            ( "radiationVolq"     , Expr::STATE_NONE ),
  radvrflux          ( "VRFlux"            , Expr::STATE_NONE ),
  kineticEnergy      ( "KineticEnergy"     , Expr::STATE_NONE ),
  totalKineticEnergy ( "TotalKineticEnergy", Expr::STATE_NONE ),
  
  
  
  // momentum related variables
  pressure  ( "pressure",   Expr::STATE_NONE ),
  dilatation( "dilatation", Expr::STATE_NONE ),
  tauxx     (  "tau_xx",    Expr::STATE_NONE ),
  tauxy     (  "tau_xy",    Expr::STATE_NONE ),
  tauxz     (  "tau_xz",    Expr::STATE_NONE ),
  tauyx     (  "tau_yx",    Expr::STATE_NONE ),
  tauyy     (  "tau_yy",    Expr::STATE_NONE ),
  tauyz     (  "tau_yz",    Expr::STATE_NONE ),
  tauzx     (  "tau_zx",    Expr::STATE_NONE ),
  tauzy     (  "tau_zy",    Expr::STATE_NONE ),
  tauzz     (  "tau_zz",    Expr::STATE_NONE ),
  
  // turbulence related
  turbulentviscosity( "TurbulentViscosity",            Expr::STATE_NONE ),
  straintensormag   ( "StrainTensorMagnitude",         Expr::STATE_NONE ),
  vremantensormag   ( "VremanTensorMagnitude",         Expr::STATE_NONE ),
  waletensormag     ( "WaleTensorMagnitude",           Expr::STATE_NONE ),
  dynamicsmagcoef   ( "DynamicSmagorinskyCoefficient", Expr::STATE_NONE ),
  
  // predictor related variables
  star("*"),
  doubleStar("**"),
  rhs("_rhs"),
  convectiveflux("_convFlux_"),
  diffusiveflux("_diffFlux_"),
  pressuresrc( "pressure_src", Expr::STATE_NONE ),
  vardenalpha( "varden_alpha", Expr::STATE_NONE ),
  vardenbeta ( "varden_beta",  Expr::STATE_NONE ),
  divmomstar ( "divmom*",     Expr::STATE_NONE ),
  drhodtstar ( "drhodt*",     Expr::STATE_NONE ),
  drhodt     ( "drhodt",       Expr::STATE_NONE ),
  drhodtnp1  ( "drhodt",       Expr::STATE_NP1  ),
  
  // mms varden
  mms_mixfracsrc( "mms_mixture_fraction_src", Expr::STATE_NONE ),
  mms_continuitysrc("mms_continuity_src", Expr::STATE_NONE),
  mms_pressurecontsrc("mms_pressure_continuity_src", Expr::STATE_NONE),
  
  // postprocessing
  continuityresidual( "ContinuityResidual", Expr::STATE_NONE )
  
  {}
  
  //------------------------------------------------------------------
  template<>
  const Expr::Tag TagNames::make_star(Expr::Tag someTag,
                                      Expr::Context newContext) const
  {
    return Expr::Tag(someTag.name() + star, newContext);
  }

  template<>
  const Expr::Tag TagNames::make_double_star(Expr::Tag someTag,
                                             Expr::Context newContext) const
  {
    return Expr::Tag(someTag.name() + doubleStar, newContext);
  }
  
  template<>
  const Expr::Tag TagNames::make_star_rhs(Expr::Tag someTag,
                                      Expr::Context newContext) const
  {
    return Expr::Tag(someTag.name() + star + rhs, newContext);
  }

  template<>
  const Expr::Tag TagNames::make_double_star_rhs(Expr::Tag someTag,
                                             Expr::Context newContext) const
  {
    return Expr::Tag(someTag.name() + doubleStar + rhs, newContext);
  }

  template<>
  const Expr::Tag TagNames::make_star(std::string someName,
                                      Expr::Context newContext) const
  {
    return Expr::Tag(someName + star, newContext);
  }
  
  template<>
  const Expr::Tag TagNames::make_double_star(std::string someName,
                                             Expr::Context newContext) const
  {
    return Expr::Tag(someName + doubleStar, newContext);
  }
  
  template<>
  const Expr::Tag TagNames::make_star_rhs(std::string someName,
                                          Expr::Context newContext) const
  {
    return Expr::Tag(someName + star + rhs, newContext);
  }
  
  template<>
  const Expr::Tag TagNames::make_double_star_rhs(std::string someName,
                                                 Expr::Context newContext) const
  {
    return Expr::Tag(someName + doubleStar + rhs, newContext);
  }


  //------------------------------------------------------------------

  const TagNames&
  TagNames::self()
  {
    static const TagNames s;
    return s;
  }
  
  //------------------------------------------------------------------
  
} // namespace Wasatch
