//----- ScalarSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ScalarSolver_h
#define Uintah_Component_Arches_ScalarSolver_h

/**************************************
CLASS
   ScalarSolver
   
   Class ScalarSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   ScalarSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ScalarSolver linearizes and solves scalar
   equation on a grid hierarchy


WARNING
   none

************************************************************************/

#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/CCVariable.h>

namespace Uintah {
namespace ArchesSpace {

class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class ScalarSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of the Scalar solver.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      //
      ScalarSolver(TurbulenceModel* turb_model, 
		   BoundaryCondition* bndry_cond,
		   PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~ScalarSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule Solve of linearized scalar equation
      //
      void solve(const LevelP& level,
		 SchedulerP& sched,
		 DataWarehouseP& old_dw,
		 DataWarehouseP& new_dw,
		 double time, double delta_t, int index);
   
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule Build of linearized matrix
      //
      void sched_buildLinearMatrix(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t, int index);
protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      //
      // Default : Construct an empty instance of the Pressure solver.
      //
      ScalarSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here
      //
      void buildLinearMatrix(const ProcessorContext* pc,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw,
			     double delta_t, const int index);

private:

      // computes coefficients
      Discretization* d_discretize;
      // computes sources
      Source* d_source;
      // linear solver
      LinearSolver* d_linearSolver;
      // turbulence model
      TurbulenceModel* d_turbModel;
      // boundary condition
      BoundaryCondition* d_boundaryCondition;
      // physical constants
      PhysicalConstants* d_physicalConsts;

      // const VarLabel* (required)
      const VarLabel* d_scalarLabel;
      const VarLabel* d_uVelocityLabel;
      const VarLabel* d_vVelocityLabel;
      const VarLabel* d_wVelocityLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;

      // const VarLabel* (computed)
      const VarLabel* d_scalarCoefLabel;
      const VarLabel* d_scalarLinSrcLabel;
      const VarLabel* d_scalarNonLinSrcLabel;

      // DataWarehouse generation
      int d_generation;

}; // End class ScalarSolver

} // End namespace ArchesSpace
} // End namespace Uintah

#endif

//
// $Log$
// Revision 1.7  2000/06/12 21:30:00  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.6  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.5  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
