#include "SolverAMG.h"

#include "util.h"
#include "Level.h"
#include "Patch.h"

#include <string>
#include <map>

using namespace std;

void
SolverAMG::initializeData(const Hierarchy& hier,
                       const HYPRE_SStructGrid& grid,
                       const HYPRE_SStructGraph& graph)
{
  initializeData(hier, grid, graph);   // Generic Solver A,b,x initialization
}

void
SolverAMG::setup(void)
{
  /*-----------------------------------------------------------
   * Solver setup phase
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("AMG setup phase\n");
  Proc0Print("----------------------------------------------------\n");
  int time_index;
  time_index = hypre_InitializeTiming("AMG Setup");
  hypre_BeginTiming(time_index);
  HYPRE_BoomerAMGCreate(&_parSolver);
  HYPRE_BoomerAMGSetCoarsenType(_parSolver, 6);
  HYPRE_BoomerAMGSetStrongThreshold(_parSolver, 0.);
  HYPRE_BoomerAMGSetTruncFactor(_parSolver, 0.3);
  /*HYPRE_BoomerAMGSetMaxLevels(_parSolver, 4);*/
  HYPRE_BoomerAMGSetTol(_parSolver, 1.0e-06);
  HYPRE_BoomerAMGSetPrintLevel(_parSolver, 1);
  HYPRE_BoomerAMGSetPrintFileName(_parSolver, "sstruct.out.log");
  HYPRE_BoomerAMGSetMaxIter(_parSolver, 200);
  HYPRE_BoomerAMGSetup(_parSolver, _parA, _parB, _parX);
  hypre_EndTiming(time_index);
  hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
}

void
SolverAMG::solve(void)
{
  /*_____________________________________________________________________
    Function solveLinearSystem:
    Solve the linear system A*x = b using AMG. Includes parameter setup
    for the AMG solver phase.
    _____________________________________________________________________*/
  /* Sparse matrix data structures for various solvers (FAC, ParCSR),
     right-hand-side b and solution x */
  int                   time_index;  // Hypre Timer

  time_index = hypre_InitializeTiming("BoomerAMG Solve");
  hypre_BeginTiming(time_index);
  
  HYPRE_BoomerAMGSolve(_parSolver, _parA, _parB, _parX); // call AMG solver
  
  hypre_EndTiming(time_index);
  hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();

  /* Read results into Solver::_results */
  int numIterations = -1;
  HYPRE_BoomerAMGGetNumIterations(_parSolver, &numIterations);
  _results.numIterations = numIterations;
  HYPRE_BoomerAMGGetFinalRelativeResidualNorm(_parSolver,
                                              &_results.finalResNorm);
  
  HYPRE_BoomerAMGDestroy(_parSolver);

  /*-----------------------------------------------------------
   * Gather the solution vector
   *-----------------------------------------------------------*/
  Proc0Print("----------------------------------------------------\n");
  Proc0Print("Gather the solution vector\n");
  Proc0Print("----------------------------------------------------\n");

  HYPRE_SStructVectorGather(_x);
} //end solve()
