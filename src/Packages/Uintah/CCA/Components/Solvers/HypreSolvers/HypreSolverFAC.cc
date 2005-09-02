//--------------------------------------------------------------------------
// File: HypreSolverFAC.cc
// 
// Hypre CG ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverFAC.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

Priorities
HypreSolverFAC::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverFAC::initPriority~
  // Set the Hypre interfaces that FAC can work with. Only SStruct
  // is supported.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreSStruct);
  return priority;
}

void
HypreSolverFAC::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondCG::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  cerr << "HypreSolverFAC::solve() BEGIN" << "\n";
  const int numDims = 3; // Hard-coded for Uintah
  const HypreSolverParams* params = _driver->getParams();

  if (_driver->getInterface() == HypreSStruct) {
    HYPRE_SStructSolver solver;
    HypreDriverSStruct* sstructDriver =
      dynamic_cast<HypreDriverSStruct*>(_driver);
    const PatchSubset* patches = sstructDriver->getPatches();
    if (patches->size() < 1) {
      cerr << "Warning: empty list of patches for FAC solver" << "\n";
      return;
    }
    const GridP grid = patches->get(0)->getLevel()->getGrid();
    int numLevels   = grid->numLevels();

    // Set the special arrays required by FAC
    int* pLevel;                  // Part ID of each level
    hypre_Index* refinementRatio; // Refinement ratio of level to level-1.
    refinementRatio = hypre_TAlloc(hypre_Index, numLevels);
    pLevel          = hypre_TAlloc(int , numLevels);
    HYPRE_SStructMatrix facA;
     for (int level = 0; level < numLevels; level++) {
      pLevel[level] = level;      // part ID of this level
      if (level == 0) {           // Dummy value
        for (int d = 0; d < numDims; d++) {
          refinementRatio[level][d] = 1;
        }
      } else {
        for (int d = 0; d < numDims; d++) {
          refinementRatio[level][d] =
            grid->getLevel(level)->getRefinementRatio()[d];
        }
      }
    }

    // Solver setup phase:
    // Prepare FAC operator hierarchy using Galerkin coarsening
    // with Dandy-black-box interpolation, on the original meshes
    hypre_AMR_RAP(sstructDriver->getA(), refinementRatio, &facA);
    // FAC parameters
    int n_pre  = refinementRatio[numLevels-1][0]-1; // # pre-relaxation sweeps
    int n_post = refinementRatio[numLevels-1][0]-1; // #post-relaxation sweeps
    // n_pre+= n_post;
    // n_post= 0;
    HYPRE_SStructFACCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_SStructFACSetMaxLevels(solver, numLevels);
    HYPRE_SStructFACSetMaxIter(solver, params->maxIterations);
    HYPRE_SStructFACSetTol(solver, params->tolerance);
    HYPRE_SStructFACSetPLevels(solver, numLevels, pLevel);
    HYPRE_SStructFACSetPRefinements(solver, numLevels, refinementRatio);
    HYPRE_SStructFACSetRelChange(solver, 0);
    HYPRE_SStructFACSetRelaxType(solver, 2); // or 1
    HYPRE_SStructFACSetNumPreRelax(solver, n_pre);
    HYPRE_SStructFACSetNumPostRelax(solver, n_post);
    HYPRE_SStructFACSetCoarseSolverType(solver, 2);
    HYPRE_SStructFACSetLogging(solver, params->logging);
    HYPRE_SStructFACSetup2(solver, facA, sstructDriver->getB(),
                           sstructDriver->getX());
    hypre_FacZeroCData(solver, facA, sstructDriver->getB(),
                       sstructDriver->getX());

    // Call the FAC solver
    HYPRE_SStructFACSolve3(solver, facA, sstructDriver->getB(),
                           sstructDriver->getX());

    // Retrieve convergence information
    HYPRE_SStructFACGetNumIterations(solver, &_results.numIterations);
    HYPRE_SStructFACGetFinalRelativeResidualNorm(solver,
                                                 &_results.finalResNorm);
    cerr << "FAC convergence statistics:" << "\n";
    cerr << "numIterations = " << _results.numIterations << "\n";
    cerr << "finalResNorm  = " << _results.finalResNorm << "\n";

    // Destroy & free
    HYPRE_SStructFACDestroy2(solver);
    hypre_TFree(pLevel);
    hypre_TFree(refinementRatio);
    HYPRE_SStructGraph facGraph = hypre_SStructMatrixGraph(facA);
    HYPRE_SStructGraphDestroy(facGraph);
    HYPRE_SStructMatrixDestroy(facA);

  } // interface == HypreSStruct

  cerr << "HypreSolverFAC::solve() END" << "\n";
}
