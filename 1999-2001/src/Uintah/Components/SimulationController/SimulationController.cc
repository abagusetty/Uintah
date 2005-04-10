//
// $Id$
//

#include <Uintah/Components/SimulationController/SimulationController.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/OS/Dir.h>
#include <Uintah/Exceptions/ProblemSetupException.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/SimulationTime.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/MPMCFDInterface.h>
#include <Uintah/Interface/MDInterface.h>
#include <Uintah/Interface/Output.h>
#include <Uintah/Interface/Analyze.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Grid/VarTypes.h>
#include <iostream>
#include <values.h>

#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
#include <list>
#include <fstream>
#include <math.h>
#endif

#include <SCICore/Malloc/Allocator.h> // for memory leak tests...

using std::cerr;
using std::cout;

using SCICore::Geometry::IntVector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Math::Abs;
using SCICore::Geometry::Min;
using SCICore::Thread::Time;
using namespace Uintah;
using SCICore::Containers::Array3;

SimulationController::SimulationController(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
   d_restarting = false;
}

SimulationController::~SimulationController()
{
}

void SimulationController::doRestart(std::string restartFromDir, int timestep,
				     bool removeOldDir)
{
   d_restarting = true;
   d_restartFromDir = restartFromDir;
   d_restartTimestep = timestep;
   d_restartRemoveOldDir = removeOldDir;
}


#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
double stdDeviation(list<double>& vals, double& mean)
{
  if (vals.size() < 2)
    return -1;

  list<double>::iterator it;

  mean = 0;
  double variance = 0;
  for (it = vals.begin(); it != vals.end(); it++)
    mean += *it;
  mean /= vals.size();

  for (it = vals.begin(); it != vals.end(); it++)
    variance += pow(*it - mean, 2);
  variance /= (vals.size() - 1);
  return sqrt(variance);
}
#endif

void SimulationController::run()
{
   UintahParallelPort* pp = getPort("problem spec");
   ProblemSpecInterface* psi = dynamic_cast<ProblemSpecInterface*>(pp);
   
   // Get the problem specification
   ProblemSpecP ups = psi->readInputFile();
   ups->writeMessages(d_myworld->myrank() == 0);
   if(!ups)
      throw ProblemSetupException("Cannot read problem specification");
   
   releasePort("problem spec");
   
   if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification");
   
   Output* output = dynamic_cast<Output*>(getPort("output"));
   output->problemSetup(ups);
   
   // Setup the initial grid
   GridP grid=scinew Grid();

   problemSetup(ups, grid);
   
   if(grid->numLevels() == 0){
      cerr << "No problem specified.  Exiting SimulationController.\n";
      return;
   }
   
   // Check the grid
   grid->performConsistencyCheck();
   // Print out meta data
   if (d_myworld->myrank() == 0)
     grid->printStatistics();

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   // Initialize the CFD and/or MPM components
   CFDInterface* cfd       = dynamic_cast<CFDInterface*>(getPort("cfd"));
   MPMInterface* mpm       = dynamic_cast<MPMInterface*>(getPort("mpm"));
   MPMCFDInterface* mpmcfd = dynamic_cast<MPMCFDInterface*>(getPort("mpmcfd"));
   if(cfd && !mpmcfd)
      cfd->problemSetup(ups, grid, sharedState);
   
   if(mpm && !mpmcfd)
      mpm->problemSetup(ups, grid, sharedState);

   if(mpmcfd)
      mpmcfd->problemSetup(ups, grid, sharedState);

   // Initialize the MD components --tan
   MDInterface* md = dynamic_cast<MDInterface*>(getPort("md"));
   if(md)
      md->problemSetup(ups, grid, sharedState);
   
   Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
   sched->problemSetup(ups);
   SchedulerP scheduler(sched);
   DataWarehouseP null_dw = 0;
   DataWarehouseP old_dw = scheduler->createDataWarehouse(null_dw);

   old_dw->setGrid(grid);
   
   scheduler->initialize();

   double t;

   // Parse time struct
   SimulationTime timeinfo(ups);

   if(d_restarting){
      // create a temporary DataArchive for reading in the checkpoints
      // archive for restarting.
      Dir restartFromDir(d_restartFromDir);
      Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
      DataArchive archive(checkpointRestartDir.getName(),
			  d_myworld->myrank(), d_myworld->size());
      
      archive.restartInitialize(d_restartTimestep, grid, old_dw, &t);
      
      output->restartSetup(restartFromDir, d_restartTimestep, t,
			   d_restartRemoveOldDir);
   } else {
      // Initialize the CFD and/or MPM data
      for(int i=0;i<grid->numLevels();i++){
	 LevelP level = grid->getLevel(i);
	 scheduleInitialize(level, scheduler, old_dw, cfd, mpm, mpmcfd, md);
      }
   }
   
   // For AMR, this will need to change
   if(grid->numLevels() != 1)
      throw ProblemSetupException("AMR problem specified; cannot do it yet");
   LevelP level = grid->getLevel(0);

   /*
   // Parse time struct
   SimulationTime timeinfo(ups);
   */
   
   double start_time = Time::currentSeconds();
   if (!d_restarting)
      t = timeinfo.initTime;
   
   scheduleComputeStableTimestep(level,scheduler, old_dw, cfd, mpm, mpmcfd, md);
   
   Analyze* analyze = dynamic_cast<Analyze*>(getPort("analyze"));
   if(analyze)
      analyze->problemSetup(ups, grid, sharedState);
   
   if(output)
      output->finalizeTimestep(t, 0, level, scheduler, old_dw);

   scheduler->execute(d_myworld, old_dw, old_dw);

#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
   int n = 0;
   list<double> wallTimes;
   double prevWallTime;
#endif

   while(t < timeinfo.maxTime) {

      double wallTime = Time::currentSeconds() - start_time;

      delt_vartype delt_var;
      old_dw->get(delt_var, sharedState->get_delt_label());

      double delt = delt_var;
      delt *= timeinfo.delt_factor;

      if(delt < timeinfo.delt_min){
	 if(d_myworld->myrank() == 0)
	    cerr << "WARNING: raising delt from " << delt
		 << " to minimum: " << timeinfo.delt_min << '\n';
	 delt = timeinfo.delt_min;
      }
      if(delt > timeinfo.delt_max){
	 if(d_myworld->myrank() == 0)
	    cerr << "WARNING: lowering delt from " << delt 
		 << " to maxmimum: " << timeinfo.delt_max << '\n';
	 delt = timeinfo.delt_max;
      }
      old_dw->override(delt_vartype(delt), sharedState->get_delt_label());
      if(d_myworld->myrank() == 0){

	size_t nalloc,  sizealloc, nfree,  sizefree, nfillbin,
	  nmmap, sizemmap, nmunmap, sizemunmap, highwater_alloc,  
	  highwater_mmap, nlonglocks, nnaps, bytes_overhead, bytes_free,
	  bytes_fragmented, bytes_inuse, bytes_inhunks;

	SCICore::Malloc::GetGlobalStats(SCICore::Malloc::DefaultAllocator(),
					nalloc, sizealloc, nfree, sizefree,
					nfillbin, nmmap, sizemmap, nmunmap,
					sizemunmap, highwater_alloc,
					highwater_mmap, nlonglocks, nnaps,
					bytes_overhead, bytes_free,
					bytes_fragmented, bytes_inuse,
					bytes_inhunks);

	if( analyze ) analyze->showStepInformation();
	else {
          cout << "Time=" << t << ", delT=" << delt 
	       << ", elap T = " << wallTime 
	       << ", DW: " << old_dw->getID() << ", Mem Use = " 
	       << sizealloc - sizefree << "\n";

#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
	  if (n > 1) // ignore first set of elapsed times
	    wallTimes.push_back(wallTime - prevWallTime);

	  if (wallTimes.size() > 1) {
	    double stdDev, mean;
	    stdDev = stdDeviation(wallTimes, mean);
	    ofstream timefile("avg_elapsed_walltime.txt");
	    timefile << mean << " +- " << stdDev << endl;
	  }
	  prevWallTime = wallTime;
	  n++;
#endif
	}
      }

      scheduler->initialize();

      /* I THINK THIS SHOULD BE null_dw, NOT old_dw... Dd: */
      DataWarehouseP new_dw = scheduler->createDataWarehouse(/*old_dw*/null_dw);
      //DataWarehouseP new_dw = scheduler->createDataWarehouse(old_dw);

      scheduleTimeAdvance(t, delt, level, scheduler, old_dw, new_dw,
			  cfd, mpm, mpmcfd, md);

      //data analyze in each step
      if(analyze) {
        analyze->performAnalyze(t, delt, level, scheduler, old_dw, new_dw);
      }
      
      t += delt;
      if(output)
	 output->finalizeTimestep(t, delt, level, scheduler, new_dw);
      
      // Begin next time step...
      scheduleComputeStableTimestep(level, scheduler, new_dw, cfd, mpm, mpmcfd,
									 md);
      scheduler->execute(d_myworld, old_dw, new_dw);

      old_dw = new_dw;
   }
}

void SimulationController::problemSetup(const ProblemSpecP& params,
					GridP& grid)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;
   
   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the patches.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.
      Point anchor(MAXDOUBLE, MAXDOUBLE, MAXDOUBLE);
      Vector spacing;
      bool have_levelspacing=false;
      if(level_ps->get("spacing", spacing))
	 have_levelspacing=true;
      bool have_patchspacing=false;
	 
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	  box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	 Point lower;
	 box_ps->require("lower", lower);
	 Point upper;
	 box_ps->require("upper", upper);
	 anchor=SCICore::Geometry::Min(lower, anchor);

	 IntVector resolution;
	 if(box_ps->get("resolution", resolution)){
	    if(have_levelspacing){
	       throw ProblemSetupException("Cannot specify level spacing and patch resolution");
	    } else {
	       Vector newspacing = (upper-lower)/resolution;
	       if(have_patchspacing){
		  Vector diff = spacing-newspacing;
		  if(diff.length() > 1.e-6)
		     throw ProblemSetupException("Using patch resolution, and the patch spacings are inconsistent");
	       } else {
		  spacing = newspacing;
	       }
	       have_patchspacing=true;
	    }
	 }
      }
	 
      if(!have_levelspacing && !have_patchspacing)
	 throw ProblemSetupException("Box resolution is not specified");
	 
      LevelP level = scinew Level(grid.get_rep(), anchor, spacing);
      
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	  box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	Point lower;
	box_ps->require("lower", lower);
	Point upper;
	box_ps->require("upper", upper);
	
	IntVector lowCell = level->getCellIndex(lower);
	IntVector highCell = level->getCellIndex(upper+Vector(1.e-6,1.e-6,1.e-6));
	Point lower2 = level->getNodePosition(lowCell);
	Point upper2 = level->getNodePosition(highCell);
	double diff_lower = (lower2-lower).length();
	if(diff_lower > 1.e-6)
	  throw ProblemSetupException("Box lower corner does not coincide with grid");
	double diff_upper = (upper2-upper).length();
	if(diff_upper > 1.e-6){
	  cerr << "upper=" << upper << '\n';
	  cerr << "lowCell =" << lowCell << '\n';
	  cerr << "highCell =" << highCell << '\n';
	  cerr << "upper2=" << upper2 << '\n';
	  cerr << "diff=" << diff_upper << '\n';
	  throw ProblemSetupException("Box upper corner does not coincide with grid");
	}
	// Determine the interior cell limits.  For no extraCells, the limits
	// will be the same.  For extraCells, the interior cells will have
	// different limits so that we can develop a CellIterator that will
	// use only the interior cells instead of including the extraCell
	// limits.
	IntVector inLowCell = lowCell;
	IntVector inHighCell = highCell;
	
	IntVector extraCells;
	if(box_ps->get("extraCells", extraCells)){
	  lowCell = lowCell-extraCells;
	  highCell = highCell+extraCells;
	}
	
	IntVector resolution(highCell-lowCell);
	IntVector inResolution(inHighCell-inLowCell);
	if(resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1)
	  throw ProblemSetupException("Degeneration patch");
	
	IntVector patches;
	IntVector inLowIndex,inHighIndex;
	if(box_ps->get("patches", patches)){
	   level->setPatchDistributionHint(patches);
	  for(int i=0;i<patches.x();i++){
	    for(int j=0;j<patches.y();j++){
	      for(int k=0;k<patches.z();k++){
		IntVector startcell = resolution*IntVector(i,j,k)/patches;
		IntVector inStartCell=inResolution*IntVector(i,j,k)/patches;
		IntVector endcell = resolution*IntVector(i+1,j+1,k+1)/patches;
		IntVector inEndCell=inResolution*IntVector(i+1,j+1,k+1)/patches;
		level->addPatch(startcell+lowCell, endcell+lowCell,
				inStartCell+inLowCell,inEndCell+inLowCell);
	      }
	    }
	  }
	} else {
	  level->addPatch(lowCell, highCell,inLowCell,inHighCell);
	}
      }
      level->finalizeLevel();
      level->assignBCS(grid_ps);
      grid->addLevel(level);
   }
}

void SimulationController::scheduleInitialize(LevelP& level,
					      SchedulerP& sched,
					      DataWarehouseP& new_dw,
					      CFDInterface* cfd,
					      MPMInterface* mpm,
					      MPMCFDInterface* mpmcfd,
					      MDInterface* md)
{
  if(mpmcfd){
    mpmcfd->scheduleInitialize(level, sched, new_dw);
  }
  else {
    if(cfd) {
      cfd->scheduleInitialize(level, sched, new_dw);
    }
    if(mpm) {
      mpm->scheduleInitialize(level, sched, new_dw);
    }
  }
  if(md) {
    md->scheduleInitialize(level, sched, new_dw);
  }
}

void SimulationController::scheduleComputeStableTimestep(LevelP& level,
							SchedulerP& sched,
							DataWarehouseP& new_dw,
							CFDInterface* cfd,
							MPMInterface* mpm,
							MPMCFDInterface* mpmcfd,
							MDInterface* md)
{
  if(mpmcfd){
    mpmcfd->scheduleComputeStableTimestep(level, sched, new_dw);
  }
  else {
     if(cfd)
        cfd->scheduleComputeStableTimestep(level, sched, new_dw);
     if(mpm)
        mpm->scheduleComputeStableTimestep(level, sched, new_dw);
   }
   if(md)
      md->scheduleComputeStableTimestep(level, sched, new_dw);
}

void SimulationController::scheduleTimeAdvance(double t, double delt,
					       LevelP& level,
					       SchedulerP& sched,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw,
					       CFDInterface* cfd,
					       MPMInterface* mpm,
					       MPMCFDInterface* mpmcfd,
					       MDInterface* md)
{
   // Temporary - when cfd/mpm are coupled this will need help
  if(mpmcfd){
      mpmcfd->scheduleTimeAdvance(t, delt, level, sched, old_dw, new_dw);
  }
  else {
   if(cfd)
      cfd->scheduleTimeAdvance(t, delt, level, sched, old_dw, new_dw);
   if(mpm)
      mpm->scheduleTimeAdvance(t, delt, level, sched, old_dw, new_dw);
  }
      
   // Added molecular dynamics module, currently it will not be coupled with 
   // cfd/mpm.  --tan
   if(md)
      md->scheduleTimeAdvance(t, delt, level, sched, old_dw, new_dw);
   
#if 0
   
   /* If we aren't doing any chemistry, skip this step */
#if 0
   if(chem)
      chem->calculateChemistryEffects();
#endif
   
   /* If we aren't doing MPM, skip this step */
   if(mpm){
#if 0
      mpm->zeroMPMGridData();
      mpm->interpolateParticlesToGrid(/* consume */ p_mass, p_velocity,
				      p_extForce, p_temperature,
				      /* produce */ g_mass, g_velocity, g_exForce,
				      g_volFraction, g_temperature);
#endif
   }
   if(mpm && !cfd){  // In other words, doing MPM only
#if 0
      mpm->exchangeMomentum2();
      mpm->computeVelocityGradients(/* arguments left as an exercise */);
      mpm->computeStressTensor();
      mpm->computeInternalForces(/* arguments left as an exercise */);
#endif
   }
   
   /* If we aren't doing CFD, sking this step */
   if(cfd && !mpm){
#if 0
      cfd->pressureAndVelocitySolve(/* consume */ g_density, g_massFraction,
				    g_temperature,
				    maybe other stuff,
				    
				    /* produce */ g_velocity, g_pressure);
#endif
   }
   
   if(mpm && cfd){
#if 0
      coupling->pressureVelocityStressSolve();
      /* produce */ cell centered pressure,
		       face centered velocity,
		       particle stresses
		       mpm->computeInternalForces();
#endif
   }
   
   if(mpm){
#if 0
      mpm->solveEquationsOfMotion(/* consume */ g_deltaPress, p_stress,
				  some boundary conditions,
				  /* produce */ p_acceleration);
      mpm->integrateAceleration(/* consume */ p_acceleration);
#endif
   }
   if(cfd){
#if 0
      /* This may be different, or a no-op for arches. - Rajesh? */
      cfd->addLagragianEffects(...);
#endif
   }
   /* Is this CFD or MPM, or what? */
   /* It's "or what" hence I prefer using the coupling module so
      neither of the others have to know about it.               */
   if(mpm && cfd){       // Do coupling's version of Exchange
#if 0
      coupling->calculateMomentumAndEnergyExchange( ... );
#endif
   }
   else if(mpm && !cfd){ // Do mpm's version of Exchange
#if 0
      mpm->exchangeMomentum();
#endif
   }
   else if(cfd && !mpm){ // Do cfd's version of Exchange
#if 0
      cfd->momentumExchange();
#endif
   }
   
   if(cfd){
#if 0
      cfd->advect(...);
      cfd->advanceInTime(...);
#endif
   }
   if(mpm){
#if 0
      mpm->interpolateGridToParticles(...);
      mpm->updateParticleVelocityAndPosition(...);
#endif
   }
#endif
}

//
// $Log$
// Revision 1.52  2001/01/06 02:41:16  witzel
// Added checkpoint/restart capabilities
//
// Revision 1.51  2000/12/10 09:06:15  sparker
// Merge from csafe_risky1
//
// Revision 1.50  2000/12/01 23:01:46  guilkey
// Adding stuff for coupled MPM and CFD.
//
// Revision 1.49  2000/11/14 04:05:52  jas
// Now storing patches with two indices: extraCell and interior cell limits.
//
// Revision 1.48  2000/11/03 19:02:36  witzel
// Added code to output average ellapsed wall clock times between
// time steps to a file "avg_elapsed_walltime.txt" if the
// OUTPUT_AVG_ELAPSED_WALLTIME macro is defined.
//
// Revision 1.47.4.5  2000/11/02 00:10:08  witzel
// Don't output date/time for OUTPUT_AVG_ELAPSED_WALLTIME -- not needed
//
// Revision 1.47.4.4  2000/11/01 21:25:48  witzel
// changed OUTPUT_AVE_ELLAPSED_WALLTIME to OUTPUT_AVG_ELAPSED_WALLTIME
//
// Revision 1.47.4.3  2000/11/01 02:03:27  witzel
// Added code to output average ellapsed wall clock times between
// time steps to a file "ave_ellapsed_walltime.txt" if the
// OUTPUT_AVE_ELLAPSED_WALLTIME macro is defined.
//
// Revision 1.47.4.2  2000/10/10 05:28:06  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.47.4.1  2000/10/07 06:12:14  sparker
// Try to fix rounding errors for cell upper index
// set a hint for the level's number of patches in each direction
//
// Revision 1.47  2000/09/26 21:26:36  witzel
// Make only process zero call printStatistics() and write messages in
// the ProblemSetup.
//
// Revision 1.46  2000/09/25 20:44:08  sparker
// Quiet g++ warnings
//
// Revision 1.45  2000/09/20 16:05:24  sparker
// Call problemSetup on scheduler
//
// Revision 1.44  2000/09/04 23:21:10  tan
// Control the information showing at each step in SimulationController by
// Analyze module.
//
// Revision 1.43  2000/09/04 00:38:32  tan
// Modified Analyze interface for scientific debugging under both
// sigle processor and mpi environment.
//
// Revision 1.42  2000/08/24 22:17:15  dav
// Modified output messages
//
// Revision 1.41  2000/08/24 21:21:31  dav
// Removed DWMpiHandler stuff
//
// Revision 1.40  2000/07/28 07:37:50  bbanerje
// Rajesh must have missed these .. adding the changed version of
// createDataWarehouse calls
//
// Revision 1.39  2000/07/27 22:39:48  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.38  2000/07/17 23:36:31  tan
// Added Analyze interface that will be especially useful for debugging
// on scitific results.
//
// Revision 1.37  2000/06/23 19:28:32  jas
// Added the reading of grid bcs right after we finalize a level.
//
// Revision 1.36  2000/06/17 07:06:42  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.35  2000/06/16 19:47:52  sparker
// Changed _ds to _dw for data warehouse variables
// Use new output interface
// Eliminated dw->carryForward
// Fixed off by 1 error in output (bugzilla #139)
//
// Revision 1.34  2000/06/16 05:03:09  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.33  2000/06/15 23:14:09  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.32  2000/06/15 21:57:13  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.31  2000/06/09 23:36:33  bard
// Removed cosmetic implementation of fudge factor (time step multiplier).
// It needs to be put in the constitutive routines to have the correct effect.
//
// Revision 1.30  2000/06/09 17:20:12  tan
// Added MD(molecular dynamics) module to SimulationController.
//
// Revision 1.29  2000/06/08 21:00:40  jas
// Added timestep multiplier (fudge factor).
//
// Revision 1.28  2000/06/05 19:51:56  guilkey
// Formatting
//
// Revision 1.27  2000/05/31 23:44:55  rawat
// modified arches and properties
//
// Revision 1.26  2000/05/30 20:19:25  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.25  2000/05/30 19:43:28  guilkey
// Unfixed the maximum timestep size fix, which wasn't much of a
// fix at all...
//
// Revision 1.24  2000/05/30 17:09:54  dav
// MPI stuff
//
// Revision 1.23  2000/05/26 18:58:15  guilkey
// Uncommented code to allow a maximum timestep to be set and effectively
// used.  The minimum time step still doesn't work, but for an explicit
// code, who cares?
//
// Revision 1.22  2000/05/20 08:09:18  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.21  2000/05/18 18:49:16  jas
// SimulationState constructor now uses the input file.
//
// Revision 1.20  2000/05/15 19:39:45  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.19  2000/05/11 21:25:51  sparker
// Fixed main time loop
//
// Revision 1.18  2000/05/11 20:10:20  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.17  2000/05/10 20:02:55  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.16  2000/05/07 06:02:10  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.15  2000/05/05 06:42:44  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.14  2000/05/04 18:38:32  jas
// Now the max_dt is used in the simulation if it is smaller than the
// stable dt computed in the constitutive model.
//
// Revision 1.13  2000/05/02 17:54:30  sparker
// Implemented more of SerialMPM
//
// Revision 1.12  2000/05/02 06:07:18  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.11  2000/04/26 06:48:36  sparker
// Streamlined namespaces
//
// Revision 1.10  2000/04/20 18:56:28  sparker
// Updates to MPM
//
// Revision 1.9  2000/04/19 20:59:25  dav
// adding MPI support
//
// Revision 1.8  2000/04/19 05:26:12  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.7  2000/04/13 20:05:56  sparker
// Compile more of arches
// Made SimulationController work somewhat
//
// Revision 1.6  2000/04/13 06:50:59  sparker
// More implementation to get this to work
//
// Revision 1.5  2000/04/12 23:00:09  sparker
// Start of reading grids
//
// Revision 1.4  2000/04/11 07:10:42  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 20:58:31  dav
// namespace updates
//
//
