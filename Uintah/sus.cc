/* REFERENCED */
//static char *id="$Id$";

/*
 *  sus.cc: Standalone Uintah Simulation - a bare-bones uintah simulation
 *          for development
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 U of U
 */

#include <Uintah/Parallel/Parallel.h>
#include <Uintah/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Uintah/Components/SimulationController/SimulationController.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Components/Schedulers/SingleProcessorScheduler.h>
#include <Uintah/Components/DataArchiver/DataArchiver.h>
#include <SCICore/Exceptions/Exception.h>
#include <ieeefp.h>

#include <iostream>
#include <string>
#include <vector>

using SCICore::Exceptions::Exception;
using namespace std;
using namespace Uintah;

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options]\n\n";
    cerr << "Valid options are:\n";
    cerr << "NOT FINISHED\n";
    exit(1);
}

int main(int argc, char** argv)
{
    fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);

    /*
     * Default values
     */
    bool do_mpm=false;
    bool do_arches=false;
    bool do_ice=false;
    bool numThreads = 0;
    string filename;

    /*
     * Parse arguments
     */
    for(int i=1;i<argc;i++){
	string s=argv[i];
	if(s == "-mpm"){
	    do_mpm=true;
	} else if(s == "-arches"){
	    do_arches=true;
	} else if(s == "-ice"){
	    do_ice=true;
	} else if(s == "-nthreads"){
	    if(++i == argc){
		cerr << "You must provide a number of threads for -nthreads\n";
		usage(s, argv[0]);
	    }
	    numThreads = atoi(argv[i]);
	} else {
	    if(filename!="")
		usage(s, argv[0]);
	    else
		filename = argv[i];
	}
    }

    if(filename == ""){
	cerr << "No input file specified\n";
	usage("", argv[0]);
    }

    /*
     * Check for valid argument combinations
     */
    if(do_mpm && (do_ice || do_arches)){
	cerr << "MPM doesn't yet work with ICE/Arches\n";
	usage("", argv[0]);
    }
    if(do_ice && do_arches){
	cerr << "ICE and Arches do not work together\n";
	usage("", argv[0]);
    }
    if(do_ice && numThreads>0){
	cerr << "ICE doesn't support threads yet\n";
	usage("", argv[0]);
    }
    if(do_arches && numThreads>0){
	cerr << "Arches doesn't do threads yet\n";
	usage("", argv[0]);
    }

    if(!(do_ice || do_arches || do_mpm)){
	cerr << "You need to specify -arches, -ice, or -mpm\n";
	usage("", argv[0]);
    }

    /*
     * Initialize MPI
     */
    Parallel::initializeManager(argc, argv);

    int MpiRank = Parallel::getRank();
    int MpiProcesses = Parallel::getSize();

    /*
     * Create the components
     */
    try {
	SimulationController* sim = scinew SimulationController( MpiRank,
							      MpiProcesses );

	// Reader
	ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
	sim->attachPort("problem spec", reader);

	// Connect a MPM module if applicable
	if(do_mpm){
	    MPMInterface* mpm;
	    if(numThreads == 0){
		mpm = scinew MPM::SerialMPM( MpiRank, MpiProcesses );
	    } else {
#ifdef WONT_COMPILE_YET
		mpm = scinew ThreadedMPM();
#else
		mpm = 0;
#endif
	    }
	    sim->attachPort("mpm", mpm);
	}

	// Connect a CFD module if applicable
	CFDInterface* cfd = 0;
	if(do_arches){
	    cfd = scinew ArchesSpace::Arches( MpiRank, MpiProcesses );
	}
	if(do_ice){
	    cfd = scinew ICESpace::ICE();
	}
	if(cfd)
	    sim->attachPort("cfd", cfd);

	// Output
	Output* output = scinew DataArchiver(MpiRank, MpiProcesses );
	sim->attachPort("output", output);

	// Scheduler
	SingleProcessorScheduler* sched = 
	   scinew SingleProcessorScheduler( MpiRank, MpiProcesses );
	sim->attachPort("scheduler", sched);

	/*
	 * Start the simulation controller
	 */
	sim->run();
    } catch (Exception& e) {
	cerr << "Caught exception: " << e.message() << '\n';
	abort();
    } catch(...){
	cerr << "Caught unknown exception\n";
	abort();
    }

    /*
     * Finalize MPI
     */
    Parallel::finalizeManager();
}

//
// $Log$
// Revision 1.13  2000/06/15 23:14:04  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.12  2000/06/15 21:56:56  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.11  2000/05/30 20:18:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.10  2000/05/15 19:39:29  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.9  2000/05/09 22:58:34  sparker
// Changed namespace names
//
// Revision 1.8  2000/04/26 06:47:56  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/19 22:43:51  dav
// more mpi stuff
//
// Revision 1.6  2000/04/19 21:19:59  dav
// more MPI stuff
//
// Revision 1.5  2000/04/13 06:50:49  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/11 07:10:29  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/20 17:17:03  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.2  2000/03/17 21:01:02  dav
// namespace mods
//
// Revision 1.1  2000/02/27 07:48:34  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
//
