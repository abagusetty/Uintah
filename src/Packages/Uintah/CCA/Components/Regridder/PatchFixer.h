#ifndef UINTAH_HOMEBREW_PATCHFIXER_H
#define UINTAH_HOMEBREW_PATCHFIXER_H

#include <Packages/Uintah/CCA/Components/Regridder/BNRTask.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <vector>
using namespace std;

namespace Uintah {

/**************************************

CLASS
   PatchFixer 
   
   This class takes a patchset and fixes it to meet Uintah's constraints
	 neighbor constraints.

GENERAL INFORMATION

   PatchFixer.h

	 Justin Luitjens
   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PatchFixer

DESCRIPTION
   This class takes a patchset and fixes it to meet Uintah's neighbor 
	 constraints by splitting patches where coarse/fine interfaces exist 
	 along faces of patches.  
  
WARNING
  
****************************************/
  class PatchFixer {
  public:
    PatchFixer(const ProcessorGroup* pg) : d_myworld(pg) 
		{
			rank=d_myworld->myrank();
			numprocs=d_myworld->size();
			comm=d_myworld->getComm();
		};
    ~PatchFixer() {};
		void FixUp(vector<PseudoPatch> &patches);	
	private:
		const ProcessorGroup *d_myworld;
		vector<int> lattice;
		vector<int> cellstolattice[3];
		vector<int> latticetocells[3];
		PseudoPatch bounds;
		IntVector csize,lsize;		
		int rank, numprocs;		
		MPI_Comm comm;
		
		void FixFace(vector<PseudoPatch> &patches,PseudoPatch patch, int dim, int side);
		void SplitPatch(int index, vector<PseudoPatch> &patches, const Split &split);
		void BuildLattice(const vector<PseudoPatch> &patches);
		void Fill(const PseudoPatch patch,const int id);
		inline int CellToLattice(int c, int d)
		{
		  return cellstolattice[d][c-bounds.low[d]];
		};
		inline void CellToLattice( int* c)
		{
		  for(int d=0;d<3;d++)
			{
			 	c[d]=CellToLattice(c[d],d);
			}
		};
		inline int LatticeToCell(int l, int d)
		{
			return latticetocells[d][l];
		};
		inline void LatticeToCell( int* l)
		{
			for(int d=0;d<3;d++)
			{
				l[d]=LatticeToCell(l[d],d);
			}
		};
	
  };

} // End namespace Uintah

#endif
