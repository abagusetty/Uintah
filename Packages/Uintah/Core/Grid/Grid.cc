
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <sci_values.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

Grid::Grid()
{
}

Grid::~Grid()
{
}

int Grid::numLevels() const
{
  return (int)d_levels.size();
}

const LevelP& Grid::getLevel( int l ) const
{
  ASSERTRANGE(l, 0, numLevels());
  return d_levels[ l ];
}

Level* Grid::addLevel(const Point& anchor, const Vector& dcell, int id)
{
  // find the new level's refinement ratio
  // this should only be called when a new grid is created, so if this level index 
  // is > 0, then there is a coarse-fine relationship between this level and the 
  // previous one.

  IntVector ratio;
  if (d_levels.size() > 0) {
    Vector r = d_levels[d_levels.size()-1]->dCell() / dcell;
    ratio = IntVector((int)r.x(), (int)r.y(), (int)r.z());
  }
  else
    ratio = IntVector(1,1,1);
  
  Level* level = scinew Level(this, anchor, dcell, (int)d_levels.size(), ratio, id);  

  d_levels.push_back( level );
  return level;
}

void Grid::performConsistencyCheck() const
{
  // Verify that patches on a single level do not overlap
  for(int i=0;i<(int)d_levels.size();i++)
    d_levels[i]->performConsistencyCheck();

  // Check overlap between levels
  // See if patches on level 0 form a connected set (warning)
  // Compute total volume - compare if not first time

  //cerr << "Grid::performConsistencyCheck not done\n";
  
  //__________________________________
  //  bullet proofing with multiple levels
  if(d_levels.size() > 0) {
    for(int i=0;i<(int)d_levels.size() -1 ;i++) {
      LevelP level     = d_levels[i];
      LevelP fineLevel = level->getFinerLevel();
      Vector dx_level     = level->dCell();
      Vector dx_fineLevel = fineLevel->dCell();

      //__________________________________
      //make sure that the refinement ratio
      //is really 2 between all levels     
      Vector refineRatio_test = dx_level/dx_fineLevel;
      Vector refineRatio = fineLevel-> getRefinementRatio().asVector();
      Vector smallNum(1e-3, 1e-3, 1e-3);
      
      if (Abs(refineRatio_test - refineRatio).length() > smallNum.length() ) {
        ostringstream desc;
        desc << " The refinement Ratio between Level " << level->getIndex()
             << " and Level " << fineLevel->getIndex() 
             << " is NOT equal to [2,2,2] but " 
             << refineRatio_test << endl;
        //throw InvalidGrid(desc.str());
      }
      //__________________________________
      // finer level can't lay outside of the coarser level
      BBox C_box,F_box;
      level->getSpatialRange(C_box);
      fineLevel->getSpatialRange(F_box);
      
      Point Cbox_min = C_box.min();
      Point Cbox_max = C_box.max(); 
      Point Fbox_min = F_box.min();
      Point Fbox_max = F_box.max();
      
      if(Fbox_min.x() < Cbox_min.x() ||
         Fbox_min.y() < Cbox_min.y() ||
         Fbox_min.z() < Cbox_min.z() ||
         Fbox_max.x() > Cbox_max.x() ||
         Fbox_max.y() > Cbox_max.y() ||
         Fbox_max.z() > Cbox_max.z() ) {
        ostringstream desc;
        desc << " The finer Level " << fineLevel->getIndex()
             << " "<< F_box.min() << " "<< F_box.max()
             << " can't lay outside of coarser level " << level->getIndex()
             << " "<< C_box.min() << " "<< C_box.max() << endl;
        throw InvalidGrid(desc.str());
      }
      
      //__________________________________
      // fine grid must have an even number of cells
      Vector cells = (Fbox_max - Fbox_min)/dx_fineLevel;
      IntVector i_cells( Round(cells.x()), 
                         Round(cells.y()), 
                         Round(cells.z()) );
      if ( i_cells.x()%2 != 0 || i_cells.y()%2 != 0 || i_cells.z()%2 != 0 ){
        ostringstream desc;
        desc << " The finer Level " << fineLevel->getIndex()
             << " must have an even number of cells " <<  i_cells << endl;
        //throw InvalidGrid(desc.str()); 
      }
    }
  }
}

void Grid::printStatistics() const
{
  cerr << "Grid statistics:\n";
  cerr << "Number of levels:\t\t" << numLevels() << '\n';
  unsigned long totalCells = 0;
  unsigned long totalPatches = 0;
  for(int i=0;i<numLevels();i++){
    LevelP l = getLevel(i);
    cerr << "Level " << i << ":\n";
    if (l->getPeriodicBoundaries() != IntVector(0,0,0))
      cerr << "  Periodic boundaries:\t\t" << l->getPeriodicBoundaries()
	   << '\n';
    cerr << "  Number of patches:\t\t" << l->numPatches() << '\n';
    totalPatches += l->numPatches();
    double ppc = double(l->totalCells())/double(l->numPatches());
    cerr << "  Total number of cells:\t" << l->totalCells() << " (" << ppc << " avg. per patch)\n";
    totalCells += l->totalCells();
  }
  cerr << "Total patches in grid:\t\t" << totalPatches << '\n';
  double ppc = double(totalCells)/double(totalPatches);
  cerr << "Total cells in grid:\t\t" << totalCells << " (" << ppc << " avg. per patch)\n";
  cerr << "\n";
}

//////////
// Computes the physical boundaries for the grid
void Grid::getSpatialRange(BBox& b) const
{
  // just call the same function for all the levels
  for(int l=0; l < numLevels(); l++) {
    getLevel(l)->getSpatialRange(b);
  }
}


//__________________________________
// Computes the length in each direction of the grid
void Grid::getLength(Vector& length, const string flag) const
{
  BBox b;
  // just call the same function for all the levels
  for(int l=0; l < numLevels(); l++) {
    getLevel(l)->getSpatialRange(b);
  }
  length = ( b.max() - b.min() );
  if (flag == "minusExtraCells") {
    Vector dx = getLevel(0)->dCell();
    IntVector extraCells = getLevel(0)->getExtraCells();
    Vector ec_length = IntVector(2,2,2) * extraCells * dx;
    length = ( b.max() - b.min() )  - ec_length;
  }
}

void 
Grid::problemSetup(const ProblemSpecP& params, const ProcessorGroup *pg, bool do_amr)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;

   // anchor/highpoint on the grid
   Point anchor(MAXDOUBLE, MAXDOUBLE, MAXDOUBLE);

   // time refinement between a level and the previous one
   int trr = 2;
   grid_ps->get("time_refinement_ratio", trr);

   int levelIndex = 0;

   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the boxes.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.

      // anchor/highpoint on the level
      Point levelAnchor(MAXDOUBLE, MAXDOUBLE, MAXDOUBLE);
      Point levelHighPoint(-MAXDOUBLE, -MAXDOUBLE, -MAXDOUBLE);

      Vector spacing;
      bool have_levelspacing=false;

      if(level_ps->get("spacing", spacing))
        have_levelspacing=true;
      bool have_patchspacing=false;
        

      // first pass - find upper/lower corner, find resolution/spacing
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
         box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
        Point lower;
        box_ps->require("lower", lower);
        Point upper;
        box_ps->require("upper", upper);
        if (levelIndex == 0) {
          anchor=Min(lower, anchor);
        }
        levelAnchor=Min(lower, levelAnchor);
        levelHighPoint=Max(upper, levelHighPoint);
        
        IntVector resolution;
        if(box_ps->get("resolution", resolution)){
           if(have_levelspacing){
              throw ProblemSetupException("Cannot specify level spacing and patch resolution");
           } else {
              // all boxes on same level must have same spacing
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

      LevelP level = addLevel(anchor, spacing);
      level->setTimeRefinementRatio(trr);
      cout << "SETTING TRR to " << trr << endl;
      IntVector anchorCell(level->getCellIndex(levelAnchor + Vector(1.e-6,1.e-6,1.e-6)));
      IntVector highPointCell(level->getCellIndex(levelHighPoint + Vector(1.e-6,1.e-6,1.e-6)));

      // second pass - set up patches and cells
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
         box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
        Point lower;
        box_ps->require("lower", lower);
        Point upper;
        box_ps->require("upper", upper);
        
        IntVector lowCell = level->getCellIndex(lower+Vector(1.e-6,1.e-6,1.e-6));
        IntVector highCell = level->getCellIndex(upper+Vector(1.e-6,1.e-6,1.e-6));
        Point lower2 = level->getNodePosition(lowCell);
        Point upper2 = level->getNodePosition(highCell);
        double diff_lower = (lower2-lower).length();
        double diff_upper = (upper2-upper).length();
        if(diff_lower > 1.e-6) {
          cerr << "lower=" << lower << '\n';
          cerr << "lowCell =" << lowCell << '\n';
          cerr << "highCell =" << highCell << '\n';
          cerr << "lower2=" << lower2 << '\n';
          cerr << "diff=" << diff_lower << '\n';
          
          throw ProblemSetupException("Box lower corner does not coincide with grid");
        }
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
        IntVector extraCells;
        box_ps->getWithDefault("extraCells", extraCells, IntVector(0,0,0));
        level->setExtraCells(extraCells);
        
        IntVector resolution(highCell-lowCell);
        if(resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1) {
          cerr << "highCell: " << highCell << " lowCell: " << lowCell << '\n';
          throw ProblemSetupException("Degenerate patch");
        }
        
        IntVector patches;
        box_ps->getWithDefault("patches", patches,IntVector(1,1,1));
        level->setPatchDistributionHint(patches);
        for(int i=0;i<patches.x();i++){
          for(int j=0;j<patches.y();j++){
            for(int k=0;k<patches.z();k++){
              IntVector startcell = resolution*IntVector(i,j,k)/patches+lowCell;
              IntVector endcell = resolution*IntVector(i+1,j+1,k+1)/patches+lowCell;
              IntVector inStartCell(startcell);
              IntVector inEndCell(endcell);
              startcell -= IntVector(startcell.x() == anchorCell.x() ? extraCells.x():0,
                                     startcell.y() == anchorCell.y() ? extraCells.y():0,
                                     startcell.z() == anchorCell.z() ? extraCells.z():0);
              endcell += IntVector(endcell.x() == highPointCell.x() ? extraCells.x():0,
                                   endcell.y() == highPointCell.y() ? extraCells.y():0,
                                   endcell.z() == highPointCell.z() ? extraCells.z():0);

              Patch* p = level->addPatch(startcell, endcell,
                                         inStartCell, inEndCell);
              p->setLayoutHint(IntVector(i,j,k));
            }
          }
        }
      }
      if (pg->size() > 1 && (level->numPatches() < pg->size()) && !do_amr) {
        throw ProblemSetupException("Number of patches must >= the number of processes in an mpi run");
      }
      
      IntVector periodicBoundaries;
      if(level_ps->get("periodic", periodicBoundaries)){
       level->finalizeLevel(periodicBoundaries.x() != 0,
                          periodicBoundaries.y() != 0,
                          periodicBoundaries.z() != 0);
      }
      else {
       level->finalizeLevel();
      }
      level->assignBCS(grid_ps);
      levelIndex++;
   }
   if(numLevels() >1 && !do_amr) {  // bullet proofing
    throw ProblemSetupException("Grid.cc:problemSetup: Multiple levels encountered in non-AMR grid");
   }
} // end problemSetup()

ostream& operator<<(ostream& out, const Grid& grid)
{
  out.setf(ios::floatfield);
  out.precision(6);
  out << "Grid has " << grid.numLevels() << " level(s)" << endl;
  for ( int levelIndex = 0; levelIndex < grid.numLevels(); levelIndex++ ) {
    LevelP level = grid.getLevel( levelIndex );
    out << "  Level " << level->getID() 
        << ", indx: "<< level->getIndex()
        << " has " << level->numPatches() << " patch(es)" << endl;
    for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
      const Patch* patch = *patchIter;
      out << *patch << endl;
    }
  }
  return out;
}

bool Grid::operator==(const Grid& othergrid) 
{
  if (numLevels() != othergrid.numLevels())
    return false;
  for (int i = 0; i < numLevels(); i++) {
    const Level* level = getLevel(i).get_rep();
    const Level* otherlevel = othergrid.getLevel(i).get_rep();
    if (level->numPatches() != otherlevel->numPatches())
      return false;
    Level::const_patchIterator iter = level->patchesBegin();
    Level::const_patchIterator otheriter = otherlevel->patchesBegin();
    for (; iter != level->patchesEnd(); iter++, otheriter++) {
      const Patch* patch = *iter;
      const Patch* otherpatch = *otheriter;
      if (patch->getLowIndex() != otherpatch->getLowIndex() ||
          patch->getHighIndex() != otherpatch->getHighIndex())
        return false;
    }
      
  }
  return true;

}
