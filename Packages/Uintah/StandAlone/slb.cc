
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Math/Primes.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>


#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sgi_stl_warnings_on.h>

// TODO:
//  DONE Parse geometry from UPS file and fill in weights array properly
//  1. Fix uniform division (40/5 = 8)
//  2. Proper arg parsing (for -div and integer submultiples)
//  DONE Convert patches to world coordinate space and print out
//  DONE Modify UPS file and write it out
//  Bounds for pieces (faster)
//  Summed area table (faster)
//  Allow integer submultiples
// 3. Test with different problems (disks with different weights)
// 5. Benchmark on a real problem
// 4. give to jim
// Zoltan

using namespace Uintah;
using namespace std;

void bisect(const string& div, int num, int factor, Uintah::Primes::FactorType factors,
            const IntVector& low, const IntVector& high, const Array3<float>& weights,
            const LevelP& level, ProblemSpecP& level_ups)
{
  int index;
  if(div[num] == 'x')
    index=0;
  else if(div[num] == 'y')
    index=1;
  else if(div[num] == 'z')
    index=2;
  else
    throw InternalError("bad bisection axis: "+div[num]);

  if(factor == -1){
    static int idx = 0;
    cerr << idx++ << ": patch: " << low << "-" << high << '\n';
    Point low_point = level->getNodePosition(low);
    Point high_point = level->getNodePosition(high);
    IntVector res = high - low;

    ProblemSpecP box = level_ups->appendChild("Box");
    box->appendElement("lower", low_point, false);
    box->appendElement("upper", high_point, false);
    box->appendElement("resolution", res, true);


  } else {
    int number = static_cast<int>(factors[factor]);  // quiet the sgi compiler

    int l = low[index];
    int h = high[index];

    vector<float> sums(h-l);
    for(int i = l; i<h; i++){
      IntVector slab_low = low;
      slab_low[index] = i;
      IntVector slab_high = high;
      slab_high[index] = i+1;
      double sum = 0;
      for(CellIterator iter(slab_low, slab_high); !iter.done(); iter++)
        sum += weights[*iter];
      //cerr << "sum[" << i-l << ": " << slab_low << "-" << slab_high << ": " << sum << '\n';
      sums[i-l] = sum;
    }
    float total = 0;
    for(int i = l; i< h; i++){
      total += sums[i-l];
      sums[i-l] = total;
    }
    double weight_per = total/number;

    for(int i = 0 ; i < number; i ++){
      double w1 = weight_per * i;
      double w2 = weight_per * (i+1);
      int s = 0;
      while(sums[s] < w1 && s < h-l)
        s++;
      int e = 0;
      while(sums[e] < w2 && e < h-l)
        e++;

      IntVector new_low = low;
      new_low[index] = s+l;
      IntVector new_high = high;
      new_high[index] = e+l;

      bisect(div, num+1, factor-1, factors, new_low, new_high, weights, level, level_ups);
    }
  }
}

void usage( char *prog_name );

void
parseArgs( int argc, char *argv[], 
           int & nump, float & weight, string & infile, string & outfile,
           string & divisions, int & submultiples)
{
  if( argc < 4 || argc > 9 ) {
    usage( argv[0] );
  }

  weight = atof( argv[1] );
  if( weight < 0.0 || weight > 1.0 ) {
    cerr << "Weight must be between 0.0 and 1.0\n";
    exit( 1 );
  }
  nump = atoi( argv[2] );
  if( nump < 1 ) {
    cerr << "Number of patches must be greater than 0.\n";
    exit( 1 );
  }

  infile = argv[3];
  outfile = argv[4];

  // parse optional arguments
  for (int i = 5; i < argc; i++) {
    if (strcmp(argv[i], "-div") == 0) {
      i++;
      if (i >= argc) {
        cerr << "-div option needs an argument (i.e., -div xyz)\n";
        exit( 1 );
      }
      for (unsigned int j = 0; j < strlen(argv[i]); j++) {
        if (argv[i][j] != 'x' && argv[i][j] != 'y' && argv[i][j] != 'z') {
          cerr << "Divisions string must be a combination of x, y, or z\n";
          exit( 1 );
        }
      }
      divisions = argv[i];
    }
    else if (strcmp(argv[i], "-sub") == 0) {
      i++;
      if (i >= argc) {
        cerr << "-sub option needs an argument (i.e., -sub 2)\n";
        exit( 1 );
      }
      submultiples = atoi(argv[i]);
      if (submultiples < 1) {
        cerr << "Number of Submultiples must be greater than 0\n";
        exit( 1 );
      }
    }
    else {
      cerr << "Unknown option " << argv[i] << '\n';
      exit( 1 );
    }
  }
}

void
usage( char *prog_name )
{
  cout << "Usage: " << prog_name << " weight num_patches infile outfile "
       << "[-div <pattern>] [-sub <submult>] \n";
  cout << "    weight: 0.0 - 1.0.  0.0 => all cells, 1.0 => all particles\n";
  cout << "    num_patches: number of patches to divvy up domain into.\n";
  cout << "    infile:      .ups file to read in.\n";
  cout << "    outfile:     modified .ups file.\n";
  cout << "    -div <pattern>:     OPTIONAL.  pattern is some combination\n"
       << "        of x,y, and z, where the default is xyz\n";
  cout << "    -sub <submult>:     divide original resolution by submult"
       << "        for quicker estimation\n";
  exit( 1 );
}

int
main(int argc, char *argv[])
{
  try {
    Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
    Uintah::Parallel::initializeManager( argc, argv, "" );

    int    nump;
    float  weight;
    string infile;
    string outfile;
    int submultiples = 1;
    string divisions = "xyz";
    
    parseArgs( argc, argv, nump, weight, infile, outfile,
               divisions, submultiples );
    
    ProblemSpecInterface* reader = scinew ProblemSpecReader(infile);
    
    // Get the problem specification
    ProblemSpecP ups = reader->readInputFile();
    ups->writeMessages(true);
    if(!ups)
      throw ProblemSetupException("Cannot read problem specification");
    
    if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification");
    
    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
    
    // Setup the initial grid
    GridP grid=scinew Grid();
    grid->problemSetup(ups, world);  
    
    for (int l = 0; l < grid->numLevels(); l++) {
      const LevelP &level = grid->getLevel(l);
      
      IntVector low, high;
      level->findCellIndexRange(low, high);
      IntVector diff = high-low;
      long cells = diff.x()*diff.y()*diff.z();
      if(cells != level->totalCells())
        throw ProblemSetupException("Currently slb can only handle square grids");
      
      Uintah::Primes::FactorType factors;
      int n = Uintah::Primes::factorize(nump, factors);
      cerr << nump << ": ";
      for(int i=0;i<n;i++){
        cerr << factors[i] << " ";
      }
      cerr << '\n';
      
      string div = divisions;
      while(static_cast<int>(div.length()) <= n)
        div += divisions;
      
      Array3<float> weights(low, high);
      weights.initialize(weight);
      
      // Parse the geometry from the UPS
      ProblemSpecP mp = ups->findBlock("MaterialProperties");
      ProblemSpecP mpm = mp->findBlock("MPM");
      for (ProblemSpecP child = mpm->findBlock("material"); child != 0;
           child = child->findNextBlock("material")) {
        for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
             geom_obj_ps != 0;
             geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
          vector<GeometryPiece*> pieces;
          GeometryPieceFactory::create(geom_obj_ps, pieces);
          
          GeometryPiece* mainpiece;
          if(pieces.size() == 0){
            throw ProblemSetupException("No piece specified in geom_object");
          } else if(pieces.size() > 1){
            mainpiece = scinew UnionGeometryPiece(pieces);
          } else {
            mainpiece = pieces[0];
          }
          
          for(CellIterator iter(low, high); !iter.done(); iter++){
            Point p = level->getCellPosition(*iter);
            if(mainpiece->inside(p))
              weights[*iter] = 1;
          }
          
          delete mainpiece;
        }
      }
      int factor = n-1;
      
      // remove the 'Box' entry from the ups
      ProblemSpecP g = ups->findBlock("Grid");
      ProblemSpecP lev = g->findBlock("Level");
      ProblemSpecP box = lev->findBlock("Box");
      
      lev->removeChild(box);
      
      bisect(div, 0, factor, factors, low, high, weights, level, lev); 
    }
    
    ofstream out(outfile.c_str());
    out << ups;
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    if(e.stackTrace())
      cerr << "Stack trace: " << e.stackTrace() << '\n';
  } catch (std::exception e){
    cerr << "Caught std exception: " << e.what() << '\n';
  } catch(...){
    cerr << "Caught unknown exception\n";
  }
}
