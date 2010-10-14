/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Grid/LinearInterpolator.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
    
LinearInterpolator::LinearInterpolator()
{
  d_size = 8;
  d_patch = 0;
}

LinearInterpolator::LinearInterpolator(const Patch* patch)
{
  d_size = 8;
  d_patch = patch;
}

LinearInterpolator::~LinearInterpolator()
{
}

LinearInterpolator* LinearInterpolator::clone(const Patch* patch)
{
  return scinew LinearInterpolator(patch);
 }
    
//__________________________________
void LinearInterpolator::findCellAndWeights(const Point& pos,
                                           vector<IntVector>& ni, 
                                           vector<double>& S,
                                           const Vector& size,
                                           const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos );
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fz = cellpos.z() - iz;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  double fz1 = 1-fz;
  S[0] = fx1 * fy1 * fz1;
  S[1] = fx1 * fy1 * fz;
  S[2] = fx1 * fy * fz1;
  S[3] = fx1 * fy * fz;
  S[4] = fx * fy1 * fz1;
  S[5] = fx * fy1 * fz;
  S[6] = fx * fy * fz1;
  S[7] = fx * fy * fz;
}

//______________________________________________________________________
//  This interpolation function from equation 14 of 
//  Jin Ma, Hongbind Lu and Ranga Komanduri
// "Structured Mesh Refinement in Generalized Interpolation Material Point Method
//  for Simulation of Dynamic Problems" CMES, vol 12, no 3, pp. 213-227 2006
void LinearInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni,
                                            vector<double>& S,
                                            constNCVariable<Stencil7>& zoi)
{
  const Level* level = d_patch->getLevel();
  IntVector refineRatio(level->getRefinementRatio());

  //__________________________________
  // Identify the nodes that are along the coarse fine interface
  //
  //             |           |
  //             |           |
  //  ___________*__o__o__o__o________
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  . 0 . |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*__o__o__o__o________
  //             |           |        
  //             |           |         
  //             |           |        
  //  Coarse fine interface nodes: *
  //  Ghost nodes on the fine level: o  (technically these don't exist
  //  Particle postition on the coarse level: 0
  
  const int ngn = 0;
  IntVector lo = d_patch->getNodeLowIndex(ngn);
  IntVector hi = d_patch->getNodeHighIndex(ngn) - IntVector(1,1,1);
  
  // Find node index of coarse cell and then map that node to fine level
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  IntVector ni_c = coarseLevel->getCellIndex(pos);
  IntVector ni_f = coarseLevel->mapNodeToFiner(ni_c);
  
  int ix = ni_f.x();
  int iy = ni_f.y();
  int iz = ni_f.z();
  
  // loop over all (o) nodes and find which lie on the CFI
  for(int x = 0; x<= refineRatio.x(); x++){
    for(int y = 0; y<= refineRatio.y(); y++){
      for(int z = 0; z<= refineRatio.z(); z++){
      
        IntVector node = IntVector(ix + x, iy + y, iz + z);
        //    x-,y-,z- patch face   x+, y+, z+ patch face
        if( (node.x() == lo.x() || node.x() == hi.x() ) ||
            (node.y() == lo.y() || node.y() == hi.y() ) ||
            (node.z() == lo.z() || node.z() == hi.z() ) ) {
          ni.push_back(node);
          cout << " ni " << node << endl;
        } 
      }
    }
  }
  
  cout << " ni.size " << ni.size() << endl;

  //__________________________________
  // Reference Nomenclature: Stencil7 Mapping
  // Lx- :  L.w
  // Lx+ :  L.e
  // Ly- :  L.s
  // Ly+ :  L.n
  // Lz- :  L.b
  // Lz+ :  L.t
   
  for (int i = 0; i< ni.size(); i++){
    Point nodepos = level->getNodePosition(ni[i]);
    double dx = pos.x() - nodepos.x();
    double dy = pos.y() - nodepos.y();
    double dz = pos.z() - nodepos.z();
    double fx = -9, fy = -9, fz = -9;
    
    Stencil7 L = zoi[ni[i]];  // fine level

    if(dx <= -L.w){                       // Lx-
      fx = 0; 
    }
    else if ( -L.w <= dx && dx <= 0 ){   // Lx-
      fx = 1 + dx/L.w;
    }
    else if ( 0 <= dx  && dx <= L.e ){    // Lx+
      fx = 1 - dx/L.e;
    }
    else if (L.e <= dx){                  // Lx+
      fx = 0;
    }

    if(dy <= -L.s){                       // Ly-
      fy = 0;
    }
    else if ( -L.s <= dy && dy <= 0 ){    // Ly-
      fy = 1 + dy/L.s;
    }
    else if ( 0 <= dy && dy <= L.n ){    // Ly+
      fy = 1 - dy/L.n;
    }
    else if (L.n <= dy){                 // Ly+
      fy = 0;
    }

    if(dz <= -L.b){                       // Lz-
      fz = 0;
    }
    else if ( -L.b <= dz && dz <= 0 ){    // Lz-
      fz = 1 + dz/L.b;
    }
    else if ( 0 <= dz && dz <= L.n ){    // Lz+
      fz = 1 - dz/L.t;
    }
    else if (L.t <= dz){                 // Lz+
      fz = 0;
    }

    S[i] = fx * fy * fz;
    cout << "  pos " << pos << " node " << ni[i] << " fx " << fx << " fy " << fy <<  " fz " << fz << "    S[i] "<< S[i] << endl;
  }
}

void LinearInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni,
                                            vector<double>& S,
                                            constNCVariable<Stencil7>& zoi,
                                            constNCVariable<Stencil7>& zoi_fine,
                                            const bool& getFiner,
                                            int& num_cur, int& num_fine,
                                            int& num_coarse, const Vector& size,
                                            bool coarse_particle,
                                            const Patch* patch)
{
  num_coarse=0;
  num_fine=0;
  const Level* lvl = d_patch->getLevel();
  vector<IntVector> cur(8);

  constNCVariable<Stencil7> zoi_use;

  int keep=0;
  if(coarse_particle){
    zoi_use=zoi_fine;
    findFinerNodes(pos,cur,lvl,patch);
    for(int i=0;i<8;i++){
      if(lvl->selectPatchForNodeIndex(cur[i])!=0){
        int use = (int) zoi_fine[cur[i]].p;
        ni[keep]=cur[i];
        keep+=use;
      }
    }
  }
  else{
    zoi_use=zoi;
    findNodes(pos,cur,lvl);
    for(int i=0;i<8;i++){
      int use = (int) zoi[cur[i]].p;
      ni[keep]=cur[i];
      keep+=use;
    }
  }
  num_cur=keep;

  double Sx,Sy,Sz,r;
  for(int i=0;i<keep;i++){
    Point node_pos = lvl->getNodePosition(ni[i]);
    Stencil7 ZOI = zoi_use[ni[i]];
    r = pos.x() - node_pos.x();
    uS(Sx,r,ZOI.e,ZOI.w);
    r = pos.y() - node_pos.y();
    uS(Sy,r,ZOI.n,ZOI.s);
    r = pos.z() - node_pos.z();
    uS(Sz,r,ZOI.t,ZOI.b);
    S[i]=Sx*Sy*Sz;
  }

  if(lvl->hasFinerLevel() && getFiner && keep != 8){
    const Level* fineLevel = lvl->getFinerLevel().get_rep();
    findFinerNodes(pos,cur,fineLevel,patch);
    for(int i=0;i<8;i++){
      if(fineLevel->selectPatchForNodeIndex(cur[i])!=0){
        ni[keep]=cur[i];
        keep++;
      }
    }

    double Sx,Sy,Sz,r;
    for(int i=keep;i<8;i++){
      Point node_pos = fineLevel->getNodePosition(ni[i]);
      Stencil7 ZOI = zoi_fine[ni[i]];
      r = pos.x() - node_pos.x();
      uS(Sx,r,ZOI.e,ZOI.w);
      r = pos.y() - node_pos.y();
      uS(Sy,r,ZOI.n,ZOI.s);
      r = pos.z() - node_pos.z();
      uS(Sz,r,ZOI.t,ZOI.b);
      S[i]=Sx*Sy*Sz;
    }
    num_fine=keep-num_cur;
  }

  return;
}
 
void LinearInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Vector& size,
                                               const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fz = cellpos.z() - iz;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  double fz1 = 1-fz;
  d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
  d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
  d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
  d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
  d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
  d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
  d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
  d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
}

void 
LinearInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Vector& size,
                                                   const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fz = cellpos.z() - iz;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  double fz1 = 1-fz;
  S[0] = fx1 * fy1 * fz1;
  S[1] = fx1 * fy1 * fz;
  S[2] = fx1 * fy * fz1;
  S[3] = fx1 * fy * fz;
  S[4] = fx * fy1 * fz1;
  S[5] = fx * fy1 * fz;
  S[6] = fx * fy * fz1;
  S[7] = fx * fy * fz;
  d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
  d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
  d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
  d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
  d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
  d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
  d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
  d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
}

int LinearInterpolator::size()
{
  return d_size;
}

void LinearInterpolator::findFinerNodes(const Point& pos,
                               vector<IntVector>& cur,
                               const Level* level, 
                               const Patch* patch)
{
        Point cellpos = level->positionToIndex(pos);
        int r = Floor(cellpos.x());
        int s = Floor(cellpos.y());
        int t = Floor(cellpos.z());

        IntVector l(patch->getExtraNodeLowIndex());
        IntVector h(patch->getExtraNodeHighIndex());

        int ix = max(max(l.x()-1,r),min(h.x()-1,r));
        int iy = max(max(l.y()-1,s),min(h.y()-1,s));
        int iz = max(max(l.z()-1,t),min(h.z()-1,t));

        cur[0] = IntVector(ix, iy, iz);
        cur[1] = IntVector(ix, iy, iz+1);
        cur[2] = IntVector(ix, iy+1, iz);
        cur[3] = IntVector(ix, iy+1, iz+1);
        cur[4] = IntVector(ix+1, iy, iz);
        cur[5] = IntVector(ix+1, iy, iz+1);
        cur[6] = IntVector(ix+1, iy+1, iz);
        cur[7] = IntVector(ix+1, iy+1, iz+1);
}
