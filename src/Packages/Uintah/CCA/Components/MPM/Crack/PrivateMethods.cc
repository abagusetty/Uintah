/********************************************************************************
    Crack.cc
    PART SEVEN: PRIVATE METHODS 

    Created by Yajun Guo in 2002-2004.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <fstream>

// The AIX header files have hz defined for some reason
// and programmers like to use this as a variable name so...
#if defined( _AIX )
  #if defined( hz )
    #undef hz
  #endif
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

using std::vector;
using std::string;

// Find the segment numbers which are connected by the node
void Crack::FindSegsFromNode(const int& m,const int& node, int segs[])
{
  // segs[R] -- the seg on the right of the node
  // segs[L] -- the seg on the left of the node

  segs[R]=segs[L]=-1;

  int ncfSegs=(int)cfSegNodes[m].size()/2;
  for(int j=0; j<ncfSegs; j++) {
    int node0=cfSegNodes[m][2*j];
    int node1=cfSegNodes[m][2*j+1];
    if(node==node1) // the right seg
      segs[R]=j;
    if(node==node0) // the left seg
      segs[L]=j;
  } // End of loop over j

  if(segs[R]<0 && segs[L]<0) {
    cout << " Failure to find the crack-front segments for node "
         << node << ". Program terminated." << endl;
    exit(1);
  }
}

// Find the previous index, and the minimum & maximum indexes
// for each crack-front node. The subroutine should be referenced 
// once a new crack-front is generated
void Crack::FindCrackFrontNodeIndexes(const int& m)
{   
  /* Task 1: cfSegPreIdx --- The previous index of a crack-front node
             node[i]=node[preIdx] (preIdx<i)
  */
  cfSegPreIdx[m].clear();
  int num=(int)cfSegNodes[m].size();
  cfSegPreIdx[m].resize(num);
  for(int i=0; i<num; i++) {
    int preIdx=-1;
    int thisNode=cfSegNodes[m][i];
    for(int j=i-1; j>=0; j--) {
      int preNode=cfSegNodes[m][j];
      if(thisNode==preNode) {
        preIdx=j;
        break;
      }
    }
    cfSegPreIdx[m][i]=preIdx;
    if(cfSegPreIdx[m][i]>=i) {
      cout << "   ! Error in finding cfSegPreIdx. Program terminated." << endl;    
      exit(1);
    }
  }

  /* Task 2: cfSegMinIdx and cfSegMaxIdx -- The minimum and maximum indexes of 
             of the sub-crack which the node is located on
  */
  cfSegMaxIdx[m].clear();
  cfSegMinIdx[m].clear();
  cfSegMaxIdx[m].resize(num);
  cfSegMinIdx[m].resize(num);
    
  int maxIdx=-1,minIdx=0;
  for(int i=0; i<num; i++) {
    if(!(i>=minIdx && i<=maxIdx)) { // node i not within the sub-crack
      // Find maxIdx
      for(int j=((i%2)!=0?i:i+1); j<num; j+=2) {
        if(j==num-1 ||
           (j<num-1 && cfSegNodes[m][j]!=cfSegNodes[m][j+1])) {
          maxIdx=j;
          break;
        }
      }
    }
    cfSegMinIdx[m][i]=minIdx;
    cfSegMaxIdx[m][i]=maxIdx;
    if(i==maxIdx) minIdx=maxIdx+1;
  }

  for(int i=0; i<num; i++) {
    if(!(i>=cfSegMinIdx[m][i] && i<=cfSegMaxIdx[m][i]) ||
      cfSegMinIdx[m][i]<0 ||
      cfSegMaxIdx[m][i]>num-1 ||
      cfSegMinIdx[m][i]%2!=0 ||
      cfSegMaxIdx[m][i]%2==0) {
      int pid;
      MPI_Comm_rank(mpi_crack_comm, &pid);
      if(pid==0) {
        cout << "   ! Error in finding cfSegMinIdx and cfSegMaxIdx."
             << " Program terminated. " << endl;
        cout << "     i=" << i << ", cfSegMinIdx[m][i]=" << cfSegMinIdx[m][i]
             << ", cfSegMaxIdx[m][i]=" << cfSegMaxIdx[m][i] << endl;
        for(int j=0; j<num; j++)
          cout << "j=" << j << ", cfSegNodes[m][j]=" << cfSegNodes[m][j] 
               << ", preIdx=" << cfSegPreIdx[m][j] << endl;
      }
      exit(1);
    } 
  }
}
  
// Find if a point is within the real global grid  
short Crack::PhysicalGlobalGridContainsPoint(const double& dx,const Point& pt)
{
  // Return true if pt is within the real global grid or
  // around it (within 1% of cell-size)

  double px=pt.x(),  py=pt.y(),  pz=pt.z();
  double lx=GLP.x(), ly=GLP.y(), lz=GLP.z();
  double hx=GHP.x(), hy=GHP.y(), hz=GHP.z();
  
  return ((px>lx || fabs(px-lx)/dx<0.01) && (px<hx || fabs(px-hx)/dx<0.01) &&
          (py>ly || fabs(py-ly)/dx<0.01) && (py<hy || fabs(py-hy)/dx<0.01) &&
          (pz>lz || fabs(pz-lz)/dx<0.01) && (pz<hz || fabs(pz-hz)/dx<0.01));
}         

// Find the intersection between a line-segment (p1->p2) and grid boundary
void Crack::FindIntersectionLineAndGridBoundary(const Point& p1, Point& p2)
{
  // If p2 is outside of global grid, find the intersection 
  // between line-segment (p1->p2) and grid boundary, and store it in p2

  double lx=GLP.x(), ly=GLP.y(), lz=GLP.z();
  double hx=GHP.x(), hy=GHP.y(), hz=GHP.z();

  double x1=p1.x(), y1=p1.y(), z1=p1.z();
  double x2=p2.x(), y2=p2.y(), z2=p2.z();

  Vector v=TwoPtsDirCos(p1,p2);
  double l=v.x(), m=v.y(), n=v.z();

  if(x2>hx || x2<lx) {
    if(x2>hx) x2=hx;
    if(x2<lx) x2=lx;
    y2=y1+(x2-x1)/l*m;
    z2=z1+(x2-x1)/l*n;
  }

  if(y2>hy || y2<ly) {
    if(y2>hy) y2=hy;
    if(y2<ly) y2=ly;
    x2=x1+(y2-y1)/m*l;
    z2=z1+(y2-y1)/m*n;
  }

  if(z2>hz || z2<lz) {
    if(z2>hz) z2=hz;
    if(z2<lz) z2=lz;
    x2=x1+(z2-z1)/n*l;
    y2=y1+(z2-z1)/n*m;
  }

  p2=Point(x2,y2,z2);
}

// Calculate normal of a triangle
Vector Crack::TriangleNormal(const Point& p1,
            const Point& p2, const Point& p3)
{
  double x21,x31,y21,y31,z21,z31;
  double a,b,c;
  Vector norm;

  x21=p2.x()-p1.x();
  x31=p3.x()-p1.x();
  y21=p2.y()-p1.y();
  y31=p3.y()-p1.y();
  z21=p2.z()-p1.z();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;

  if(Vector(a,b,c).length()>1.e-16)
     norm=Vector(a,b,c)/Vector(a,b,c).length();
  else
     norm=Vector(a,b,c);

  return norm;
}

// Detect the relation between two points and a plane
short Crack::ParticleNodeCrackPLaneRelation(const Point& p, 
                            const Point& g, const Point& n1,
                           const Point& n2, const Point& n3)
{
  // p, g     -- two points (usually particle and node)
  // n1,n2,n3 -- three points on the plane

  short cross;
  // cross=0 if p and g are on the same side of the plane,
  // cross=1 if p-g crosses the plane and p is above the plane,
  // cross=2 if p-g crosses the plane and p is below the plane.

  double x1,y1,z1,x2,y2,z2,x3,y3,z3,xp,yp,zp,xg,yg,zg;
  double x21,y21,z21,x31,y31,z31,a,b,c,d,dp,dg;

  x1=n1.x(); y1=n1.y(); z1=n1.z();
  x2=n2.x(); y2=n2.y(); z2=n2.z();
  x3=n3.x(); y3=n3.y(); z3=n3.z();
  xp=p.x();  yp=p.y();  zp=p.z();
  xg=g.x();  yg=g.y();  zg=g.z();

  x21=x2-x1; y21=y2-y1; z21=z2-z1;
  x31=x3-x1; y31=y3-y1; z31=z3-z1;

  a=y21*z31-z21*y31;
  b=z21*x31-x21*z31;
  c=x21*y31-y21*x31;
  d=-a*x1-b*y1-c*z1;

  dp=a*xp+b*yp+c*zp+d;
  dg=a*xg+b*yg+c*zg+d;

  if(fabs(dg)<1.e-16) { // node on crack plane
    if(dp>0.) 
      cross=1;  // p above carck
    else
      cross=2;  // p below crack
  }
  else { // node not on crack plane
    if(dp*dg>0.)
      cross=0;  // p, g on same side
    else if(dp>0.)
      cross=1;  // p above, g below
    else
      cross=2;  // p below, g above
  }

  return cross;
}


// Compute signed volume of a tetrahedron
double Crack::Volume(const Point& p1, const Point& p2,
                     const Point& p3, const Point& p)
{
   // p1,p2,p3 -- three corners on bottom. p -- vertex

   double vol;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x,y,z;

   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x = p.x(); y = p.y(); z = p.z();

   vol=-(x1-x2)*(y3*z-y*z3)-(x3-x)*(y1*z2-y2*z1)
       +(y1-y2)*(x3*z-x*z3)+(y3-y)*(x1*z2-x2*z1)
       -(z1-z2)*(x3*y-x*y3)-(z3-z)*(x1*y2-x2*y1);

   if(fabs(vol)<1.e-16)
     return (0.);
   else
     return(vol);
}

IntVector Crack::CellOffset(const Point& p1, const Point& p2, Vector dx)
{
  int nx,ny,nz;
  if(fabs(p1.x()-p2.x())/dx.x()<1e-6) // p1.x()=p2.x()
    nx=NGN-1;
  else
    nx=NGN;
  if(fabs(p1.y()-p2.y())/dx.y()<1e-6) // p1.y()=p2.y()
    ny=NGN-1;
  else
    ny=NGN;
  if(fabs(p1.z()-p2.z())/dx.z()<1e-6) // p1.z()=p2.z()
    nz=NGN-1;
  else
    nz=NGN;

  return IntVector(nx,ny,nz);
}

// Detect if line-segment (p3-p4) is included in line-segment (p1-p2)
short Crack::TwoLinesDuplicate(const Point& p1,const Point& p2,
                               const Point& p3,const Point& p4)
{
   double l12,l31,l32,l41,l42;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4;
   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x4=p4.x(); y4=p4.y(); z4=p4.z();

   l12=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
   l31=sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1));
   l32=sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2));
   l41=sqrt((x4-x1)*(x4-x1)+(y4-y1)*(y4-y1)+(z4-z1)*(z4-z1));
   l42=sqrt((x4-x2)*(x4-x2)+(y4-y2)*(y4-y2)+(z4-z2)*(z4-z2));

   if(fabs(l31+l32-l12)/l12<1.e-6 && fabs(l41+l42-l12)/l12<1.e-6 && l41>l31)
     return 1;
   else
     return 0;
}

// Find parameters (A[14]) of J-contour
void Crack::FindJPathCircle(const Point& origin, const Vector& v1,
                    const Vector& v2,const Vector& v3, double A[])
{
   /* J-contour circle's equation
      A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0
      and A10x+A11y+A12z+A13=0
      where r is radius of the circle */

   double x0,y0,z0;
   double l1,m1,n1,l2,m2,n2,l3,m3,n3;

   x0=origin.x(); y0=origin.y(); z0=origin.z();

   l1=v1.x(); m1=v1.y(); n1=v1.z();
   l2=v2.x(); m2=v2.y(); n2=v2.z();
   l3=v3.x(); m3=v3.y(); n3=v3.z();

   double term1,term2;
   term1=l1*x0+m1*y0+n1*z0;
   term2=l2*x0+m2*y0+n2*z0;

   A[0]=l1*l1+l2*l2;
   A[1]=m1*m1+m2*m2;
   A[2]=n1*n1+n2*n2;
   A[3]=2*(l1*m1+l2*m2);
   A[4]=2*(l1*n1+l2*n2);
   A[5]=2*(m1*n1+m2*n2);
   A[6]=-2*(l1*term1+l2*term2);
   A[7]=-2*(m1*term1+m2*term2);
   A[8]=-2*(n1*term1+n2*term2);
   A[9]=A[0]*x0*x0+A[1]*y0*y0+A[2]*z0*z0+A[3]*x0*y0+A[4]*x0*z0+A[5]*y0*z0;

   A[10]=l3;
   A[11]=m3;
   A[12]=n3;
   A[13]=-(l3*x0+m3*y0+n3*z0);
}

// Find the intersection between J-contour and crack plane
bool Crack::FindIntersectionOfJPathAndCrackPlane(const int& m,
              const double& radius, const double M[],Point& crossPt)
{  /*
      J-contour's equations:
        Ax^2+By^2+Cz^2+Dxy+Exz+Fyz+Gx+Hy+Iz+J-r^2=0 and a1x+b1y+c1z+d1=0
      crack plane equation:
        a2x+b2y+c2z+d2=0
      r -- J-contour's radius. Parameters are stroed in M.
   */
   double A,B,C,D,E,F,G,H,I,J,a1,b1,c1,d1;
   A=M[0];      a1=M[10];
   B=M[1];      b1=M[11];
   C=M[2];      c1=M[12];
   D=M[3];      d1=M[13];
   E=M[4];
   F=M[5];
   G=M[6];
   H=M[7];
   I=M[8];
   J=M[9];

   int numCross=0;
   crossPt=Point(-9e32,-9e32,-9e32);
   for(int i=0; i<(int)ce[m].size(); i++) {  // Loop over crack segments
     // Find equation of crack segment: a2x+b2y+c2z+d2=0
     double a2,b2,c2,d2;   // parameters of a 3D plane
     Point pt1,pt2,pt3;    // three vertices of the segment
     pt1=cx[m][ce[m][i].x()];
     pt2=cx[m][ce[m][i].y()];
     pt3=cx[m][ce[m][i].z()];
     FindPlaneEquation(pt1,pt2,pt3,a2,b2,c2,d2);

     /* Define crack-segment coordinates (X',Y',Z')
        The origin located at p1, and X'=p1->p2
        v1,v2,v3 -- dirction cosines of new axes X',Y' and Z'
     */
     Vector v1,v2,v3;
     double term1 = sqrt(a2*a2+b2*b2+c2*c2);
     v2=Vector(a2/term1,b2/term1,c2/term1);
     v1=TwoPtsDirCos(pt1,pt2);
     Vector v12=Cross(v1,v2);
     v3=v12/v12.length();        // right-hand system
     // Transform matrix from global to local
     Matrix3 T=Matrix3(v1.x(),v1.y(),v1.z(),v2.x(),v2.y(),v2.z(),
                       v3.x(),v3.y(),v3.z());

     /* Find intersection between J-path circle and crack plane
        first combine a1x+b1y+c1z+d1=0 And a2x+b2y+c2z+d2=0, get
        x=p1*z+q1 & y=p2*z+q2 (CASE 1) or
        x=p1*y+q1 & z=p2*y+q2 (CASE 2) or
        y=p1*x+q1 & z=p2*y+q2 (CASE 3), depending on the equations
        then combine with equation of the circle, getting the intersection
     */
     int CASE=0;
     double delt1,delt2,delt3,p1,q1,p2,q2;
     double abar,bbar,cbar,abc;
     Point crossPt1,crossPt2;

     delt1=a1*b2-a2*b1;
     delt2=a1*c2-a2*c1;
     delt3=b1*c2-b2*c1;
     if(fabs(delt1)>=fabs(delt2) && fabs(delt1)>=fabs(delt3)) CASE=1;
     if(fabs(delt2)>=fabs(delt1) && fabs(delt2)>=fabs(delt3)) CASE=2;
     if(fabs(delt3)>=fabs(delt1) && fabs(delt3)>=fabs(delt2)) CASE=3;


     double x1=0.,y1=0.,z1=0.,x2=0.,y2=0.,z2=0.;
     switch(CASE) {
       case 1:
         p1=(b1*c2-b2*c1)/delt1;
         q1=(b1*d2-b2*d1)/delt1;
         p2=(a2*c1-a1*c2)/delt1;
         q2=(a2*d1-a1*d2)/delt1;
         abar=p1*p1*A+p2*p2*B+C+p1*p2*D+p1*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*B+(p1*q2+p2*q1)*D+q1*E+q2*F+p1*G+p2*H+I;
         cbar=q1*q1*A+q2*q2*B+q1*q2*D+q1*G+q2*H+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         z1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*z1+q1;
         y1=p2*z1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         z2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*z2+q1;
         y2=p2*z2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 2:
         p1=(b2*c1-b1*c2)/delt2;
         q1=(c1*d2-c2*d1)/delt2;
         p2=(a2*b1-a1*b2)/delt2;
         q2=(a2*d1-a1*d2)/delt2;
         abar=p1*p1*A+B+p2*p2*C+p1*D+p1*p2*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*C+q1*D+(p1*q2+p2*q1)*E+q2*F+p1*G+H+p2*I;
         cbar=q1*q1*A+q2*q2*C+q1*q2*E+q1*G+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         y1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*y1+q1;
         z1=p2*y1+q2;
         crossPt1=Point(x1,y1,z1);
         //the second solution
         y2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*y2+q1;
         z2=p2*y2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 3:
         p1=(a2*c1-a1*c2)/delt3;
         q1=(c1*d2-c2*d1)/delt3;
         p2=(a1*b2-a2*b1)/delt3;
         q2=(b2*d1-b1*d2)/delt3;
         abar=A+p1*p1*B+p2*p2*C+p1*D+p2*E+p1*p2*F;
         bbar=2*p1*q1*B+2*p2*q2*C+q1*D+q2*E+(p1*q2+p2*q1)*F+G+p1*H+p2*I;
         cbar=q1*q1*B+q2*q2*C+q1*q2*F+q1*H+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         x1=0.5*(-bbar+sqrt(abc))/abar;
         y1=p1*x1+q1;
         z1=p2*x1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         x2=0.5*(-bbar-sqrt(abc))/abar;
         y2=p1*x2+q1;
         z2=p2*x2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
     }

     /* Detect if crossPt1 & crossPt2 in the triangular segment.
        Transform and rotate the coordinates of crossPt1 and crossPt2 into
        crack-segment coordinates (X', Y' and Z')
     */
     Point p1p,p2p,p3p,crossPt1p,crossPt2p;
     p1p     =Point(0.,0.,0.)+T*(pt1-pt1);
     p2p     =Point(0.,0.,0.)+T*(pt2-pt1);
     p3p     =Point(0.,0.,0.)+T*(pt3-pt1);
     crossPt1p=Point(0.,0.,0.)+T*(crossPt1-pt1);
     crossPt2p=Point(0.,0.,0.)+T*(crossPt2-pt1);
     if(PointInTriangle(crossPt1p,p1p,p2p,p3p)) {
       numCross++;
       crossPt=crossPt1;
     }
     if(PointInTriangle(crossPt2p,p1p,p2p,p3p)) {
       numCross++;
       crossPt=crossPt2;
     }
   } // End of loop over crack segments

   if(numCross==0)
     return 0;
   else
     return 1;
}

// Calculate direction cosines of line p1->p2
Vector Crack::TwoPtsDirCos(const Point& p1,const Point& p2)
{
  double dx,dy,dz,ds;
  dx=p2.x()-p1.x();
  dy=p2.y()-p1.y();
  dz=p2.z()-p1.z();
  ds=sqrt(dx*dx+dy*dy+dz*dz);
  return Vector(dx/ds, dy/ds, dz/ds);
}

// Find the equation of a plane by three points on it
void Crack::FindPlaneEquation(const Point& p1,const Point& p2,
            const Point& p3, double& a,double& b,double& c,double& d)
{
  // plane equation ax+by+cz+d=0

  double x21,x31,y21,y31,z21,z31;

  x21=p2.x()-p1.x();
  y21=p2.y()-p1.y();
  z21=p2.z()-p1.z();

  x31=p3.x()-p1.x();
  y31=p3.y()-p1.y();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  d=-p1.x()*a-p1.y()*b-p1.z()*c;
}

// Detect if a point is within a triangle (2D case)
short Crack::PointInTriangle(const Point& p,const Point& pt1,
                           const Point& pt2,const Point& pt3)
{
  // y=0 for all points

  double x1,z1,x2,z2,x3,z3,x,z;
  double area_p1p2p,area_p2p3p,area_p3p1p,area_p123;

  x1=pt1.x(); z1=pt1.z();
  x2=pt2.x(); z2=pt2.z();
  x3=pt3.x(); z3=pt3.z();
  x =p.x();   z =p.z();

  area_p1p2p=x1*z2+x2*z+x*z1-x1*z-x2*z1-x*z2;
  area_p2p3p=x2*z3+x3*z+x*z2-x2*z-x3*z2-x*z3;
  area_p3p1p=x3*z1+x1*z+x*z3-x3*z-x1*z3-x*z1;

  area_p123=fabs(x1*z2+x2*z3+x3*z1-x1*z3-x2*z1-x3*z2);

  if(fabs(area_p1p2p)/area_p123<1.e-3) area_p1p2p=0.;
  if(fabs(area_p2p3p)/area_p123<1.e-3) area_p2p3p=0.;
  if(fabs(area_p3p1p)/area_p123<1.e-3) area_p3p1p=0.;

  return (area_p1p2p<=0. && area_p2p3p<=0. && area_p3p1p<=0.);
}

// Detect if doing fracture analysis at this time step
void Crack::DetectIfDoingFractureAnalysisAtThisTimeStep(double time)
{
  static double timeforcalculateJK=0.0;
  static double timeforpropagation=0.0;

  double calJKInterval=0.0;
  if(d_calFractParameters=="true") {
    if(time>=timeforcalculateJK) {
      calFractParameters=YES;
      timeforcalculateJK+=calJKInterval;
    }
    else {
     calFractParameters=NO;
    }
  }
  else if(d_calFractParameters=="false") {
   calFractParameters=0;
  }
  else if(d_calFractParameters=="every_time_step"){
    calFractParameters=1;
  }

  double propagationInterval=0.0;
  if(d_doCrackPropagation=="true") {
    if(time>=timeforpropagation){
      doCrackPropagation=YES;
      timeforpropagation+=propagationInterval;
    }
    else {
      doCrackPropagation=NO;
    }
  }
  else if(d_doCrackPropagation=="false") {
    doCrackPropagation=0;
  }
  else if(d_doCrackPropagation=="every_time_step"){
    doCrackPropagation=1;
  }
}

// Apply symmetric boundary condition to crack points
void Crack::ApplySymmetricBCsToCrackPoints(const Vector& cs,
                        const Point& old_pt,Point& new_pt)
{
  // cs -- cell size
  for(Patch::FaceType face = Patch::startFace;
       face<=Patch::endFace; face=Patch::nextFace(face)) {
    if(GridBCType[face]=="symmetry") {
      if( face==Patch::xminus && fabs(old_pt.x()-GLP.x())/cs.x()<1.e-2 )
        new_pt(0)=GLP.x(); // On symmetric face x-
      if( face==Patch::xplus  && fabs(old_pt.x()-GHP.x())/cs.x()<1.e-2 )
        new_pt(0)=GHP.x(); // On symmetric face x+
      if( face==Patch::yminus && fabs(old_pt.y()-GLP.y())/cs.y()<1.e-2 )
        new_pt(1)=GLP.y(); // On symmetric face y-
      if( face==Patch::yplus  && fabs(old_pt.y()-GHP.y())/cs.y()<1.e-2 )
        new_pt(1)=GHP.y(); // On symmetric face y+
      if( face==Patch::zminus && fabs(old_pt.z()-GLP.z())/cs.z()<1.e-2 )
        new_pt(2)=GLP.z(); // On symmetric face z-
      if( face==Patch::zplus  && fabs(old_pt.z()-GHP.z())/cs.z()<1.e-2 )
        new_pt(2)=GHP.z(); // On symmetric face z+
    }
  }
}

// Calculate normals, tangential normals and bi-normals of crack plane 
// at  crack-front nodes
short Crack::CalculateCrackFrontNormals(const int& mm)
{
  int i=-1,l=-1,k=-1;
  int cfNodeSize=(int)cfSegNodes[mm].size();

  /* Task 1: Calculate tangential normals at crack-front nodes
             by cubic spline fitting, and/or smooth crack front
  */
  short  flag=1;       // Smooth successfully
  double ep=1.e-6;     // Tolerance

  cfSegV3[mm].clear();
  cfSegV3[mm].resize(cfNodeSize);

  // Minimum and maximum index of each sub-crack
  int minIdx=-1,maxIdx=-1;
  int minNode=-1,maxNode=-1;
  int numSegs=-1,numPts=-1;
  vector<Point>  pts; // Crack-front point subset of the sub-crack
  vector<Vector> V3;  // Crack-front point tangential vector
  vector<double> dis; // Arc length from the starting point
  vector<int>    idx;

  for(k=0; k<cfNodeSize;k++) {
    // Step a: Collect crack points for current sub-crack
    int node=cfSegNodes[mm][k];
    int segs[2];
    FindSegsFromNode(mm,node,segs);

    if(k>maxIdx) { // The next sub-crack
      maxIdx=cfSegMaxIdx[mm][k];
      minIdx=cfSegMinIdx[mm][k];

      // numbers of segs and points of this sub-crack  
      minNode=cfSegNodes[mm][minIdx];
      maxNode=cfSegNodes[mm][maxIdx];
      numSegs=(maxIdx-minIdx+1)/2;
      numPts=numSegs+1; 

      // Allocate memory for the sub-crack
      pts.resize(numPts);
      V3.resize(numPts);
      dis.resize(numPts);
      idx.resize(maxIdx+1);
    }

    if(k>=minIdx && k<=maxIdx) { // For the sub-crack
      short preIdx=cfSegPreIdx[mm][k];
      int ki=(k-minIdx+1)/2;
      if(preIdx<0 || preIdx==minIdx) {
        pts[ki]=cx[mm][cfSegNodes[mm][k]];
        // Arc length
        if(k==minIdx) dis[ki]=0.;
        else dis[ki]=dis[ki-1]+(pts[ki]-pts[ki-1]).length();
      }
      idx[k]=ki; 
      if(k<maxIdx) continue; // Collect next points
    }

    // Step b: Define how to smooth the sub-crack
    int n=numPts;             // number of points (>=2)
    int m=(int)(numSegs/2)+2; // number of intervals (>=2)
    int n1=7*m-3;

    // Arries starting from 1
    double* S=new double[n+1]; // arc-length to the first point
    double* X=new double[n+1]; // x indexed from 1
    double* Y=new double[n+1]; // y indexed from 1
    double* Z=new double[n+1]; // z indexed from 1
    for(i=1; i<=n; i++) {
      S[i]=dis[i-1];
      X[i]=pts[i-1].x();
      Y[i]=pts[i-1].y();
      Z[i]=pts[i-1].z();
    }

    int*    g=new int[n+1];    // segID
    int*    j=new int[m+1];    // number of points
    double* s=new double[m+1]; // positions of intervals
    double* ex=new double[n1+1];
    double* ey=new double[n1+1];
    double* ez=new double[n1+1];

    // Positins of the intervals
    s[1]=S[1]-(S[2]-S[1])/50.;
    for(l=2; l<=m; l++) s[l]=s[1]+(S[n]-s[1])/m*(l-1);

    // Number of points in each seg & the segs to which
    // the points belongs
    for(l=1; l<=m; l++) { // Loop over segs
      j[l]=0; // Number of points in the seg
      for(i=1; i<=n; i++) {
        if((l<m  && S[i]>s[l] && S[i]<=s[l+1]) ||
           (l==m && S[i]>s[l] && S[i]<=S[n])) {
          j[l]++; // Number of points in seg l
          g[i]=l; // Seg ID of point i
        }
      }
    }

    // Step c: Smooth the sub-crack points
    if(CubicSpline(n,m,n1,S,X,s,j,ex,ep) &&
       CubicSpline(n,m,n1,S,Y,s,j,ey,ep) &&
       CubicSpline(n,m,n1,S,Z,s,j,ez,ep)) {// Smooth successfully
      for(i=1; i<=n; i++) {
        l=g[i];
        double t=0.,dtdS=0.;
        if(l<m)  {
          t=2*(S[i]-s[l])/(s[l+1]-s[l])-1.;
          dtdS=2./(s[l+1]-s[l]);
        }
        if(l==m) {
          t=2*(S[i]-s[l])/(S[n]-s[l])-1.;
          dtdS=2./(S[n]-s[l]);
        }

        double Xv0,Xv1,Xv2,Xv3,Yv0,Yv1,Yv2,Yv3,Zv0,Zv1,Zv2,Zv3;
        Xv0=ex[7*l-6]; Xv1=ex[7*l-5]; Xv2=ex[7*l-4]; Xv3=ex[7*l-3];
        Yv0=ey[7*l-6]; Yv1=ey[7*l-5]; Yv2=ey[7*l-4]; Yv3=ey[7*l-3];
        Zv0=ez[7*l-6]; Zv1=ez[7*l-5]; Zv2=ez[7*l-4]; Zv3=ez[7*l-3];

        double t0,t1,t2,t3,t0p,t1p,t2p,t3p;
        t0 =1.; t1 =t;    t2 =2*t*t-1.; t3 =4*t*t*t-3*t;
        t0p=0.; t1p=dtdS; t2p=4*t*dtdS; t3p=(12.*t*t-3.)*dtdS;

        V3[i-1].x(Xv1*t1p+Xv2*t2p+Xv3*t3p);
        V3[i-1].y(Yv1*t1p+Yv2*t2p+Yv3*t3p);
        V3[i-1].z(Zv1*t1p+Zv2*t2p+Zv3*t3p);
        pts[i-1].x(Xv0*t0+Xv1*t1+Xv2*t2+Xv3*t3);
        pts[i-1].y(Yv0*t0+Yv1*t1+Yv2*t2+Yv3*t3);
        pts[i-1].z(Zv0*t0+Zv1*t1+Zv2*t2+Zv3*t3);
      }
    }
    else { // Not smooth successfully, use the raw data
      flag=0;
      for(i=0; i<n; i++) {
        Point pt1=(i==0   ? pts[i] : pts[i-1]);
        Point pt2=(i==n-1 ? pts[i] : pts[i+1]);
        V3[i]=TwoPtsDirCos(pt1,pt2);
      }
    }

    delete [] g;
    delete [] j;
    delete [] s;
    delete [] ex;
    delete [] ey;
    delete [] ez;
    delete [] S;
    delete [] X;
    delete [] Y;
    delete [] Z;

    // Step d: Smooth crack-front points and store tangential vectors
    for(i=minIdx;i<=maxIdx;i++) { // Loop over all nodes on the sub-crack 
      int ki=idx[i];
      
      // Smooth crack-front points
      // int ni=cfSegNodes[mm][i];
      // cx[mm][ni]=pts[ki];  

      // Store tangential vectors
      if(minNode==maxNode && (i==minIdx || i==maxIdx)) {
        // for the first and last points (They coincide) of enclosed cracks
        int k1=idx[minIdx];
        int k2=idx[maxIdx];
        Vector averageV3=(V3[k1]+V3[k2])/2.;
        cfSegV3[mm][i]=-averageV3/averageV3.length();
      }
      else {
        cfSegV3[mm][i]=-V3[ki]/V3[ki].length();
      }
    }
    pts.clear();
    idx.clear();
    dis.clear();
    V3.clear();
  } // End of loop over k

  /* Task 2: Calculate normals of crack plane at crack-front nodes
  */
  cfSegV2[mm].clear();
  cfSegV2[mm].resize(cfNodeSize);
  for(k=0; k<cfNodeSize; k++) {
    int node=cfSegNodes[mm][k];
    int preIdx=cfSegPreIdx[mm][k];

    if(preIdx<0) {// Not operated
      Vector v2T=Vector(0.,0.,0.);
      double totalArea=0.;
      for(i=0; i<(int)ce[mm].size(); i++) { //Loop over crack elems
        // Three nodes of the elems
        int n1=ce[mm][i].x();
        int n2=ce[mm][i].y();
        int n3=ce[mm][i].z();
        if(node==n1 || node==n2 || node==n3) {
          // Three points of the triangle
          Point p1=cx[mm][n1];
          Point p2=cx[mm][n2];
          Point p3=cx[mm][n3];
          // Lengths of sides of the triangle
          double a=(p1-p2).length();
          double b=(p1-p3).length();
          double c=(p2-p3).length();
          // Half of perimeter of the triangle
          double s=(a+b+c)/2.;
          // Area of the triangle
          double thisArea=sqrt(s*(s-a)*(s-b)*(s-c));
          // Normal of the triangle
          Vector thisNorm=TriangleNormal(p1,p2,p3);
          // Area-weighted normal vector
          v2T+=thisNorm*thisArea;
          // Total area of crack plane related to the node
          totalArea+=thisArea;
        }
      } // End of loop over crack elems
      v2T/=totalArea;
      cfSegV2[mm][k]=v2T/v2T.length();
    }
    else { // Calculated
      cfSegV2[mm][k]=cfSegV2[mm][preIdx];
    }
  } // End of loop over crack-front nodes

  /* Task 3: Calculate bi-normals of crack plane at crack-front nodes
             and adjust crack-plane normals to make sure the three axes
             are perpendicular to each other.
  */
  cfSegV1[mm].clear();
  cfSegV1[mm].resize(cfNodeSize);
  for(k=0; k<cfNodeSize; k++) {
    Vector V1=Cross(cfSegV2[mm][k],cfSegV3[mm][k]);
    cfSegV1[mm][k]=V1/V1.length();
    Vector V2=Cross(cfSegV3[mm][k],cfSegV1[mm][k]);
    cfSegV2[mm][k]=V2/V2.length();
  }

  return flag;
}

short Crack::CubicSpline(const int& n, const int& m, const int& n1,
                         double x[], double y[], double z[],
                         int j[], double e[], const double& ep)
{
  short flag=1;
  int i,k,n3,l,j1,nk,lk,llk,jj,lly,nnj,mmi,nn,ii,my,jm,ni,nij;
  double h1,h2,xlk,xlk1,a1,a2,a3,a4,t;

  double** f=new double*[n1+1];
  for(i=0; i<n1+1; i++) f[i]=new double[14];

  for(i=1; i<=n1; i++) {
    e[i]=0.;
    for(k=1; k<=13; k++) f[i][k]=0.;
  }

  n3=0;
  for(l=1; l<=m; l++) {
    if(l<m)
      h1=1./(z[l+1]-z[l]);
    else
      h1=1./(x[n]-z[m]);

    j1=j[l];
    for(k=1; k<=j1; k++) {
      nk=n3+k;
      xlk=2.*(x[nk]-z[l])*h1-1.;
      xlk1=xlk*xlk;
      a1=1.;
      a2=xlk;
      a3=2.*xlk1-1.;
      a4=(4.*xlk1-3.)*xlk;
      e[7*l-6]+=a1*y[nk];
      e[7*l-5]+=a2*y[nk];
      e[7*l-4]+=a3*y[nk];
      e[7*l-3]+=a4*y[nk];
      f[7*l-6][7]+=a1*a1;
      f[7*l-5][7]+=a2*a2;
      f[7*l-4][7]+=a3*a3;
      f[7*l-3][7]+=a4*a4;
      f[7*l-6][8]+=a1*a2;
      f[7*l-5][8]+=a2*a3;
      f[7*l-4][8]+=a3*a4;
      f[7*l-6][9]+=a1*a3;
      f[7*l-5][9]+=a2*a4;
      f[7*l-6][10]+=a1*a4;
    }

    f[7*l-5][6]=f[7*l-6][8];
    f[7*l-4][5]=f[7*l-6][9];
    f[7*l-3][4]=f[7*l-6][10];

    f[7*l-4][6]=f[7*l-5][8];
    f[7*l-3][5]=f[7*l-5][9];

    f[7*l-3][6]=f[7*l-4][8];

    f[7*l-6][4]=-0.5;
    f[7*l-4][2]=-0.5;
    f[7*l-5][3]=0.5;
    f[7*l-3][1]=0.5;
    f[7*l-6][11]=0.5;
    f[7*l-5][10]=0.5;
    f[7*l-4][9]=0.5;
    f[7*l-3][8]=0.5;
    f[7*l-5][4]=-h1;
    f[7*l-5][11]=h1;
    f[7*l-4][3]=4.*h1;
    f[7*l-4][10]=4.*h1;
    f[7*l-4][11]=8.*h1*h1;
    f[7*l-4][4]=-8.*h1*h1;
    f[7*l-3][2]=-9.*h1;
    f[7*l-3][9]=9.*h1;
    f[7*l-3][3]=48.*h1*h1;
    f[7*l-3][10]=48.*h1*h1;

    if(l<=m-1) {
      if(l<m-1)
        h2=1./(z[l+2]-z[l+1]);
      else
        h2=1./(x[n]-z[m]);

      f[7*l-2][3]=1.;
      f[7*l-2][4]=1.;
      f[7*l-2][5]=1.;
      f[7*l-2][6]=1.;
      f[7*l-2][11]=1.;
      f[7*l-2][13]=1.;
      f[7*l-2][10]=-1.;
      f[7*l-2][12]=-1.;
      f[7*l-1][3]=2.*h1;
      f[7*l-1][4]=8.*h1;
      f[7*l-1][5]=18.*h1;
      f[7*l-1][10]=-2.*h2;
      f[7*l-1][11]=8.*h2;
      f[7*l-1][12]=-18.*h2;
      f[7*l][3]=16.*h1*h1;
      f[7*l][4]=96.*h1*h1;
      f[7*l][10]=-16.*h2*h2;
      f[7*l][11]=96.*h2*h2;
    }
    n3+=j[l];
  }

  lk=7;
  llk=lk-1;
  for(jj=1; jj<=llk; jj++) {
    lly=lk-jj;
    nnj=n1+1-jj;
    for(i=1; i<=lly; i++) {
      for(k=2; k<=13; k++) f[jj][k-1]= f[jj][k];
      f[jj][13]=0.;
      mmi=14-i;
      f[nnj][mmi]=0.;
    }
  }

  nn=n1-1;
  for(i=1; i<=nn; i++) {
    k=i;
    ii=i+1;
    for(my=ii; my<=lk; my++) {
      if(fabs(f[my][1])<=fabs(f[k][1])) continue;
      k=my;
    }

    if(k!=i) {
      t=e[i];
      e[i]=e[k];
      e[k]=t;
      for(jj=1; jj<=13; jj++) {
        t=f[i][jj];
        f[i][jj]=f[k][jj];
        f[k][jj]=t;
      }
    }

    if(ep>=fabs(f[i][1])) {
      flag=0;
      return flag; // unsuccessful
    }
    else {
      e[i]/=f[i][1];
      for(jj=2; jj<=13; jj++) f[i][jj]/=f[i][1];

      ii=i+1;
      for(my=ii; my<=lk; my++) {
        t=f[my][1];
        e[my]-=t*e[i];
        for(jj=2; jj<=13; jj++) f[my][jj-1]=f[my][jj]-t*f[i][jj];
        f[my][13]=0.;
      }

      if(lk==n1) continue;
      lk++;
    }
  }

  e[n1]/=f[n1][1];
  jm=2;
  nn=n1-1;
  for(i=1; i<=nn; i++) {
    ni=n1-i;
    for(jj=2; jj<=jm; jj++) {
      nij=ni-1+jj;
      e[ni]-=f[ni][jj]*e[nij];
    }
    if(jm==13) continue;
    jm++;
  }

  return flag;
}

