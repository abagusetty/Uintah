/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  FEM.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/FEM/FEM.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>
#include "Matrix.h"


using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_FEM()
{
  return gov::cca::Component::pointer(new FEM());
}


FEM::FEM()
{
  uiPort.setParent(this);
  goPort.setParent(this);
}

FEM::~FEM()
{
  cerr << "called ~FEM()\n";
}

void FEM::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  myGoPort::pointer gop(&goPort);
  myPDEMatrixPort::pointer matrixp(&matrixPort);
  svc->addProvidesPort(uip,"ui","gov.cca.UIPort", props);
  svc->addProvidesPort(gop,"go","gov.cca.GoPort", props);
  svc->addProvidesPort(matrixp,"matrix","gov.cca.PDEMatrixPort", props);
  svc->registerUsesPort("mesh","gov.cca.MeshPort", props);
  svc->registerUsesPort("pde","gov.cca.PDEDescriptionPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}


//Computes the 1st order differentiation arrays for linear triangles.
// x, y 3 x 1 arries
void FEM::diffTriangle(double b[3], double c[3], double &area,
		       const double x[3], const double y[3])
{
  double A2=(x[1]*y[2]-x[2]*y[1])-(x[0]*y[2]-x[2]*y[0])+(x[0]*y[1]-x[1]*y[0]);
  area=A2/2;
  for(int i=0; i<3; i++){
    int i1=(i+1)%3;
    int i2=(i+2)%3;
    b[i]=(y[i1]-y[i2])/A2;
    c[i]=(x[i2]-x[i1])/A2;
  }
}

//define double source(int index) here



//
void FEM::localMatrices(double A[3][3], double f[3], 
			const double x[3], const double y[3])
{
  double b[3], c[3], area;
  diffTriangle(b,c,area,x,y);
  for(int i=0; i<3; i++){
    int i1=(i+1)%3;
    int i2=(i+2)%3;
    f[i]=area*(2*source(i)+source(i1)+source(i2))/12;        
    for(int j=0; j<3; j++){
        A[i][j]=area*(b[i]*b[j]+c[i]*c[j]);
    }
  }
}



//create the global matrices from the local matrices
void FEM::globalMatrices(Matrix &Ag, std::vector<double> fg,
			 const std::vector<double> nodes1d,
			 const std::vector<int> &tmesh1d)
{
  
 int N=fg.size();
 for(int i=0; i<N; i++){
   fg[i]=0;
   for(int j=0; j<N; j++){
     Ag.setElement(i,j,0);
   }
 }

 //get number of triangles 
 int Ntri=tmesh1d.size()/3;

 for(int i=0; i<Ntri; i++){
   double x[3], y[3];
   for(int j=0; j<3; j++){
     x[j]=nodes1d[tmesh1d[i*3+j]*2];
     y[j]=nodes1d[tmesh1d[i*3+j]*2+1];
   }
   double A[3][3], f[3];
   localMatrices(A,f,x,y);
   for(int row=0; row<3; row++){
     int grow=tmesh1d[i*3+row];
     if(isConst(grow)){
       fg[grow]+=f[row];
       for(int col=0; col<3; col++){
	 int gcol=tmesh1d[i*3+col];
	 if(isConst(gcol)){
	   Ag.setElement(grow,gcol,Ag.getElement(grow,gcol)+A[row][col]);
	 }
	 else{
	   //u(gcol) is the constant boundary value
	   fg[grow]-=A[row][col]*boundary(gcol);
	 }
       }
     }
   }
 }
 //put into global function
 for(int grow=0; grow<N; grow++){
   if(isConst(grow)){
     for(int k=0; k<N; k++){
       Ag.setElement(grow,k,0);
       Ag.setElement(k,grow,0);
     }
     Ag.setElement(grow,grow,1);
     fg[grow]= boundary(grow); 
   }
 }
}

double FEM::source(int index)
{
  return 0;
}

double FEM::boundary(int index)
{
  return 0;
}

bool FEM::isConst(int index)
{
  return false;
}


void myUIPort::ui() 
{
  QMessageBox::warning(0, "FEM", "You have clicked the UI button!");
}


int myGoPort::go() 
{
  QMessageBox::warning(0, "FEM", "Go ...");
  return 0;
}
 

gov::cca::Matrix::pointer myPDEMatrixPort::getMatrix()
{
  cerr<<"getMatrix not implemented"<<endl;
  return gov::cca::Matrix::pointer(0);
}

CIA::array1<double> myPDEMatrixPort::getVector()
{
  cerr<<"getVector not implemented"<<endl;
  CIA::array1<double > v;
  return v;
}
