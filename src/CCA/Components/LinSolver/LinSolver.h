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
 *  Viwer.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef SCIRun_LinSolver_h
#define SCIRun_LinSolver_h


#include <Core/CCA/spec/cca_sidl.h>
#include "Matrix.h"

//namespace SCIRun {
  

#define myGoPort LinSolverGoPort

class LinSolver;

class myField2DPort : public virtual gov::cca::ports::Field2DPort {
public:
   virtual ~myField2DPort(){}
   void setParent(LinSolver *com){this->com=com;}
   SIDL::array1<double>  getField();
 private:
   LinSolver *com;
};

class myGoPort : public virtual gov::cca::ports::GoPort {
public:
  virtual ~myGoPort(){}
  virtual int go();
  void setParent(LinSolver *com){this->com=com;}
  LinSolver *com;
};


class LinSolver: public gov::cca::Component{
                
  public:
    LinSolver();
    virtual ~LinSolver();
    gov::cca::Services::pointer getServices(){return services;}
    virtual void setServices(const gov::cca::Services::pointer& svc);
    bool jacobi(const gov::cca::Matrix::pointer &A, 
		const SIDL::array1<double> &b);
    SIDL::array1<double> solution;
 private:

    LinSolver(const LinSolver&);
    LinSolver& operator=(const LinSolver&);
    myGoPort goPort;
    myField2DPort fieldPort;
    gov::cca::Services::pointer services;
  };
//}




#endif
