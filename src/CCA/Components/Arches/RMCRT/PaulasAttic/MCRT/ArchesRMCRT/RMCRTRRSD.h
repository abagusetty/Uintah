/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef RMCRTRRSD_h
#define RMCRTRRSD_h

#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTFactory.h>

namespace Uintah {
  class RMCRTRRSD:public RMCRTFactory{

public:
  RMCRTRRSD();

  virtual ~RMCRTRRSD();

  void ToArray(int size, double *array, char *_argv);
 
  double MeshSize(int &Nchalf, double &Lhalf, double &ratio);

  int RMCRTsolver(const int &i_n,
      const int &j_n,
      const int &k_n,
      const int &theta_n,
      const int &phi_);
  
  
};

}

#endif
