/*
 * The MIT License
 *
 * Copyright (c) 2013-2021 The University of Utah
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

#ifndef UINTAH_GPUSTENCIL7_H
#define UINTAH_GPUSTENCIL7_H

#if defined(HAVE_CUDA) || defined(HAVE_HIP) || defined(HAVE_SYCL)
#include <sci_defs/gpu_defs.h>
#else
#include <array>
#endif

namespace Uintah {

#ifdef HAVE_SYCL

  // ABB: The last index should never be used, since we only need
  //      7-point stencil in SYCL
  typedef sycl::vec<double, 8> GPUStencil7;
  // Mapping: [w, e, s, n, b, t, p] --> [0, 1, 2, 3, 4, 5, 6, X]

#elif defined(HAVE_CUDA) || defined(HAVE_HIP)

  struct GPUStencil7 {

    // The order of this is designed to match the order of faces in Patch
    // Do not change it!
    //     -x +x -y +y -z +z
    double  w, e, s, n, b, t;

    // diagonal term
    double p;

    HOST_DEVICE double& operator[](int index) {
      CHECK_RANGE(index, 0, 7);
      return (&w)[index];
    }
    HOST_DEVICE const double& operator[](int index) const {
      CHECK_RANGE(index, 0, 7);
      return (&w)[index];
    }
    HOST_DEVICE void initialize(double a){
      w = a;
      e = a;
      s = a;
      n = a;
      b = a;
      t = a;
      p = a;
    }
  };

#else

  typedef std::array<double, 7> GPUStencil7;

#endif // HAVE_SYCL

}

#endif // UINTAH_GPUSTENCIL7_H
