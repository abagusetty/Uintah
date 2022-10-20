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

#pragma once
// #ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_HPP
// #define CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_HPP

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
// #include <Core/Geometry/GPUVector.h>
#include <Core/Grid/Task.h>

#include <oneapi/mkl/rng/device.hpp>
#include <sci_defs/gpu_defs.h>

// #include <oneapi/dpl/random>
// hypre_HandleComputeStream(hypre_handle())->parallel_for(
//    sycl::range<1>(n), [ = ](sycl::item<1> idx)
// {
//    std::uint64_t offset = idx.get_linear_id();
//    oneapi::dpl::default_engine engine(1234ULL, offset);
//    oneapi::dpl::uniform_real_distribution<T> distr(0., 1.);
//    urand[idx] = distr(engine);
// }).wait();

namespace Uintah {

typedef sycl::int3 GPUIntVector;
typedef sycl::double3 GPUVector;
typedef sycl::double3 GPUPoint;

//______________________________________________________________________
//
constexpr int d_MAXLEVELS = 5; // FIX ME!
constexpr int d_MAX_RAYS = 500;
constexpr int d_flowCell = -1; // HARDWIRED!
//__________________________________

//
class unifiedMemory { // this should be moved upstream
public:               // This only works for cuda > 6.X
#if 0                 // turn off until titan has cuda > 6.0 installed
  void *operator new(size_t len)
  {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
  }

  void operator delete(void *ptr)
  {
    cudaFree(ptr);
  }

  void *operator new[] (size_t len)
  {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
  }

  void operator delete[] (void* ptr)
  {
    cudaFree(ptr);
  }
#endif
};
#if 0 // turn off until titan has cuda > 6.0 installed
//______________________________________________________________________
//
//  http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
// String Class for unified managed Memory
class GPUString : public unifiedMemory
{
  int length;
  char *data;

  public:
    GPUString() : length(0), data(0) {}
    // Constructor for C-GPUString initializer
    GPUString(const char *s) : length(0), data(0)
    {
      _realloc(strlen(s));
      strcpy(data, s);
    }

    // Copy constructor
    GPUString(const GPUString& s) : length(0), data(0)
    {
      _realloc(s.length);
      strcpy(data, s.data);
    }

    // destructor
    ~GPUString() {
      cudaFree(data);
    }

    // Assignment operator
    GPUString& operator=(const char* s)
    {
      _realloc(strlen(s));
      strcpy(data, s);
      return *this;
    }

    // Element access (from host or device)
    char& operator[](int pos)
    {
      return data[pos];
    }

    // C-String access host or device
    const char* c_str() const
    {
      return data;
    }

  private:
    void _realloc(int len)
    {
      cudaFree(data);
      length = len;
      cudaMallocManaged(&data, length+1);
    }
};
#endif
//______________________________________________________________________
//
struct varLabelNames : public unifiedMemory {
public:
#if 0 // turn off until titan has cuda > 6.0 installed
    GPUString divQ;
    GPUString abskg;
    GPUString sigmaT4;
    GPUString celltype;
    GPUString VRFlux;
    GPUString boundFlux;
    GPUString radVolQ;
#endif
};

//______________________________________________________________________
//
struct patchParams {
  sycl::int3 lo;   // cell low index not including extra or ghost cells
  sycl::int3 hi;   // cell high index not including extra or ghost cells
  sycl::int3 loEC; // low index including extraCells
  sycl::int3 hiEC; // high index including extraCells
  GPUVector dx;    // cell spacing
  int ID;          // patch ID
};

//______________________________________________________________________
//
struct levelParams {
  sycl::double3 Dx;    // cell spacing
  sycl::int3 regionLo; // never use these regionLo/Hi in the kernel
  sycl::int3
      regionHi; // they vary on every patch and must be passed into the kernel
  bool hasFinerLevel;
  int index; // level index
  sycl::int3 refinementRatio;
  sycl::double3 anchor; // level anchor

  //__________________________________
  //  GPU version of level::getCellPosition()
  __attribute__((always_inline)) sycl::double3
  getCellPosition(const sycl::int3 &cell) {
    // Issue: cannot convert between vector values of different size
    // ('sycl::vec<double, 3>::vector_t' (aka 'double
    // __attribute__((ext_vector_type(3)))') and 'sycl::vec<int, 3>::vector_t'
    // (aka 'int __attribute__((ext_vector_type(3)))'))

    // sycl::double3 result = anchor + (Dx * cell) + (0.5 * Dx);

    sycl::double3 result;
    result.x() = anchor.x() + (Dx.x() * cell.x()) + (0.5 * Dx.x());
    result.y() = anchor.y() + (Dx.y() * cell.y()) + (0.5 * Dx.y());
    result.z() = anchor.z() + (Dx.z() * cell.z()) + (0.5 * Dx.z());
    return result;
  }
  __attribute__((always_inline)) sycl::double3
  getCellPosition(const sycl::int3 &cell) const {
    // Issue: cannot convert between vector values of different size
    // ('sycl::vec<double, 3>::vector_t' (aka 'double
    // __attribute__((ext_vector_type(3)))') and 'sycl::vec<int, 3>::vector_t'
    // (aka 'int __attribute__((ext_vector_type(3)))'))

    // sycl::double3 result = anchor + (Dx * cell) + (0.5 * Dx);

    sycl::double3 result;
    result.x() = anchor.x() + (Dx.x() * cell.x()) + (0.5 * Dx.x());
    result.y() = anchor.y() + (Dx.y() * cell.y()) + (0.5 * Dx.y());
    result.z() = anchor.z() + (Dx.z() * cell.z()) + (0.5 * Dx.z());
    return result;
  }

  //__________________________________
  //  GPU version of level::mapCellToCoarser()
  __attribute__((always_inline)) sycl::int3
  mapCellToCoarser(const sycl::int3 &idx) {
    sycl::int3 ratio = idx / refinementRatio;

    // If the fine cell index is negative
    // you must add an offset to get the right
    // coarse cell. -Todd
    sycl::int3 offset(0, 0, 0);

    if (idx.x() < 0 && refinementRatio.x() > 1) {
      offset.x() = (int)fmod((double)idx.x(), (double)refinementRatio.x());
    }
    if (idx.y() < 0 && refinementRatio.y() > 1) {
      offset.y() = (int)fmod((double)idx.y(), (double)refinementRatio.y());
    }
    if (idx.z() < 0 && refinementRatio.z() > 1) {
      offset.z() = (int)fmod((double)idx.z(), (double)refinementRatio.z());
    }
    return ratio + offset;
  }
  __attribute__((always_inline)) sycl::int3
  mapCellToCoarser(const sycl::int3 &idx) const {
    sycl::int3 ratio = idx / refinementRatio;

    // If the fine cell index is negative
    // you must add an offset to get the right
    // coarse cell. -Todd
    sycl::int3 offset(0, 0, 0);

    if (idx.x() < 0 && refinementRatio.x() > 1) {
      offset.x() = (int)fmod((double)idx.x(), (double)refinementRatio.x());
    }
    if (idx.y() < 0 && refinementRatio.y() > 1) {
      offset.y() = (int)fmod((double)idx.y(), (double)refinementRatio.y());
    }
    if (idx.z() < 0 && refinementRatio.z() > 1) {
      offset.z() = (int)fmod((double)idx.z(), (double)refinementRatio.z());
    }
    return ratio + offset;
  }
};

//______________________________________________________________________
//
struct BoundingBox {
  sycl::double3 lo;
  sycl::double3 hi;

  bool inside(sycl::double3 &p) {
    return ((p.x() >= lo.x()) && (p.y() >= lo.y()) && (p.z() >= lo.z()) &&
            (p.x() <= hi.x()) && (p.y() <= hi.y()) && (p.z() <= hi.z()));
  }
};

//______________________________________________________________________
//
struct gridParams {
  int maxLevels;
  struct BoundingBox domain_BB;
};

//______________________________________________________________________
//
struct RMCRT_flags {
  bool modifies_divQ;
  bool solveDivQ;
  bool allowReflect;
  bool solveBoundaryFlux;
  bool CCRays;
  bool usingFloats; // if the communicated vars (sigmaT4 & abskg) are floats

  double sigma;     // StefanBoltzmann constant
  double sigmaScat; // scattering coefficient
  double threshold;

  int nDivQRays;        // number of rays per cell used to compute divQ
  int nFluxRays;        // number of boundary flux rays
  int nRaySteps;        // number of ray steps taken
  int whichROI_algo;    // which Region of Interest algorithm
  int rayDirSampleAlgo; // which ray direction sampling algorithm (Monte-Carlo
                        // or Latin-Hyper_Cube)
  unsigned int startCell{0};
  unsigned int endCell{0};
};

//__________________________________
//  Struct for managing the boundary faces
struct BoundaryFaces {
  BoundaryFaces() : nFaces(0) {}

  int nFaces;       // number of faces
  int faceArray[6]; // vector of faces

  // add Face to array
  void addFace(int f) {
    faceArray[nFaces] = f;
    nFaces++;
  }

  // returns the number of faces
  int size() { return nFaces; }

  // print facesArray
  void print(int tid) {
    for (int f = 0; f < nFaces; f++) {
      printf("  tid: %i face[%i]: %i\n", tid, f, faceArray[f]);
    }
  }
};

enum rayDirSampleAlgorithm { NAIVE, LATIN_HYPER_CUBE };
//
enum DIR { X = 0, Y = 1, Z = 2, NONE = -9 };
//           -x      +x       -y       +y     -z     +z
enum FACE {
  EAST = 0,
  WEST = 1,
  NORTH = 2,
  SOUTH = 3,
  TOP = 4,
  BOT = 5,
  nFACES = 6
};
//
enum ROI_algo {
  fixed,   // user specifies fixed low and high point for a bounding box
  dynamic, // user specifies thresholds that are used to dynamically determine
           // ROI
  patch_based,      // The patch extents + halo are the ROI
  boundedRayLength, // the patch extents + boundedRayLength/Dx are the ROI
  entireDomain      // The ROI is the entire computatonal Domain
};

//______________________________________________________________________
//
sycl::double3
findRayDirectionDevice(oneapi::mkl::rng::device::uniform<double> *distr,
                       oneapi::mkl::rng::device::philox4x32x10<1> *engine);
//______________________________________________________________________
//
sycl::double3 findRayDirectionHyperCubeDevice(
    oneapi::mkl::rng::device::uniform<double> *distr,
    oneapi::mkl::rng::device::philox4x32x10<1> *engine, const int nDivQRays,
    const int bin_i, const int bin_j);
//______________________________________________________________________
//
void randVectorDevice(int int_array[], const int size,
                      oneapi::mkl::rng::device::uniform<double> *distr,
                      oneapi::mkl::rng::device::philox4x32x10<1> *engine);

//______________________________________________________________________
//
void rayDirection_cellFaceDevice(
    oneapi::mkl::rng::device::uniform<double> *distr,
    oneapi::mkl::rng::device::philox4x32x10<1> *engine,
    const sycl::int3 &origin, const sycl::int3 &indexOrder,
    const sycl::int3 &signOrder, const int iRay, sycl::double3 &directionVector,
    double &cosTheta);
//______________________________________________________________________
//
void rayDirectionHyperCube_cellFaceDevice(
    oneapi::mkl::rng::device::uniform<double> *distr,
    oneapi::mkl::rng::device::philox4x32x10<1> *engine,
    const sycl::int3 &origin, const int3 &indexOrder, const int3 &signOrder,
    const int iRay, sycl::double3 &dirVector, double &cosTheta, const int bin_i,
    const int bin_j, const int nFluxRays);

//______________________________________________________________________
//
void rayLocation_cellFaceDevice(
    oneapi::mkl::rng::device::uniform<double> *distr,
    oneapi::mkl::rng::device::philox4x32x10<1> *engine, const int face,
    const sycl::double3 Dx, const sycl::double3 CC_pos,
    sycl::double3 &rayOrigin);
//______________________________________________________________________
//
bool has_a_boundaryDevice(const sycl::int3 &c,
                          sycl::device_ptr<int> GPUGridVariable_ptr,
                          sycl::int3 &GPUGridVariable_size,
                          sycl::int3 &GPUGridVariable_offset,
                          BoundaryFaces &boundaryFaces);
bool has_a_boundaryDevice(const sycl::int3 &c,
                          const GPUGridVariable<int> &celltype,
                          BoundaryFaces &boundaryFaces);
//______________________________________________________________________
//
void raySignStepDevice(sycl::double3 &sign, int cellStep[],
                       const sycl::double3 &inv_direction_vector);
//______________________________________________________________________
//
// bool containsCellDevice(const sycl::int3& fineLevel_ROI_Lo,
//                         const sycl::int3& fineLevel_ROI_Hi,
// 			const sycl::int3& cell,
//                         const int dir);
//______________________________________________________________________
//
void reflectDevice(double &fs, sycl::int3 &cur, sycl::int3 &prevCell,
                   const double abskg, bool &in_domain, int &step, double &sign,
                   double &ray_direction);
//______________________________________________________________________
//

template <class T>
SYCL_EXTERNAL void
updateSumIDevice(levelParams level, sycl::double3 &ray_direction,
                 sycl::double3 &ray_location, const sycl::int3 &origin,
                 const sycl::double3 &Dx, sycl::device_ptr<T> sigmaT4OverPi_ptr,
                 sycl::int3 sigmaT4OverPi_size, sycl::int3 sigmaT4OverPi_offset,
                 sycl::device_ptr<T> abskg_ptr, sycl::int3 abskg_size,
                 sycl::int3 abskg_offset, sycl::device_ptr<int> celltype_ptr,
                 sycl::int3 celltype_size, sycl::int3 celltype_offset,
                 double &sumI, RMCRT_flags RT_flags);

//______________________________________________________________________
//  Multi-level

template <class T>
void updateSumI_MLDevice(
    sycl::double3 &ray_direction, sycl::double3 &ray_location,
    const sycl::int3 &origin, gridParams gridP,
    const sycl::int3 &fineLevel_ROI_Lo, const sycl::int3 &fineLevel_ROI_Hi,
    const int3 *regionLo, const int3 *regionHi,
    sycl::device_ptr<T> *sigmaT4OverPi_ptr, sycl::int3 *sigmaT4OverPi_size,
    sycl::int3 *sigmaT4OverPi_offset, sycl::device_ptr<T> *abskg_ptr,
    sycl::int3 *abskg_size, sycl::int3 *abskg_offset,
    sycl::device_ptr<int> *celltype_ptr, sycl::int3 *celltype_size,
    sycl::int3 *celltype_offset, double &sumI, RMCRT_flags RT_flags,
    const levelParams *d_levels);

//______________________________________________________________________
//
void setupRandNumsSeedAndSequences(int numStates, unsigned long long patchID,
                                   unsigned long long curTimestep);

//______________________________________________________________________
//
double randDblExcDevice(oneapi::mkl::rng::device::uniform<double> *distr,
                        oneapi::mkl::rng::device::philox4x32x10<1> *engine);

double randDblDevice(oneapi::mkl::rng::device::uniform<double> *distr,
                     oneapi::mkl::rng::device::philox4x32x10<1> *engine);

//______________________________________________________________________
//
bool isDbgCellDevice(sycl::int3 me);

//______________________________________________________________________
//
template <class T>
void launchRayTraceKernel(DetailedTask *dtask, sycl::range<2> &dimGrid,
                          sycl::range<2> &dimBlock, const int matlIndex,
                          levelParams level, patchParams patch,
                          gpuStream_t *stream, RMCRT_flags RT_flags,
                          int curTimestep, GPUDataWarehouse *abskg_gdw,
                          GPUDataWarehouse *sigmaT4_gdw,
                          GPUDataWarehouse *celltype_gdw,
                          GPUDataWarehouse *new_gdw);
//______________________________________________________________________
//
template <class T>
void launchRayTraceDataOnionKernel(
    DetailedTask *dtask, sycl::range<1> &dimGrid, sycl::range<1> &dimBlock,
    int matlIndex, patchParams patchP, gridParams gridP, levelParams *levelP,
    sycl::int3 fineLevel_ROI_Lo, sycl::int3 fineLevel_ROI_Hi,
    gpuStream_t *stream, unsigned int deviceID, RMCRT_flags RT_flags,
    int curTimestep, GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *celltype_gdw, GPUDataWarehouse *new_gdw);

} // end namespace Uintah

// #endif
