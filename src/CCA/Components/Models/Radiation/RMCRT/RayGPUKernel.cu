/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <CCA/Components/Schedulers/GPUMemoryPool.h>

#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUStencil7.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Util/GPU.h>

#include <sci_defs/uintah_defs.h>

#define DEBUG -9 // 1: divQ, 2: boundFlux, 3: scattering
// #define FIXED_RANDOM_NUM        // also edit in
// src/Core/Math/MersenneTwister.h to compare with Ray:CPU

#define FIXED_RAY_DIR                                                          \
  -9 // Sets ray direction.  1: (0.7071,0.7071, 0), 2: (0.7071, 0, 0.7071), 3:
     // (0, 0.7071, 0.7071)
     //                      4: (0.7071, 0.7071, 7071), 5: (1,0,0)  6: (0, 1,
     //                      0),   7: (0,0,1)
#define SIGN 1 // Multiply the FIXED_RAY_DIRs by value

#ifdef HAVE_CUDA
#define gpuMemcpyToSymbolAsync cudaMemcpyToSymbolAsync
#define gpurand curand
#define gpurand_init curand_init
#elif defined(HAVE_HIP)
#define gpuMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define gpurand hiprand
#define gpurand_init hiprand_init
#endif

//__________________________________
//  To Do
//  - Investigate using multiple GPUs per node.
//  - Implement fixed and dynamic ROI.
//  - dynamic block size?
//  - Implement labelNames in unified memory.
//  - investigate the performance with different patch configurations
//  - deterministic random numbers
//  - Ray steps

//__________________________________
//
//  To use cuda-gdb on a single GPU you must set the environmental variable
//  CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
//
// mpirun -np 1 xterm -e cuda-gdb sus -gpu -nthreads 2 <args>
//__________________________________

namespace Uintah {

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
template <typename T>
__global__ void
rayTraceKernel(dim3 dimGrid, dim3 dimBlock, const int matl, levelParams level,
               patchParams patch, gpurandState *randNumStates,
               RMCRT_flags RT_flags, int curTimeStep,
               GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
               GPUDataWarehouse *cellType_gdw,
               GPUDataWarehouse *new_gdw) {

  // Not used right now
  //  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y
  //  * blockIdx.z; int threadID = threadIdx.x +  blockDim.x * threadIdx.y +
  //  (blockDim.x * blockDim.y) * threadIdx.z;

  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x + patch.loEC.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y + patch.loEC.y;

  const GPUGridVariable<T> sigmaT4OverPi;
  const GPUGridVariable<T> abskg; // Need to use getRegion() to get the data
  const GPUGridVariable<int> cellType;

  GPUGridVariable<double> divQ;
  GPUGridVariable<GPUStencil7> boundFlux;
  GPUGridVariable<double> radiationVolQ;

  //  sigmaT4_gdw->print();

  sigmaT4_gdw->getLevel(sigmaT4OverPi, "sigmaT4", matl, level.index);
  cellType_gdw->getLevel(cellType, "cellType", matl, level.index);

  if (RT_flags.usingFloats) {
    abskg_gdw->getLevel(abskg, "abskgRMCRT", matl, level.index);
  } else {
    abskg_gdw->getLevel(abskg, "abskg", matl, level.index);
  }

  if (RT_flags.modifies_divQ) {
    new_gdw->getModifiable(divQ, "divQ", patch.ID, matl);
    new_gdw->getModifiable(boundFlux, "RMCRTboundFlux", patch.ID, matl);
    new_gdw->getModifiable(radiationVolQ, "radiationVolq", patch.ID, matl);
  } else {
    // these should be allocateAndPut() calls
    new_gdw->get(divQ, "divQ", patch.ID, matl);
    new_gdw->get(boundFlux, "RMCRTboundFlux", patch.ID, matl);
    new_gdw->get(radiationVolQ, "radiationVolq", patch.ID, matl);

    // Extra Cell Loop
    if ((tidX >= patch.loEC.x) && (tidY >= patch.loEC.y) &&
        (tidX < patch.hiEC.x) &&
        (tidY < patch.hiEC.y)) { // patch boundary check
#pragma unroll
      for (int z = patch.loEC.z; z < patch.hiEC.z;
           z++) { // loop through z slices
        GPUIntVector c = make_int3(tidX, tidY, z);
        divQ[c] = 0.0;
        radiationVolQ[c] = 0.0;
        boundFlux[c] = {0.0};
      }
    }
  }

  bool doLatinHyperCube = (RT_flags.rayDirSampleAlgo == LATIN_HYPER_CUBE);

  const int nFluxRays = RT_flags.nFluxRays; // for readability

  // This rand_i array is only needed for LATIN_HYPER_CUBE scheme
  const int size = 1000;
  int rand_i[size]; // Give it a buffer room of 1000.  But we should only use
                    // nFluxRays items in it. Hopefully this 1000 will always be
                    // greater than nFluxRays.
                    // TODO, a 4D array is probably better here (x,y,z, ray#),
                    // saves on memory (no unused buffer) and computation time
                    // (don't need to compute the rays twice)
  if (nFluxRays > size) {
    printf("\n\n\nERROR!  rayTraceKernel() - Cannot have more rays than the "
           "rand_i array size.  nFluxRays is %d, size of the array is.%d\n\n\n",
           nFluxRays, size);
    // We have to return, otherwise the upcoming math in
    // rayDirectionHyperCube_cellFaceDevice will generate nan values.
    return;
  }

  //______________________________________________________________________
  //           R A D I O M E T E R
  //______________________________________________________________________
  // TO BE FILLED IN

  //______________________________________________________________________
  //          B O U N D A R Y F L U X
  //______________________________________________________________________
  setupRandNumsSeedAndSequences(randNumStates,
                                (dimGrid.x * dimGrid.y * dimGrid.z *
                                 dimBlock.x * dimBlock.y * dimBlock.z),
                                patch.ID, curTimeStep);

  if (RT_flags.solveBoundaryFlux) {

    __shared__ int3 dirIndexOrder[6];
    __shared__ int3 dirSignSwap[6];

    //_____________________________________________
    //   Ordering for Surface Method
    // This block of code is used to properly place ray origins, and orient ray
    // directions onto the correct face.  This is necessary, because by default,
    // the rays are placed and oriented onto a default face, then require
    // adjustment onto the proper face.
    dirIndexOrder[EAST] = make_int3(2, 1, 0);
    dirIndexOrder[WEST] = make_int3(2, 1, 0);
    dirIndexOrder[NORTH] = make_int3(0, 2, 1);
    dirIndexOrder[SOUTH] = make_int3(0, 2, 1);
    dirIndexOrder[TOP] = make_int3(0, 1, 2);
    dirIndexOrder[BOT] = make_int3(0, 1, 2);

    // Ordering is slightly different from 6Flux since here, rays pass through
    // origin cell from the inside faces.
    dirSignSwap[EAST] = make_int3(-1, 1, 1);
    dirSignSwap[WEST] = make_int3(1, 1, 1);
    dirSignSwap[NORTH] = make_int3(1, -1, 1);
    dirSignSwap[SOUTH] = make_int3(1, 1, 1);
    dirSignSwap[TOP] = make_int3(1, 1, -1);
    dirSignSwap[BOT] = make_int3(1, 1, 1);
    __syncthreads();
    //__________________________________
    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if ((tidX >= patch.lo.x) && (tidY >= patch.lo.y) && (tidX < patch.hi.x) &&
        (tidY < patch.hi.y)) { // patch boundary check
#pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices

        GPUIntVector origin = make_int3(tidX, tidY, z); // for each thread

        // get a new set of random numbers
        if (doLatinHyperCube) {
          randVectorDevice(rand_i, nFluxRays, randNumStates);
        }

        //boundFlux[origin].initialize(0.0);
        boundFlux[origin] = {0.0};

        BoundaryFaces boundaryFaces;

        // which surrounding cells are boundaries
        boundFlux[origin].p =
            has_a_boundaryDevice(origin, cellType, boundaryFaces);

        GPUPoint CC_pos = level.getCellPosition(origin);

//__________________________________
// Loop over boundary faces of the cell and compute incident radiative flux
#pragma unroll
        for (int i = 0; i < boundaryFaces.size(); i++) {

          int RayFace = boundaryFaces.faceArray[i];
          int UintahFace[6] = {WEST, EAST, SOUTH, NORTH, BOT, TOP};

          double sumI = 0;
          double sumProjI = 0;
          double sumI_prev = 0;
          double sumCosTheta = 0; // used to force sumCosTheta/nRays == 0.5 or
                                  // sum (d_Omega * cosTheta) == pi

//__________________________________
// Flux ray loop
#pragma unroll
          for (int iRay = 0; iRay < nFluxRays; iRay++) {

            GPUVector direction_vector;
            GPUVector rayOrigin;
            double cosTheta;

            if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
              rayDirectionHyperCube_cellFaceDevice(
                  randNumStates, origin, dirIndexOrder[RayFace],
                  dirSignSwap[RayFace], iRay, direction_vector, cosTheta,
                  rand_i[iRay], iRay, nFluxRays);
            } else {
              rayDirection_cellFaceDevice(
                  randNumStates, origin, dirIndexOrder[RayFace],
                  dirSignSwap[RayFace], iRay, direction_vector, cosTheta);
            }

            rayLocation_cellFaceDevice(randNumStates, RayFace, patch.dx, CC_pos,
                                       rayOrigin);

            updateSumIDevice<T>(level, direction_vector, rayOrigin, origin,
                                patch.dx, sigmaT4OverPi, abskg, cellType, sumI,
                                randNumStates, RT_flags);

            sumProjI +=
                cosTheta * (sumI - sumI_prev); // must subtract sumI_prev, since
                                               // sumI accumulates intensity

            sumCosTheta += cosTheta;

            sumI_prev = sumI;

          } // end of flux ray loop

          sumProjI = sumProjI * (double)nFluxRays / sumCosTheta /
                     2.0; // This operation corrects for error in the first
                          // moment over a half range of the solid angle (Modest
                          // Radiative Heat Transfer page 545 1st edition)
          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          boundFlux[origin][face] = sumProjI * 2 * M_PI / (double)nFluxRays;

        } // boundary faces loop
      }   // z slices loop
    }     // X-Y Thread loop
  }

  //______________________________________________________________________
  //         S O L V E   D I V Q
  //______________________________________________________________________
  // Setup the original seeds so we can get the same random numbers again.
  setupRandNumsSeedAndSequences(randNumStates,
                                (dimGrid.x * dimGrid.y * dimGrid.z *
                                 dimBlock.x * dimBlock.y * dimBlock.z),
                                patch.ID, curTimeStep);

  if (RT_flags.solveDivQ) {
    const int nDivQRays = RT_flags.nDivQRays; // for readability

    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if ((tidX >= patch.lo.x) && (tidY >= patch.lo.y) && (tidX < patch.hi.x) &&
        (tidY < patch.hi.y)) { // patch boundary check
#pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices

        GPUIntVector origin = make_int3(tidX, tidY, z); // for each thread

        // Get the same set of random numbers as we had before.  We need the
        // same rays.
        if (doLatinHyperCube) {
          randVectorDevice(rand_i, nFluxRays, randNumStates);
        }

        double sumI = 0;
        GPUPoint CC_pos = level.getCellPosition(origin);

        // don't compute in intrusions and walls
        if (cellType[origin] != d_flowCell) {
          continue;
        }

//__________________________________
// ray loop
#pragma unroll

        for (int iRay = 0; iRay < nDivQRays; iRay++) {

          GPUVector direction_vector;
          if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
            direction_vector = findRayDirectionHyperCubeDevice(
                randNumStates, nDivQRays, rand_i[iRay], iRay);
          } else { // Naive Monte-Carlo sampling
            direction_vector = findRayDirectionDevice(randNumStates);
          }

          GPUVector rayOrigin =
              rayOriginDevice(randNumStates, CC_pos, patch.dx, RT_flags.CCRays);

          updateSumIDevice<T>(level, direction_vector, rayOrigin, origin,
                              patch.dx, sigmaT4OverPi, abskg, cellType, sumI,
                              randNumStates, RT_flags);
        } // Ray loop

        //__________________________________
        //  Compute divQ
        divQ[origin] = -4.0 * M_PI * abskg[origin] *
                       (sigmaT4OverPi[origin] - (sumI / RT_flags.nDivQRays));

        // radiationVolq is the incident energy per cell (W/m^3) and is
        // necessary when particle heat transfer models (i.e. Shaddix) are used
        radiationVolQ[origin] =
            4.0 * M_PI * abskg[origin] * (sumI / RT_flags.nDivQRays);

      } // end z-slice loop
    }   // end domain boundary check
  }     // solve divQ
} // end ray trace kernel

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer data onion kernel
//---------------------------------------------------------------------------
// hard-wired for 2-levels now, but this should be fast and fixes
__constant__ levelParams d_levels[d_MAXLEVELS];

template <typename T>
__global__
#if NDEBUG // Uinth has a DNDEBUG compiler defined flag in normal trunk builds.
           // Debug builds have no compiler flags we can capture.
__launch_bounds__(640, 1) // For 96 registers with 320 threads.  Allows two kernels to fit within an SM.
                          // Seems to be the performance sweet spot in release mode.
#endif
    void
    rayTraceDataOnionKernel(
        dim3 dimGrid, dim3 dimBlock, int matl, patchParams finePatch,
        gridParams gridP, GPUIntVector fineLevel_ROI_Lo,
        GPUIntVector fineLevel_ROI_Hi, int3 *regionLo, int3 *regionHi,
        gpurandState *randNumStates, RMCRT_flags RT_flags, int curTimeStep,
        GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
        GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *old_gdw,
        GPUDataWarehouse *new_gdw) {

  int maxLevels = gridP.maxLevels;
  int fineL = maxLevels - 1;
  levelParams fineLevel = d_levels[fineL];

  // compute startCell and endCell relative to the block
  int startCell =
      RT_flags.startCell +
      ((RT_flags.endCell - RT_flags.startCell) / gridDim.x) * blockIdx.x;
  int endCell =
      RT_flags.startCell +
      ((RT_flags.endCell - RT_flags.startCell) / gridDim.x) * (blockIdx.x + 1);
  RT_flags.startCell = startCell;
  RT_flags.endCell = endCell;

  //__________________________________
  //
  const GPUGridVariable<T> abskg[d_MAXLEVELS];
  const GPUGridVariable<T> sigmaT4OverPi[d_MAXLEVELS];
  const GPUGridVariable<int> cellType[d_MAXLEVELS];

  //  new_gdw->print();

  //__________________________________
  // coarse level data for the entire level
  for (int l = 0; l < maxLevels; ++l) {
    if (d_levels[l].hasFinerLevel) {
      if (RT_flags.usingFloats) {
        abskg_gdw->getLevel(abskg[l], "abskgRMCRT", matl, l);
      } else {
        abskg_gdw->getLevel(abskg[l], "abskg", matl, l);
      }
      sigmaT4_gdw->getLevel(sigmaT4OverPi[l], "sigmaT4", matl, l);
      cellType_gdw->getLevel(cellType[l], "cellType", matl, l);
    }
  }

  //__________________________________
  //  fine level data for the region of interest.
  //  ToDo:  replace get with getRegion() calls so
  //  so the halo can be > 0
  if (RT_flags.whichROI_algo == patch_based) {
    if (RT_flags.usingFloats) {
      abskg_gdw->get(abskg[fineL], "abskgRMCRT", finePatch.ID, matl, fineL);
    } else {
      abskg_gdw->get(abskg[fineL], "abskg", finePatch.ID, matl, fineL);
    }
    sigmaT4_gdw->get(sigmaT4OverPi[fineL], "sigmaT4", finePatch.ID, matl,
                     fineL);
    cellType_gdw->get(cellType[fineL], "cellType", finePatch.ID, matl, fineL);
  }

  GPUGridVariable<double> divQ_fine;
  GPUGridVariable<GPUStencil7> boundFlux_fine;
  GPUGridVariable<double> radiationVolQ_fine;

  //__________________________________
  //  fine level data for this patch
  if (RT_flags.modifies_divQ) {
    new_gdw->getModifiable(divQ_fine, "divQ", finePatch.ID, matl, fineL);
    new_gdw->getModifiable(boundFlux_fine, "RMCRTboundFlux", finePatch.ID, matl,
                           fineL);
    new_gdw->getModifiable(radiationVolQ_fine, "radiationVolq", finePatch.ID,
                           matl, fineL);
  } else {
// these should be allocateAntPut() calls
    new_gdw->get(divQ_fine, "divQ", finePatch.ID, matl, fineL);
    new_gdw->get(boundFlux_fine, "RMCRTboundFlux", finePatch.ID, matl, fineL);
    new_gdw->get(radiationVolQ_fine, "radiationVolq", finePatch.ID, matl,
                 fineL);
    //__________________________________
    // initialize Extra Cell Loop

    int3 finePatchSize = make_int3(finePatch.hi.x - finePatch.lo.x,
                                   finePatch.hi.y - finePatch.lo.y,
                                   finePatch.hi.z - finePatch.lo.z);
    unsigned short threadID = threadIdx.x + RT_flags.startCell;
    GPUIntVector c = make_int3(
        (threadID % finePatchSize.x) + finePatch.lo.x,
        ((threadID % (finePatchSize.x * finePatchSize.y)) / (finePatchSize.x)) +
            finePatch.lo.y,
        (threadID / (finePatchSize.x * finePatchSize.y)) + finePatch.lo.z);

    while (threadID < RT_flags.endCell) {
      divQ_fine[c] = 0.0;
      radiationVolQ_fine[c] = 0.0;
      //boundFlux_fine[c].initialize(0.0);
      boundFlux_fine[c] = {0.0};

      // move to the next cell
      threadID += blockDim.x;
      c.x = (threadID % finePatchSize.x) + finePatch.lo.x;
      c.y = ((threadID % (finePatchSize.x * finePatchSize.y)) /
             (finePatchSize.x)) +
            finePatch.lo.y;
      c.z = (threadID / (finePatchSize.x * finePatchSize.y)) + finePatch.lo.z;
    }
  }

  // We're going to change thread to cell mappings, so make sure all vars have
  // been initialized before continuing
  __syncthreads();

  //__________________________________
  //
  bool doLatinHyperCube = (RT_flags.rayDirSampleAlgo == LATIN_HYPER_CUBE);

  const int nFluxRays = RT_flags.nFluxRays; // for readability

  // This rand_i array is only needed for LATIN_HYPER_CUBE scheme
  // const int size = 500;
  int rand_i[d_MAX_RAYS]; // Give it a buffer room for many rays.
                          // Hopefully this 500 will always be greater than the
                          // number of rays.
                          // TODO, a 4D array is probably better here (x,y,z,
                          // ray#), saves on memory (no unused buffer)
  if (nFluxRays > d_MAX_RAYS || RT_flags.nDivQRays > d_MAX_RAYS) {
    printf("\n\n\nERROR!  rayTraceKernel() - Cannot have more rays than the "
           "rand_i array size.  Flux rays: %d, divQ rays: %d, size of the "
           "array is.%d\n\n\n",
           nFluxRays, RT_flags.nFluxRays, d_MAX_RAYS);
    // We have to return, otherwise the upcoming math in
    // rayDirectionHyperCube_cellFaceDevice will generate nan values.
    return;
  }
  setupRandNumsSeedAndSequences(randNumStates,
                                (dimGrid.x * dimGrid.y * dimGrid.z *
                                 dimBlock.x * dimBlock.y * dimBlock.z),
                                finePatch.ID, curTimeStep);

  //______________________________________________________________________
  //           R A D I O M E T E R
  //______________________________________________________________________
  // TO BE FILLED IN

  //______________________________________________________________________
  //          B O U N D A R Y F L U X
  //______________________________________________________________________
  if (RT_flags.solveBoundaryFlux) {
    int3 dirIndexOrder[6];
    int3 dirSignSwap[6];

    //_____________________________________________
    //   Ordering for Surface Method
    // This block of code is used to properly place ray origins, and orient ray
    // directions onto the correct face.  This is necessary, because by default,
    // the rays are placed and oriented onto a default face, then require
    // adjustment onto the proper face.
    dirIndexOrder[EAST] = make_int3(2, 1, 0);
    dirIndexOrder[WEST] = make_int3(2, 1, 0);
    dirIndexOrder[NORTH] = make_int3(0, 2, 1);
    dirIndexOrder[SOUTH] = make_int3(0, 2, 1);
    dirIndexOrder[TOP] = make_int3(0, 1, 2);
    dirIndexOrder[BOT] = make_int3(0, 1, 2);

    // Ordering is slightly different from 6Flux since here, rays pass through
    // origin cell from the inside faces.
    dirSignSwap[EAST] = make_int3(-1, 1, 1);
    dirSignSwap[WEST] = make_int3(1, 1, 1);
    dirSignSwap[NORTH] = make_int3(1, -1, 1);
    dirSignSwap[SOUTH] = make_int3(1, 1, 1);
    dirSignSwap[TOP] = make_int3(1, 1, -1);
    dirSignSwap[BOT] = make_int3(1, 1, 1);

    int3 finePatchSize = make_int3(finePatch.hi.x - finePatch.lo.x,
                                   finePatch.hi.y - finePatch.lo.y,
                                   finePatch.hi.z - finePatch.lo.z);
    unsigned short threadID = threadIdx.x + RT_flags.startCell;
    GPUIntVector origin = make_int3(
        (threadID % finePatchSize.x) + finePatch.lo.x,
        ((threadID % (finePatchSize.x * finePatchSize.y)) / (finePatchSize.x)) +
            finePatch.lo.y,
        (threadID / (finePatchSize.x * finePatchSize.y)) + finePatch.lo.z);
    while (threadID < RT_flags.endCell) {
      // get a new set of random numbers
      if (doLatinHyperCube) {
        randVectorDevice(rand_i, nFluxRays, randNumStates);
      }

      if (cellType[fineL][origin] ==
          d_flowCell) { // don't solve for fluxes in intrusions

        //boundFlux_fine[origin].initialize(0.0); // FIXME: Already initialized?
        boundFlux_fine[origin] = {0.0};

        BoundaryFaces boundaryFaces;

        // which surrounding cells are boundaries
        boundFlux_fine[origin].p =
            has_a_boundaryDevice(origin, cellType[fineL], boundaryFaces);

        GPUPoint CC_pos = fineLevel.getCellPosition(origin);

//__________________________________
// Loop over boundary faces of the cell and compute incident radiative flux
#pragma unroll
        for (int i = 0; i < boundaryFaces.size(); i++) {

          int RayFace = boundaryFaces.faceArray[i];
          int UintahFace[6] = {WEST, EAST, SOUTH, NORTH, BOT, TOP};

          double sumI = 0;
          double sumProjI = 0;
          double sumI_prev = 0;
          double sumCosTheta = 0; // used to force sumCosTheta/nRays == 0.5 or
                                  // sum (d_Omega * cosTheta) == pi

//__________________________________
// Flux ray loop
#pragma unroll
          for (int iRay = 0; iRay < nFluxRays; iRay++) {

            GPUVector direction_vector;
            GPUVector rayOrigin;
            double cosTheta;

            if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
              rayDirectionHyperCube_cellFaceDevice(
                  randNumStates, origin, dirIndexOrder[RayFace],
                  dirSignSwap[RayFace], iRay, direction_vector, cosTheta,
                  rand_i[iRay], iRay, nFluxRays);
            } else { // Naive Monte-Carlo sampling
              rayDirection_cellFaceDevice(
                  randNumStates, origin, dirIndexOrder[RayFace],
                  dirSignSwap[RayFace], iRay, direction_vector, cosTheta);
            }

            rayLocation_cellFaceDevice(randNumStates, RayFace, finePatch.dx,
                                       CC_pos, rayOrigin);

            updateSumI_MLDevice<T>(direction_vector, rayOrigin, origin, gridP,
                                   fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo,
                                   regionHi, sigmaT4OverPi, abskg, cellType,
                                   sumI, randNumStates, RT_flags);

            sumProjI +=
                cosTheta * (sumI - sumI_prev); // must subtract sumI_prev, since
                                               // sumI accumulates intensity

            sumCosTheta += cosTheta;

            sumI_prev = sumI;

          } // end of flux ray loop

          sumProjI = sumProjI * (double)RT_flags.nFluxRays / sumCosTheta /
                     2.0; // This operation corrects for error in the first
                          // moment over a half range of the solid angle (Modest
                          // Radiative Heat Transfer page 545 1rst edition)

          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          boundFlux_fine[origin][face] =
              sumProjI * 2 * M_PI / (double)RT_flags.nFluxRays;

        } // boundary faces loop
      }   // end if checking for intrusions

      // move to the next cell
      threadID += blockDim.x;
      origin.x = (threadID % finePatchSize.x) + finePatch.lo.x;
      origin.y = ((threadID % (finePatchSize.x * finePatchSize.y)) /
                  (finePatchSize.x)) +
                 finePatch.lo.y;
      origin.z =
          (threadID / (finePatchSize.x * finePatchSize.y)) + finePatch.lo.z;
    } // while loop
  }

  //______________________________________________________________________
  //         S O L V E   D I V Q
  //______________________________________________________________________
  if (RT_flags.solveDivQ) {

    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    int3 finePatchSize = make_int3(finePatch.hi.x - finePatch.lo.x,
                                   finePatch.hi.y - finePatch.lo.y,
                                   finePatch.hi.z - finePatch.lo.z);
    unsigned short threadID = threadIdx.x + RT_flags.startCell;
    GPUIntVector origin = make_int3(
        (threadID % finePatchSize.x) + finePatch.lo.x,
        ((threadID % (finePatchSize.x * finePatchSize.y)) / (finePatchSize.x)) +
            finePatch.lo.y,
        (threadID / (finePatchSize.x * finePatchSize.y)) + finePatch.lo.z);
    while (threadID < RT_flags.endCell) {

      // don't compute in intrusions and walls
      if (cellType[fineL][origin] != d_flowCell) {
        continue;
      }
      GPUPoint CC_pos = d_levels[fineL].getCellPosition(origin);

      double sumI = 0;

//__________________________________
// ray loop
#pragma unroll
      for (int iRay = 0; iRay < RT_flags.nDivQRays; iRay++) {

        GPUVector ray_direction;
        if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
          ray_direction = findRayDirectionHyperCubeDevice(
              randNumStates, RT_flags.nDivQRays, rand_i[iRay], iRay);
        } else { // Naive Monte-Carlo sampling
          ray_direction = findRayDirectionDevice(randNumStates);
        }

        GPUVector rayOrigin = rayOriginDevice(
            randNumStates, CC_pos, d_levels[fineL].Dx, RT_flags.CCRays);

        updateSumI_MLDevice<T>(ray_direction, rayOrigin, origin, gridP,
                               fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo,
                               regionHi, sigmaT4OverPi, abskg, cellType, sumI,
                               randNumStates, RT_flags);
      } // Ray loop

      //__________________________________
      //  Compute divQ
      divQ_fine[origin] =
          -4.0 * M_PI * abskg[fineL][origin] *
          (sigmaT4OverPi[fineL][origin] - (sumI / RT_flags.nDivQRays));

      // radiationVolq is the incident energy per cell (W/m^3) and is necessary
      // when particle heat transfer models (i.e. Shaddix) are used
      radiationVolQ_fine[origin] = 4.0 * M_PI * (sumI / RT_flags.nDivQRays);

      // move to the next cell
      threadID += blockDim.x;
      origin.x = (threadID % finePatchSize.x) + finePatch.lo.x;
      origin.y = ((threadID % (finePatchSize.x * finePatchSize.y)) /
                  (finePatchSize.x)) +
                 finePatch.lo.y;
      origin.z =
          (threadID / (finePatchSize.x * finePatchSize.y)) + finePatch.lo.z;
      // printf("Got [%d,%d,%d] from %d on counter %d\n", origin.x, origin.y,
      // origin.z, threadID, cellCounter);
    } // end while loop
  }   // solve divQ
}

//______________________________________________________________________
//
//______________________________________________________________________
__device__ GPUVector findRayDirectionDevice(gpurandState *randNumStates) {
  // Random Points On Sphere
  // add fuzz to prevent infs in 1/dirVector calculation
  double plusMinus_one =
      2.0 * randDblExcDevice(randNumStates) - 1.0 + DBL_EPSILON;
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one); // Radius of circle at z
  double theta =
      2.0 * M_PI * randDblExcDevice(randNumStates); // Uniform betwen 0-2Pi

  GPUVector dirVector;
  dirVector.x = r * cos(theta); // Convert to cartesian coordinates
  dirVector.y = r * sin(theta);
  dirVector.z = plusMinus_one;

#if (FIXED_RAY_DIR == 1)
   dirVector = make_double3(0.707106781186548, 0.707106781186548, 0.) * SIGN;
#elif (FIXED_RAY_DIR == 2)
  dirVector = make_double3(0.707106781186548, 0.0, 0.707106781186548) * SIGN;
#elif (FIXED_RAY_DIR == 3)
  dirVector = make_double3(0.0, 0.707106781186548, 0.707106781186548) * SIGN;
#elif (FIXED_RAY_DIR == 4)
  dirVector =
      make_double3(0.707106781186548, 0.707106781186548, 0.707106781186548) *
      SIGN;
#elif (FIXED_RAY_DIR == 5)
  dirVector = make_double3(1, 0, 0) * SIGN;
#elif (FIXED_RAY_DIR == 6)
  dirVector = make_double3(0, 1, 0) * SIGN;
#elif (FIXED_RAY_DIR == 7)
  dirVector = make_double3(0, 0, 1) * SIGN;
#else
#endif

  return dirVector;
}

//______________________________________________________________________
//  Uses stochastically selected regions in polar and azimuthal space to
//  generate the Monte-Carlo directions. Samples Uniformly on a hemisphere
//  and as hence does not include the cosine in the sample.
//______________________________________________________________________
__device__ void rayDirectionHyperCube_cellFaceDevice(
    gpurandState *randNumStates, const GPUIntVector &origin,
    const int3 &indexOrder, const int3 &signOrder, const int iRay,
    GPUVector &dirVector, double &cosTheta, const int bin_i, const int bin_j,
    const int nFluxRays) {
  // randomly sample within each randomly selected region (may not be needed,
  // alternatively choose center of subregion)
  cosTheta =
      (randDblExcDevice(randNumStates) + (double)bin_i) / (double)nFluxRays;

  double theta = acos(cosTheta); // polar angle for the hemisphere
  double phi = 2.0 * M_PI * (randDblExcDevice(randNumStates) + (double)bin_j) /
               (double)nFluxRays; // Uniform betwen 0-2Pi

  cosTheta = cos(theta);

  // Convert to Cartesian
  GPUVector tmp;
  tmp[0] = sin(theta) * cos(phi);
  tmp[1] = sin(theta) * sin(phi);
  tmp[2] = cosTheta;

  // Put direction vector as coming from correct face,
  dirVector[0] = tmp[indexOrder.x] * signOrder.x;
  dirVector[1] = tmp[indexOrder.y] * signOrder.y;
  dirVector[2] = tmp[indexOrder.z] * signOrder.z;
}

//______________________________________________________________________
//
__device__ GPUVector findRayDirectionHyperCubeDevice(gpurandState *randNumStates,
                                                     const int nDivQRays,
                                                     const int bin_i,
                                                     const int bin_j) {
  // Random Points On Sphere
  double plusMinus_one =
      2.0 * (randDblExcDevice(randNumStates) + (double)bin_i) / nDivQRays - 1.0;

  // Radius of circle at z
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);

  // Uniform betwen 0-2Pi
  double phi = 2.0 * M_PI * (randDblExcDevice(randNumStates) + (double)bin_j) /
               nDivQRays;

  GPUVector dirVector;
  dirVector[0] = r * cos(phi); // Convert to cartesian
  dirVector[1] = r * sin(phi);
  dirVector[2] = plusMinus_one;

  return dirVector;
}
//______________________________________________________________________
//  Populate vector with integers which have been randomly shuffled.
//  This is sampling without replacement and can be used to in a
//  Latin-Hyper-Cube sampling scheme.  The algorithm used is the
//  modern Fisher-Yates shuffle.
//______________________________________________________________________
__device__ void randVectorDevice(int int_array[], const int size,
                                 gpurandState *randNumStates) {

  for (int i = 0; i < size; i++) { // populate sequential array from 0 to size-1
    int_array[i] = i;
  }

  for (int i = size - 1; i > 0;
       i--) { // fisher-yates shuffle starting with size-1
    int rand_int =
        randIntDevice(randNumStates, i); // Random number between 0 & i
    int swap = int_array[i];
    int_array[i] = int_array[rand_int];
    int_array[rand_int] = swap;
  }
}
//______________________________________________________________________
// Compute the Ray direction from a cell face
__device__ void rayDirection_cellFaceDevice(
    gpurandState *randNumStates, const GPUIntVector &origin,
    const GPUIntVector &indexOrder, const GPUIntVector &signOrder,
    const int iRay, GPUVector &directionVector, double &cosTheta) {
  // Surface Way to generate a ray direction from the positive z face
  double phi =
      2 * M_PI *
      randDblDevice(randNumStates); // azimuthal angle.  Range of 0 to 2pi
  double theta =
      acos(randDblDevice(randNumStates)); // polar angle for the hemisphere
  cosTheta = cos(theta);
  double sinTheta = sin(theta);

  // Convert to Cartesian
  GPUVector tmp;
  tmp[0] = sinTheta * cos(phi);
  tmp[1] = sinTheta * sin(phi);
  tmp[2] = cosTheta;

  // Put direction vector as coming from correct face,
  directionVector[0] = tmp[indexOrder[0]] * signOrder[0];
  directionVector[1] = tmp[indexOrder[1]] * signOrder[1];
  directionVector[2] = tmp[indexOrder[2]] * signOrder[2];
}

//______________________________________________________________________
//  Compute the physical location of a ray's origin
__device__ GPUVector rayOriginDevice(gpurandState *randNumStates,
                                     const GPUPoint CC_pos, const GPUVector dx,
                                     const bool useCCRays) {
  GPUVector rayOrigin;
  if (useCCRays == false) {
    rayOrigin[0] = CC_pos.x - 0.5 * dx.x + randDblDevice(randNumStates) * dx.x;
    rayOrigin[1] = CC_pos.y - 0.5 * dx.y + randDblDevice(randNumStates) * dx.y;
    rayOrigin[2] = CC_pos.z - 0.5 * dx.z + randDblDevice(randNumStates) * dx.z;
  } else {
    rayOrigin[0] = CC_pos.x;
    rayOrigin[1] = CC_pos.y;
    rayOrigin[2] = CC_pos.z;
  }
  return rayOrigin;
}

//______________________________________________________________________
//  Compute the Ray location from a cell face
__device__ void rayLocation_cellFaceDevice(
    gpurandState *randNumStates, const GPUIntVector &origin,
    const GPUIntVector &indexOrder, const GPUIntVector &shift,
    const double &DyDx, const double &DzDx, GPUVector &location) {
  GPUVector tmp;
  tmp[0] = randDblDevice(randNumStates);
  tmp[1] = 0;
  tmp[2] = randDblDevice(randNumStates) * DzDx;

  // Put point on correct face
  location[0] = tmp[indexOrder[0]] + (double)shift[0];
  location[1] = tmp[indexOrder[1]] + (double)shift[1] * DyDx;
  location[2] = tmp[indexOrder[2]] + (double)shift[2] * DzDx;

  location[0] += (double)origin.x;
  location[1] += (double)origin.y;
  location[2] += (double)origin.z;
}
//______________________________________________________________________
//
//  Compute the Ray location on a cell face
__device__ void rayLocation_cellFaceDevice(gpurandState *randNumStates,
                                           const int face, const GPUVector Dx,
                                           const GPUPoint CC_pos,
                                           GPUVector &rayOrigin) {
  double cellOrigin[3];
  // left, bottom, back corner of the cell
  cellOrigin[X] = CC_pos.x - 0.5 * Dx[X];
  cellOrigin[Y] = CC_pos.y - 0.5 * Dx[Y];
  cellOrigin[Z] = CC_pos.z - 0.5 * Dx[Z];

  switch (face) {
  case WEST:
    rayOrigin[X] = cellOrigin[X];
    rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
    rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
    break;
  case EAST:
    rayOrigin[X] = cellOrigin[X] + Dx[X];
    rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
    rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
    break;
  case SOUTH:
    rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
    rayOrigin[Y] = cellOrigin[Y];
    rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
    break;
  case NORTH:
    rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
    rayOrigin[Y] = cellOrigin[Y] + Dx[Y];
    rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
    break;
  case BOT:
    rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
    rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
    rayOrigin[Z] = cellOrigin[Z];
    break;
  case TOP:
    rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
    rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
    rayOrigin[Z] = cellOrigin[Z] + Dx[Z];
    break;
  default:
    //      throw InternalError("Ray::rayLocation_cellFace,  Invalid FaceType
    //      Specified", __FILE__, __LINE__);
    return;
  }
}
//______________________________________________________________________
//
__device__ bool has_a_boundaryDevice(const GPUIntVector &c,
                                     const GPUGridVariable<int> &celltype,
                                     BoundaryFaces &boundaryFaces) {

  GPUIntVector adj = c;
  bool hasBoundary = false;

  adj[0] = c[0] - 1; // west

  if (celltype[adj] +
      1) { // cell type of flow is -1, so when cellType+1 isn't false, we
    boundaryFaces.addFace(WEST); // know we're at a boundary
    hasBoundary = true;
  }

  adj[0] += 2; // east

  if (celltype[adj] + 1) {
    boundaryFaces.addFace(EAST);
    hasBoundary = true;
  }

  adj[0] -= 1;
  adj[1] = c[1] - 1; // south

  if (celltype[adj] + 1) {
    boundaryFaces.addFace(SOUTH);
    hasBoundary = true;
  }

  adj[1] += 2; // north

  if (celltype[adj] + 1) {
    boundaryFaces.addFace(NORTH);
    hasBoundary = true;
  }

  adj[1] -= 1;
  adj[2] = c[2] - 1; // bottom

  if (celltype[adj] + 1) {
    boundaryFaces.addFace(BOT);
    hasBoundary = true;
  }

  adj[2] += 2; // top

  if (celltype[adj] + 1) {
    boundaryFaces.addFace(TOP);
    hasBoundary = true;
  }

  return (hasBoundary);
}

//______________________________________________________________________
__device__ void raySignStepDevice(GPUVector &sign, int cellStep[],
                                  const GPUVector &inv_direction_vector) {
  // get new step and sign
  for (int d = 0; d < 3; d++) {
    double me = copysign((double)1.0, inv_direction_vector[d]); // +- 1

    sign[d] = fmax(0.0, me); // 0, 1

    cellStep[d] = int(me);
  }
}

//______________________________________________________________________
//
__device__ bool containsCellDevice(GPUIntVector low, GPUIntVector high,
                                   GPUIntVector cell, const int dir) {
  return low[dir] <= cell[dir] && high[dir] > cell[dir];
}

//______________________________________________________________________
//          // used by dataOnion it will be replaced
__device__ void reflect(double &fs, GPUIntVector &cur, GPUIntVector &prevCell,
                        const double abskg, bool &in_domain, int &step,
                        double &sign, double &ray_direction) {
  fs = fs * (1 - abskg);

  // put cur back inside the domain
  cur = prevCell;
  in_domain = true;

  // apply reflection condition
  step *= -1; // begin stepping in opposite direction
  sign *= -1;
  ray_direction *= -1;
}

//______________________________________________________________________
template <typename T>
__device__ void
updateSumIDevice(levelParams level, GPUVector &ray_direction,
                 GPUVector &ray_origin, const GPUIntVector &origin,
                 const GPUVector &Dx, const GPUGridVariable<T> &sigmaT4OverPi,
                 const GPUGridVariable<T> &abskg,
                 const GPUGridVariable<int> &celltype, double &sumI,
                 gpurandState *randNumStates, RMCRT_flags RT_flags)

{

  GPUIntVector cur = origin;
  GPUIntVector prevCell = cur;
  // Step and sign for ray marching
  int step[3];    // Gives +1 or -1 based on sign
  GPUVector sign; //   is 0 for negative ray direction

  GPUVector inv_ray_direction = 1.0 / ray_direction;

  raySignStepDevice(sign, step, ray_direction);

  GPUPoint CC_pos = level.getCellPosition(origin);

  // rayDx is the distance from bottom, left, back, corner of cell to ray
  GPUVector rayDx;
  rayDx[0] =
      ray_origin.x -
      (CC_pos.x - 0.5 * Dx.x); // this can be consolidated using GPUVector
  rayDx[1] = ray_origin.y - (CC_pos.y - 0.5 * Dx.y);
  rayDx[2] = ray_origin.z - (CC_pos.z - 0.5 * Dx.z);

  GPUVector tMax;
  tMax.x = (sign.x * Dx.x - rayDx.x) * inv_ray_direction.x;
  tMax.y = (sign.y * Dx.y - rayDx.y) * inv_ray_direction.y;
  tMax.z = (sign.z * Dx.z - rayDx.z) * inv_ray_direction.z;

  // Length of t to traverse one cell
  GPUVector tDelta;
  tDelta = Abs(inv_ray_direction) * Dx;

  // Initializes the following values for each ray
  bool in_domain = true;
  double tMax_prev = 0;
  double intensity = 1.0;
  double fs = 1.0;
  int nReflect = 0; // Number of reflections
  double optical_thickness = 0;
  double expOpticalThick_prev = 1.0;
  double rayLength = 0.0;
  GPUVector ray_location = ray_origin;

#ifdef RAY_SCATTER
  double scatCoeff = RT_flags.sigmaScat;          //[m^-1]  !! HACK !! This needs to come from data warehouse
  if (scatCoeff == 0) scatCoeff = 1e-99;  // avoid division by zero

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log( randDblExcDevice( randNumStates ) ) / scatCoeff;
#endif

  //+++++++Begin ray tracing+++++++++++++++++++
  // Threshold while loop
  while (intensity > RT_flags.threshold) {

    DIR dir = NONE;

    while (in_domain) {

      prevCell = cur;
      double disMin = -9; // Represents ray segment length.

      //__________________________________
      //  Determine which cell the ray will enter next
      dir = NONE;
      if (tMax.x < tMax.y) {   // X < Y
        if (tMax.x < tMax.z) { // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if (tMax.y < tMax.z) { // Y < Z
          dir = Y;
        } else {
          dir = Z;
        }
      }

      //__________________________________
      //  update marching variables
      cur[dir] = cur[dir] + step[dir];
      disMin = (tMax[dir] - tMax_prev);
      tMax_prev = tMax[dir];
      tMax[dir] = tMax[dir] + tDelta[dir];
      rayLength += disMin;

      ray_location.x = ray_location.x + (disMin * ray_direction.x);
      ray_location.y = ray_location.y + (disMin * ray_direction.y);
      ray_location.z = ray_location.z + (disMin * ray_direction.z);

      in_domain = (celltype[cur] == d_flowCell);

      optical_thickness += abskg[prevCell] * disMin;

      RT_flags.nRaySteps++;

      // Eqn 3-15(see below reference) while
      // Third term inside the parentheses is accounted for in Inet. Chi is
      // accounted for in Inet calc.
      double expOpticalThick = exp(-optical_thickness);

      sumI += sigmaT4OverPi[prevCell] *
              (expOpticalThick_prev - expOpticalThick) * fs;

      expOpticalThick_prev = expOpticalThick;

#ifdef RAY_SCATTER
      if ( (rayLength > scatLength) && in_domain){

        // get new scatLength for each scattering event
        scatLength = -log( randDblExcDevice( randNumStates ) ) / scatCoeff;

        ray_direction     = findRayDirectionDevice( randNumStates );

        inv_ray_direction = 1.0/ray_direction;

        // get new step and sign
        int stepOld = step[dir];
        raySignStepDevice( sign, step, ray_direction);

        // if sign[dir] changes sign, put ray back into prevCell (back scattering)
        // a sign change only occurs when the product of old and new is negative
        if( step[dir] * stepOld < 0 ){
          cur = prevCell;
        }

        GPUPoint CC_pos = level.getCellPosition(cur);

         // rayDx is the distance from bottom, left, back, corner of cell to ray
        rayDx[0] = ray_origin.x - ( CC_pos.x - 0.5*Dx.x );         // this can be consolidated using GPUVector
        rayDx[1] = ray_origin.y - ( CC_pos.y - 0.5*Dx.y );
        rayDx[2] = ray_origin.z - ( CC_pos.z - 0.5*Dx.z );

        tMax.x = (sign.x * Dx.x - rayDx.x) * inv_ray_direction.x;
        tMax.y = (sign.y * Dx.y - rayDx.y) * inv_ray_direction.y;
        tMax.z = (sign.z * Dx.z - rayDx.z) * inv_ray_direction.z;

        // Length of t to traverse one cell
        tDelta    = Abs(inv_ray_direction) * Dx;
        tMax_prev = 0;
        rayLength = 0;  // allow for multiple scattering events per ray
      }
#endif

    } // end domain while loop.

    //  wall emission 12/15/11
    double wallEmissivity = abskg[cur];

    if (wallEmissivity > 1.0) { // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[cur] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if (!RT_flags.allowReflect) {
      intensity = 0;
    }

    //__________________________________
    //  Reflections
    if ((intensity > RT_flags.threshold) && RT_flags.allowReflect) {
      reflect(fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir],
              ray_direction[dir]);
      ++nReflect;
    }

  } // threshold while loop.
} // end of updateSumI function

//______________________________________________________________________
//  Multi-level
template <typename T>
__device__ void updateSumI_MLDevice(
    GPUVector &ray_direction, GPUVector &ray_origin, const GPUIntVector &origin,
    gridParams gridP, const GPUIntVector &fineLevel_ROI_Lo,
    const GPUIntVector &fineLevel_ROI_Hi, const int3 *regionLo,
    const int3 *regionHi, const GPUGridVariable<T> *sigmaT4OverPi,
    const GPUGridVariable<T> *abskg, const GPUGridVariable<int> *cellType,
    double &sumI, gpurandState *randNumStates, RMCRT_flags RT_flags) {
  int maxLevels = gridP.maxLevels; // for readability
  int L = maxLevels - 1;           // finest level
  int prevLev = L;

  GPUIntVector cur = origin;
  GPUIntVector prevCell = cur;

  // Step and sign for ray marching
  int step[3]; // Gives +1 or -1 based on sign
  GPUVector sign;
  GPUVector inv_ray_direction = 1.0 / ray_direction;

  raySignStepDevice(sign, step, inv_ray_direction);
  //__________________________________
  // define tMax & tDelta on all levels
  // go from finest to coarset level so you can compare
  // with 1L rayTrace results.
  GPUPoint CC_posOrigin = d_levels[L].getCellPosition(origin);

  // rayDx is the distance from bottom, left, back, corner of cell to ray
  GPUVector rayDx;
  GPUVector Dx = d_levels[L].Dx;
  rayDx[0] =
      ray_origin.x -
      (CC_posOrigin.x - 0.5 * Dx.x); // this can be consolidated using GPUVector
  rayDx[1] = ray_origin.y - (CC_posOrigin.y - 0.5 * Dx.y);
  rayDx[2] = ray_origin.z - (CC_posOrigin.z - 0.5 * Dx.z);

  GPUVector tMaxV;
  tMaxV.x = (sign.x * Dx.x - rayDx.x) * inv_ray_direction.x;
  tMaxV.y = (sign.y * Dx.y - rayDx.y) * inv_ray_direction.y;
  tMaxV.z = (sign.z * Dx.z - rayDx.z) * inv_ray_direction.z;

  GPUVector tDelta[d_MAXLEVELS];
  for (int Lev = maxLevels - 1; Lev > -1; Lev--) {
    // Length of t to traverse one cell
    tDelta[Lev].x = fabs(inv_ray_direction[0]) * d_levels[Lev].Dx.x;
    tDelta[Lev].y = fabs(inv_ray_direction[1]) * d_levels[Lev].Dx.y;
    tDelta[Lev].z = fabs(inv_ray_direction[2]) * d_levels[Lev].Dx.z;
  }

  // Initializes the following values for each ray
  bool in_domain = true;
  GPUVector tMaxV_prev = make_double3(0.0, 0.0, 0.0);
  double old_length = 0.0;

  double intensity = 1.0;
  double fs = 1.0;
  int nReflect = 0; // Number of reflections
  bool onFineLevel = true;
  double optical_thickness = 0;
  double expOpticalThick_prev = 1.0;
  double rayLength = 0.0;
  GPUVector ray_location = ray_origin;
  GPUPoint CC_pos = CC_posOrigin;

  //______________________________________________________________________
  //  Threshold  loop

  while (intensity > RT_flags.threshold) {
    DIR dir = NONE;

    while (in_domain) {

      prevCell = cur;
      prevLev = L;

      //__________________________________
      //  Determine the princple direction the ray is traveling
      //
      dir = NONE;
      if (tMaxV.x < tMaxV.y) {   // X < Y
        if (tMaxV.x < tMaxV.z) { // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if (tMaxV.y < tMaxV.z) { // Y < Z
          dir = Y;
        } else {
          dir = Z;
        }
      }

      // next cell index and position
      cur[dir] = cur[dir] + step[dir];

      //__________________________________
      // Logic for moving between levels
      //  - Currently you can only move from fine to coarse level
      //  - Don't jump levels if ray is at edge of domain

      CC_pos = d_levels[L].getCellPosition(cur);
      in_domain =
          gridP.domain_BB.inside(CC_pos); // position could be outside of domain

      bool ray_outside_ROI =
          (containsCellDevice(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir) ==
           false);
      bool ray_outside_Region =
          (containsCellDevice(regionLo[L], regionHi[L], cur, dir) == false);

      bool jumpFinetoCoarserLevel =
          (onFineLevel && ray_outside_ROI && in_domain);
      bool jumpCoarsetoCoarserLevel =
          ((onFineLevel == false) && ray_outside_Region && (L > 0) &&
           in_domain);

      if (jumpFinetoCoarserLevel) {
        cur = d_levels[L].mapCellToCoarser(cur);
        L = d_levels[L].getCoarserLevelIndex(); // move to a coarser level
        onFineLevel = false;
      } else if (jumpCoarsetoCoarserLevel) {
        // GPUIntVector c_old = cur;                     // needed for debugging
        cur = d_levels[L].mapCellToCoarser(cur);
        L = d_levels[L].getCoarserLevelIndex(); // move to a coarser level
      }

      //__________________________________
      //  update marching variables
      double distanceTraveled = (tMaxV[dir] - old_length);
      old_length = tMaxV[dir];
      tMaxV_prev = tMaxV;

      tMaxV[dir] = tMaxV[dir] + tDelta[L][dir];

      ray_location.x = ray_location.x + (distanceTraveled * ray_direction.x);
      ray_location.y = ray_location.y + (distanceTraveled * ray_direction.y);
      ray_location.z = ray_location.z + (distanceTraveled * ray_direction.z);

      //__________________________________
      // when moving to a coarse level tmax will change only in the direction
      // the ray is moving
      if (jumpFinetoCoarserLevel || jumpCoarsetoCoarserLevel) {
        GPUVector dx = d_levels[L].Dx;
        double rayDx_Level = ray_location[dir] - (CC_pos[dir] - 0.5 * dx[dir]);
        double tMax_tmp =
            (sign[dir] * dx[dir] - rayDx_Level) * inv_ray_direction[dir];

        tMaxV = tMaxV_prev;
        tMaxV[dir] += tMax_tmp;
      }

      // if the cell isn't a flow cell then terminate the ray
      in_domain = in_domain && (cellType[L][cur] == d_flowCell);

      rayLength += distanceTraveled;

      optical_thickness += abskg[prevLev][prevCell] * distanceTraveled;

      double expOpticalThick = exp(-optical_thickness);

      sumI += sigmaT4OverPi[prevLev][prevCell] *
              (expOpticalThick_prev - expOpticalThick) * fs;

      expOpticalThick_prev = expOpticalThick;

    } // end domain while loop.  ++++++++++++++
    //__________________________________
    //
    double wallEmissivity = abskg[L][cur];

    if (wallEmissivity > 1.0) { // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[L][cur] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if (!RT_flags.allowReflect) {
      intensity = 0;
    }

    //__________________________________
    //  Reflections
    if ((intensity > RT_flags.threshold) && RT_flags.allowReflect) {
      reflect(fs, cur, prevCell, abskg[L][cur], in_domain, step[dir], sign[dir],
              ray_direction[dir]);
      ++nReflect;
    }
  } // threshold while loop.
} // end of updateSumI function

//______________________________________________________________________
// Returns random number between 0 & 1.0 including 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation
//______________________________________________________________________
__device__ double randDblDevice(gpurandState *globalState) {
  int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int tid = blockId * blockDim.x * blockDim.y * blockDim.z +
            threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
            threadIdx.x;
  gpurandState localState = globalState[tid];
  double val = gpurand(&localState);
  globalState[tid] = localState;

#ifdef FIXED_RANDOM_NUM
  return 0.3;
#else
  return (double)val * (1.0 / 4294967295.0);
#endif
}

//______________________________________________________________________
// Returns random number between 0 & 1.0 excluding 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation
//______________________________________________________________________
__device__ double randDblExcDevice(gpurandState *globalState) {
  // 3D blocks of threads, thread ID is computed by:
  int tid = threadIdx.x + blockDim.x * threadIdx.y +
            (blockDim.x * blockDim.y) * threadIdx.z;

  gpurandState localState = globalState[tid];
  double val = gpurand(&localState);
  globalState[tid] = localState;

#ifdef FIXED_RANDOM_NUM
  return 0.3;
#else
  return ((double)val + 0.5) * (1.0 / 4294967296.0);
#endif
}

//______________________________________________________________________
// Returns random integer in [0,n]
// rnd_integer_from_A_to_B = A + gpurand() * (B-A);
//  A = 0
//______________________________________________________________________
__device__ int randIntDevice(gpurandState *globalState, const int B) {
  double val = randDblDevice(globalState);
  return val * B;
}

//______________________________________________________________________
//  Each thread gets same seed, a different sequence number, no offset
//  This will create repeatable results.
__device__ void setupRandNumsSeedAndSequences(gpurandState *randNumStates,
                                              int numStates,
                                              unsigned long long patchID,
                                              unsigned long long curTimeStep) {
  // Generate random numbers using gpurand_init().

  // Format is gpurand_init(seed, sequence, offset, state);

  // Note, it seems a very large sequence really slows things down (bits in the
  // high order region) I measured kernels taking an additional 300 milliseconds
  // due to it!  So the sequence is kept small, using lower order bits only, and
  // intead the seed is given a number with bits in both the high order and low
  // order regions.

  // Unfortunately this isn't perfect.  "Sequences generated with different
  // seeds usually do not have statistically correlated values, but some choices
  // of seeds may give statistically correlated sequences. Sequences generated
  // with the same seed and different sequence numbers will not have
  // statistically correlated values." from here:
  // http://docs.nvidia.com/cuda/gpurand/device-api-overview.html#axzz4SPy8xMuj

  // For RMCRT we will take the tradeoff of possibly having statistically
  // correlated values over the 300 millisecond hit.

  // Generate what should be a unique seed.  To get a unique number the code
  // below computes a tID which is a combination of a patchID, threadID, and the
  // current timestep. This uses the left 20 bits from the patchID, the next 20
  // bits from the curTimeStep and the last 24 bits from the indexId.  Combined
  // that should be unique.

  // Standard CUDA way of computing a threadID
  int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * blockDim.x * blockDim.y * blockDim.z +
                 threadIdx.z * blockDim.y * blockDim.x +
                 threadIdx.y * blockDim.x + threadIdx.x;

  unsigned long long tID =
      (((patchID & 0xFFFFF) << 44) | ((curTimeStep & 0xFFFFF) << 24) |
       (threadId & 0xFFFFFF));

  gpurand_init(tID, threadId, 0, &randNumStates[threadId]);

  // If you want to take the 300 millisecond hit, use this line below instead.
  // gpurand_init(1234, tID, 0, &randNumStates[threadId]);
}

//______________________________________________________________________
//  is cell a debug cell
__device__ bool isDbgCellDevice(GPUIntVector me) {
  int size = 2;
  GPUIntVector dbgCell[2];
  dbgCell[0] = make_int3(0, 0, 0);
  dbgCell[1] = make_int3(5, 5, 5);

  for (int i = 0; i < size; i++) {
    if (me == dbgCell[i]) {
      return true;
    }
  }
  return false;
}

//______________________________________________________________________
//
// Math.h has an std::isnan and std::isinf.  CUDA has an isnan and isinf macro
// (not in a namespace, and not a function) This .cu file sees both, so trying
// to use the CUDA isnan gives compiler ambiguity errors. Dan Sutherland with
// Sandia said they solved this problem by using their own isnan and isinf So
// here is the code for that.  They're also renamed to isNan and isInf to keep
// things separate. (Code was found at
//http://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
// and adapted from
// https://github.com/Itseez/opencv/blob/3.0.0/modules/hal/include/opencv2/hal/defs.h#L447
// )
typedef unsigned long long uint64;

__device__ int isInf(double x) {
  union {
    uint64 u;
    double f;
  } ieee754;
  ieee754.f = x;
  return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) == 0x7ff00000 &&
         ((unsigned)ieee754.u == 0);
}

__device__ int isNan(double x) {
  union {
    uint64 u;
    double f;
  } ieee754;
  ieee754.f = x;
  return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
             ((unsigned)ieee754.u != 0) >
         0x7ff00000;
}

__device__ int isInf(float value) {
  union {
    uint64 u;
    double f;
  } ieee754;
  ieee754.f = value;
  return (ieee754.u & 0x7fffffff) == 0x7f800000;
}

__device__ int isNan(float value) {
  union {
    uint64 u;
    double f;
  } ieee754;
  ieee754.f = value;
  return (ieee754.u & 0x7fffffff) > 0x7f800000;
}

//______________________________________________________________________
//
template <typename T>
void
launchRayTraceKernel(DetailedTask *dtask, dim3 dimGrid, dim3 dimBlock,
                     const int matlIndx, levelParams level, patchParams patch,
                     gpuStream_t *stream, RMCRT_flags RT_flags, int curTimeStep,
                     GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
                     GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *new_gdw) {
  // setup random number generator states on the device, 1 for each thread
  gpurandState *randNumStates;
  int numStates =
      dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z;

  randNumStates = (gpurandState *)GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(
      0, numStates * sizeof(gpurandState));

  // Create a host array, load it with data, and send it over to the GPU
  int nRandNums = 512;
  double *d_debugRandNums;
  size_t randNumsByteSize = nRandNums * sizeof(double);
  d_debugRandNums =
      (double *)GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(0, randNumsByteSize);

  // Making sure we have kernel/mem copy overlapping
  double *h_debugRandNums = new double[nRandNums];
  GPU_RT_SAFE_CALL(gpuHostRegister(h_debugRandNums, randNumsByteSize, gpuHostRegisterPortable));

  // perform computations here on h_debugRandNums
  for (int i = 0; i < nRandNums; i++) {
    h_debugRandNums[i] = i;
  }
  dtask->addTempHostMemoryToBeFreedOnCompletion(h_debugRandNums);
  GPU_RT_SAFE_CALL(gpuMemcpyAsync(d_debugRandNums, h_debugRandNums, randNumsByteSize,
                                  gpuMemcpyHostToDevice, *stream));

  rayTraceKernel<T><<<dimGrid, dimBlock, 0, *stream>>>(
      dimGrid, dimBlock, matlIndx, level, patch, randNumStates, RT_flags,
      curTimeStep, abskg_gdw, sigmaT4_gdw, cellType_gdw, new_gdw);

  // TODO: allocated 
}

//______________________________________________________________________
//
template <typename T>
void launchRayTraceDataOnionKernel(
    DetailedTask *dtask, dim3 dimGrid, dim3 dimBlock, int matlIndex,
    patchParams patch, gridParams gridP, levelParams *levelP,
    GPUIntVector fineLevel_ROI_Lo, GPUIntVector fineLevel_ROI_Hi,
    gpuStream_t *stream, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *old_gdw,
    GPUDataWarehouse *new_gdw) {

  // copy regionLo & regionHi to device memory
  int maxLevels = gridP.maxLevels;

  int3 *dev_regionLo;
  int3 *dev_regionHi;

  size_t size = d_MAXLEVELS * sizeof(int3);
  dev_regionLo = (int3 *)GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(0, size);
  dev_regionHi = (int3 *)GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(0, size);

  // More GPU stuff to allow kernel/copy overlapping
  int3 *myLo = new int3[d_MAXLEVELS];
  int3 *myHi = new int3[d_MAXLEVELS];
  GPU_RT_SAFE_CALL(gpuHostRegister(myLo, sizeof(int3) * d_MAXLEVELS, gpuHostRegisterPortable));
  GPU_RT_SAFE_CALL(gpuHostRegister(myHi, sizeof(int3) * d_MAXLEVELS, gpuHostRegisterPortable));

  for (int l = 0; l < maxLevels; ++l) {
    myLo[l] =
        levelP[l].regionLo; // never use levelP regionLo or hi in the kernel.
    myHi[l] = levelP[l].regionHi; // They are different on each patch
  }

  GPU_RT_SAFE_CALL(gpuMemcpyAsync(dev_regionLo, myLo, size,
                                    gpuMemcpyHostToDevice, *stream));
  GPU_RT_SAFE_CALL(gpuMemcpyAsync(dev_regionHi, myHi, size,
                                    gpuMemcpyHostToDevice, *stream));

  //__________________________________
  // copy levelParams array to constant memory on device
  GPU_RT_SAFE_CALL(gpuMemcpyToSymbolAsync(
      d_levels, levelP, (maxLevels * sizeof(levelParams)), 0,
      gpuMemcpyHostToDevice, *stream));

  //__________________________________
  // setup random number generator states on the device, 1 for each thread
  int numStates =
      dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z;

  gpurandState *randNumStates;
  randNumStates = (gpurandState *)GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(
      0, numStates * sizeof(gpurandState));

  rayTraceDataOnionKernel<T><<<dimGrid, dimBlock, 0, *stream>>>(
      dimGrid, dimBlock, matlIndex, patch, gridP, fineLevel_ROI_Lo,
      fineLevel_ROI_Hi, dev_regionLo, dev_regionHi, randNumStates, RT_flags,
      curTimeStep, abskg_gdw, sigmaT4_gdw, cellType_gdw, old_gdw, new_gdw);
}

//______________________________________________________________________
//  Explicit template instantiations

template void launchRayTraceKernel<double>(
    DetailedTask *dtask, dim3 dimGrid, dim3 dimBlock, const int matlIndx,
    levelParams level, patchParams patch, gpuStream_t *stream,
    RMCRT_flags RT_flags, int curTimeStep, GPUDataWarehouse *abskg_gdw,
    GPUDataWarehouse *sigmaT4_gdw, GPUDataWarehouse *cellType_gdw,
    GPUDataWarehouse *new_gdw);

template void launchRayTraceKernel<float>(
    DetailedTask *dtask, dim3 dimGrid, dim3 dimBlock, const int matlIndx,
    levelParams level, patchParams patch, gpuStream_t *stream,
    RMCRT_flags RT_flags, int curTimeStep, GPUDataWarehouse *abskg_gdw,
    GPUDataWarehouse *sigmaT4_gdw, GPUDataWarehouse *celltype_gdw,
    GPUDataWarehouse *new_gdw);

template void launchRayTraceDataOnionKernel<double>(
    DetailedTask *dtask, dim3 dimGrid, dim3 dimBlock, int matlIndex,
    patchParams patch, gridParams gridP, levelParams *levelP,
    GPUIntVector fineLevel_ROI_Lo, GPUIntVector fineLevel_ROI_Hi,
    gpuStream_t *stream, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *old_gdw,
    GPUDataWarehouse *new_gdw);

template void launchRayTraceDataOnionKernel<float>(
    DetailedTask *dtask, dim3 dimGrid, dim3 dimBlock, int matlIndex,
    patchParams patch, gridParams gridP, levelParams *levelP,
    GPUIntVector fineLevel_ROI_Lo, GPUIntVector fineLevel_ROI_Hi,
    gpuStream_t *stream, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *old_gdw,
    GPUDataWarehouse *new_gdw);

} // end namespace Uintah
