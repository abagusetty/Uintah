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

#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.hpp>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <CCA/Components/Schedulers/GPUMemoryPool.h>

#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUStencil7.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Util/GPU.h>

#include <sci_defs/uintah_defs.h>

#define DEBUG -9         // 1: divQ, 2: boundFlux, 3: scattering
#define FIXED_RANDOM_NUM // also edit in
// src/Core/Math/MersenneTwister.h to compare with Ray:CPU

#define FIXED_RAY_DIR                                                          \
  -9 // Sets ray direction.  1: (0.7071,0.7071, 0), 2: (0.7071, 0, 0.7071), 3:
     // (0, 0.7071, 0.7071)
     //                      4: (0.7071, 0.7071, 7071), 5: (1,0,0)  6: (0, 1,
     //                      0),   7: (0,0,1)
#define SIGN 1 // Multiply the FIXED_RAY_DIRs by value

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

template <typename T>
using localAcc =
    sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::target::local>;

namespace Uintah {

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
template <class T>
void rayTraceKernel(sycl::nd_item<2> &item, const int matl, levelParams level,
                    patchParams patch, RMCRT_flags RT_flags, int curTimeStep,
                    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
                    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *new_gdw) {

//   sycl::group thread_block = item.get_group();

//   // calculate the thread indices
//   int tidX = item.get_global_id(1) + patch.loEC.x();
//   int tidY = item.get_global_id(0) + patch.loEC.y();

//   sycl::device_ptr<T> sigmaT4OverPi_ptr{nullptr};
//   sycl::int3 sigmaT4OverPi_size;
//   sycl::int3 sigmaT4OverPi_offset;
//   sycl::device_ptr<T> abskg_ptr{nullptr};
//   sycl::int3 abskg_size;
//   sycl::int3 abskg_offset;
//   sycl::device_ptr<int> cellType_ptr{nullptr};
//   sycl::int3 cellType_size;
//   sycl::int3 cellType_offset;
//   sycl::device_ptr<double> divQ_ptr{nullptr};
//   sycl::int3 divQ_size;
//   sycl::int3 divQ_offset;
//   sycl::device_ptr<GPUStencil7> boundFlux_ptr{nullptr};
//   sycl::int3 boundFlux_size;
//   sycl::int3 boundFlux_offset;
//   sycl::device_ptr<double> radiationVolQ_ptr{nullptr};
//   sycl::int3 radiationVolQ_size;
//   sycl::int3 radiationVolQ_offset;

//   sigmaT4_gdw->getLevel(item, sigmaT4OverPi_ptr, sigmaT4OverPi_size, sigmaT4OverPi_offset, "sigmaT4", matl, level.index);
//   cellType_gdw->getLevel(item, cellType_ptr, cellType_size, cellType_offset, "cellType", matl, level.index);

//   if (RT_flags.usingFloats) {
//     abskg_gdw->getLevel(item, abskg_ptr, abskg_size, abskg_offset, "abskgRMCRT",
//                         matl, level.index);
//   } else {
//     abskg_gdw->getLevel(item, abskg_ptr, abskg_size, abskg_offset, "abskg",
//                         matl, level.index);
//   }

//   if (RT_flags.modifies_divQ) {
//     new_gdw->getModifiable(item, divQ_ptr, divQ_size, divQ_offset, "divQ",
//                            patch.ID, matl);
//     new_gdw->getModifiable(item, boundFlux_ptr, boundFlux_size,
//                            boundFlux_offset, "RMCRTboundFlux", patch.ID, matl);
//     new_gdw->getModifiable(item, radiationVolQ_ptr, radiationVolQ_size,
//                            radiationVolQ_offset, "radiationVolq", patch.ID,
//                            matl);
//   } else {
//     // these should be allocateAndPut() calls
//     new_gdw->get(item, divQ_ptr, divQ_size, divQ_offset, "divQ", patch.ID,
//                  matl);
//     new_gdw->get(item, boundFlux_ptr, boundFlux_size, boundFlux_offset,
//                  "RMCRTboundFlux", patch.ID, matl);
//     new_gdw->get(item, radiationVolQ_ptr, radiationVolQ_size,
//                  radiationVolQ_offset, "radiationVolq", patch.ID, matl);

//     // Extra Cell Loop
//     if ((tidX >= patch.loEC.x()) && (tidY >= patch.loEC.y()) &&
//         (tidX < patch.hiEC.x()) && (tidY < patch.hiEC.y())) {
// // patch boundary check
// #pragma unroll
//       // loop through z slice
//       for (int z = patch.loEC.z(); z < patch.hiEC.z(); z++) {
//         sycl::int3 c{tidX, tidY, z};

//         size_t divQ_index =
//             c.z() - divQ_offset.z() +
//             divQ_size.z() * (c.y() - divQ_offset.y() +
//                              (c.x() - divQ_offset.x()) * divQ_size.y());
//         size_t radiationVolQ_index =
//             c.z() - radiationVolQ_offset.z() +
//             radiationVolQ_size.z() *
//                 (c.y() - radiationVolQ_offset.y() +
//                  (c.x() - radiationVolQ_offset.x()) * radiationVolQ_size.y());
//         size_t boundFlux_index =
//             c.z() - boundFlux_offset.z() +
//             boundFlux_size.z() *
//                 (c.y() - boundFlux_offset.y() +
//                  (c.x() - boundFlux_offset.x()) * boundFlux_size.y());

//         divQ_ptr[divQ_index] = 0.0;
//         radiationVolQ_ptr[radiationVolQ_index] = 0.0;
//         boundFlux_ptr[boundFlux_index] = 0.0;
//       }
//     }
//   }

//   bool doLatinHyperCube = (RT_flags.rayDirSampleAlgo == LATIN_HYPER_CUBE);

//   const int nFluxRays = RT_flags.nFluxRays; // for readability

//   // This rand_i array is only needed for LATIN_HYPER_CUBE scheme
//   const int size = 1000;
//   int rand_i[size]; // Give it a buffer room of 1000.  But we should only use
//                     // nFluxRays items in it. Hopefully this 1000 will always be
//                     // greater than nFluxRays.
//                     // TODO, a 4D array is probably better here (x,y,z, ray#),
//                     // saves on memory (no unused buffer) and computation time
//                     // (don't need to compute the rays twice)
//   if (nFluxRays > size) {
//     // printf("\n\n\nERROR!  rayTraceKernel() - Cannot have more rays than the "
//     //        "rand_i array size.  nFluxRays is %d, size of the array
//     //        is.%d\n\n\n", nFluxRays, size);
//     // We have to return, otherwise the upcoming math in
//     // rayDirectionHyperCube_cellFaceDevice will generate nan values.
//     return;
//   }

//   //______________________________________________________________________
//   //           R A D I O M E T E R
//   //______________________________________________________________________
//   // TO BE FILLED IN

//   //______________________________________________________________________
//   //          B O U N D A R Y F L U X
//   //______________________________________________________________________
//   // setupRandNumsSeedAndSequences((dimGrid.x * dimGrid.y * dimGrid.z *
//   // dimBlock.x * dimBlock.y * dimBlock.z),
//   //                               patch.ID, curTimeStep);

//   if (RT_flags.solveBoundaryFlux) {

  // using tile_t = sycl::int3[6];
  // tile_t& dirIndexOrder = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
  // tile_t& dirSignSwap = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
  
//     //_____________________________________________
//     //   Ordering for Surface Method
//     // This block of code is used to properly place ray origins, and orient ray
//     // directions onto the correct face.  This is necessary, because by default,
//     // the rays are placed and oriented onto a default face, then require
//     // adjustment onto the proper face.
//     dirIndexOrder[EAST] = {2, 1, 0};
//     dirIndexOrder[WEST] = {2, 1, 0};
//     dirIndexOrder[NORTH] = {0, 2, 1};
//     dirIndexOrder[SOUTH] = {0, 2, 1};
//     dirIndexOrder[TOP] = {0, 1, 2};
//     dirIndexOrder[BOT] = {0, 1, 2};

//     // Ordering is slightly different from 6Flux since here, rays pass through
//     // origin cell from the inside faces.
//     dirSignSwap[EAST] = {-1, 1, 1};
//     dirSignSwap[WEST] = {1, 1, 1};
//     dirSignSwap[NORTH] = {1, -1, 1};
//     dirSignSwap[SOUTH] = {1, 1, 1};
//     dirSignSwap[TOP] = {1, 1, -1};
//     dirSignSwap[BOT] = {1, 1, 1};

//     sycl::group_barrier(thread_block);

//     //__________________________________
//     // GPU equivalent of GridIterator loop - calculate sets of rays per thread
//     if ((tidX >= patch.lo.x()) && (tidY >= patch.lo.y()) &&
//         (tidX < patch.hi.x()) &&
//         (tidY < patch.hi.y())) { // patch boundary check
// #pragma unroll
//       for (int z = patch.lo.z(); z < patch.hi.z();
//            z++) { // loop through z slices

//         sycl::int3 origin{tidX, tidY, z}; // for each thread

//         // get a new set of random numbers
//         if (doLatinHyperCube) {
//           randVectorDevice(rand_i, nFluxRays);
//         }

//         size_t boundFlux_idx =
//             origin.z() - boundFlux_offset.z() +
//             boundFlux_size.z() *
//                 (origin.y() - boundFlux_offset.y() +
//                  (origin.x() - boundFlux_offset.x()) * boundFlux_size.y());
//         boundFlux_ptr[boundFlux_idx] = 0;

//         BoundaryFaces boundaryFaces;

//         // which surrounding cells are boundaries
//         (boundFlux_ptr[boundFlux_idx])[6] =
//             has_a_boundaryDevice(origin, cellType_ptr, cellType_size,
//                                  cellType_offset, boundaryFaces);

//         sycl::double3 CC_pos = level.getCellPosition(origin);

// //__________________________________
// // Loop over boundary faces of the cell and compute incident radiative flux
// #pragma unroll
//         for (int i = 0; i < boundaryFaces.size(); i++) {

//           int RayFace = boundaryFaces.faceArray[i];
//           int UintahFace[6] = {WEST, EAST, SOUTH, NORTH, BOT, TOP};

//           double sumI = 0;
//           double sumProjI = 0;
//           double sumI_prev = 0;
//           double sumCosTheta = 0; // used to force sumCosTheta/nRays == 0.5 or
//                                   // sum (d_Omega * cosTheta) == pi

// //__________________________________
// // Flux ray loop
// #pragma unroll
//           for (int iRay = 0; iRay < nFluxRays; iRay++) {
//             sycl::double3 direction_vector;
//             sycl::double3 rayOrigin;
//             double cosTheta;

//             if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
//               rayDirectionHyperCube_cellFaceDevice(
//                   origin, dirIndexOrder[RayFace], dirSignSwap[RayFace], iRay,
//                   direction_vector, cosTheta, rand_i[iRay], iRay, nFluxRays);
//             } else {
//               rayDirection_cellFaceDevice(origin, dirIndexOrder[RayFace],
//                                           dirSignSwap[RayFace], iRay,
//                                           direction_vector, cosTheta);
//             }

//             rayLocation_cellFaceDevice(RayFace, patch.dx, CC_pos, rayOrigin);

//             updateSumIDevice(level, direction_vector, rayOrigin, origin,
//                              patch.dx, sigmaT4OverPi_ptr, sigmaT4OverPi_size,
//                              sigmaT4OverPi_offset, abskg_ptr, abskg_size,
//                              abskg_offset, cellType_ptr, cellType_size,
//                              cellType_offset, sumI, RT_flags);

//             sumProjI +=
//                 cosTheta * (sumI - sumI_prev); // must subtract sumI_prev, since
//                                                // sumI accumulates intensity

//             sumCosTheta += cosTheta;

//             sumI_prev = sumI;
//           } // end of flux ray loop

//           sumProjI = sumProjI * (double)nFluxRays / sumCosTheta /
//                      2.0; // This operation corrects for error in the first
//                           // moment over a half range of the solid angle (Modest
//                           // Radiative Heat Transfer page 545 1st edition)
//           //__________________________________
//           //  Compute Net Flux to the boundary
//           int face = UintahFace[RayFace];
//           (boundFlux_ptr[boundFlux_idx])[face] =
//               sumProjI * 2 * M_PI / (double)nFluxRays;
//         } // boundary faces loop
//       }   // z slices loop
//     }     // X-Y Thread loop
//   }

//   //______________________________________________________________________
//   //         S O L V E   D I V Q
//   //______________________________________________________________________
//   // Setup the original seeds so we can get the same random numbers again.
//   // setupRandNumsSeedAndSequences((dimGrid.x * dimGrid.y * dimGrid.z *
//   // dimBlock.x * dimBlock.y * dimBlock.z),
//   //                               patch.ID, curTimeStep);

//   if (RT_flags.solveDivQ) {
//     const int nDivQRays = RT_flags.nDivQRays; // for readability

//     // GPU equivalent of GridIterator loop - calculate sets of rays per thread
//     if ((tidX >= patch.lo.x()) && (tidY >= patch.lo.y()) &&
//         (tidX < patch.hi.x()) &&
//         (tidY < patch.hi.y())) { // patch boundary check
// #pragma unroll
//       for (int z = patch.lo.z(); z < patch.hi.z();
//            z++) {                            // loop through z slices
//         sycl::int3 origin = {tidX, tidY, z}; // for each thread

//         // Get the same set of random numbers as we had before.  We need the
//         // same rays.
//         if (doLatinHyperCube) {
//           randVectorDevice(rand_i, nFluxRays);
//         }

//         double sumI = 0;
//         sycl::double3 CC_pos = level.getCellPosition(origin);

//         // don't compute in intrusions and walls
//         size_t cellType_idx =
//             origin.z() - cellType_offset.z() +
//             cellType_size.z() *
//                 (origin.y() - cellType_offset.y() +
//                  (origin.x() - cellType_offset.x()) * cellType_size.y());
//         if (cellType_ptr[cellType_idx] != d_flowCell) {
//           continue;
//         }

// //__________________________________
// // ray loop
// #pragma unroll
//         for (int iRay = 0; iRay < nDivQRays; iRay++) {
//           sycl::double3 direction_vector;
//           if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
//             direction_vector =
//                 findRayDirectionHyperCubeDevice(nDivQRays, rand_i[iRay], iRay);
//           } else { // Naive Monte-Carlo sampling
//             direction_vector = findRayDirectionDevice();
//           }

//    Compute the physical location of a ray's origin	
// 	sycl::double3 rayOrigin = (RT_flags_CCRays == false) ?
// 	  (CC_pos - 0.5 * patch.dx + randDblDevice(distr, engine) * patch.dx) : CC_pos;

//           updateSumIDevice(level, direction_vector, rayOrigin, origin, patch.dx,
//                            sigmaT4OverPi_ptr, sigmaT4OverPi_size,
//                            sigmaT4OverPi_offset, abskg_ptr, abskg_size,
//                            abskg_offset, cellType_ptr, cellType_size,
//                            cellType_offset, sumI, RT_flags);
//         } // Ray loop

//         //__________________________________
//         //  Compute divQ
//         size_t divQ_idx =
//             origin.z() - divQ_offset.z() +
//             divQ_size.z() * (origin.y() - divQ_offset.y() +
//                              (origin.x() - divQ_offset.x()) * divQ_size.y());
//         size_t sigmaT4OverPi_idx =
//             origin.z() - sigmaT4OverPi_offset.z() +
//             sigmaT4OverPi_size.z() * (origin.y() - sigmaT4OverPi_offset.y() +
//                                       (origin.x() - sigmaT4OverPi_offset.x()) *
//                                           sigmaT4OverPi_size.y());
//         size_t abskg_idx =
//             origin.z() - abskg_offset.z() +
//             abskg_size.z() * (origin.y() - abskg_offset.y() +
//                               (origin.x() - abskg_offset.x()) * abskg_size.y());
//         size_t radiationVolQ_idx =
//             origin.z() - radiationVolQ_offset.z() +
//             radiationVolQ_size.z() * (origin.y() - radiationVolQ_offset.y() +
//                                       (origin.x() - radiationVolQ_offset.x()) *
//                                           radiationVolQ_size.y());

//         divQ_ptr[divQ_idx] = -4.0 * M_PI * abskg_ptr[abskg_idx] *
//                              (sigmaT4OverPi_ptr[sigmaT4OverPi_idx] -
//                               (sumI / RT_flags.nDivQRays));

//         // radiationVolq is the incident energy per cell (W/m^3) and is
//         // necessary when particle heat transfer models (i.e. Shaddix) are used
//         radiationVolQ_ptr[radiationVolQ_idx] =
//             4.0 * M_PI * abskg_ptr[abskg_idx] * (sumI / RT_flags.nDivQRays);
//       } // end z-slice loop
//     }   // end domain boundary check
//   }     // solve divQ

} // end ray trace kernel

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer data onion kernel
//---------------------------------------------------------------------------
// hard-wired for 2-levels now, but this should be fast and fixes

template <class T>
void rayTraceDataOnionKernel(
    sycl::nd_item<1> &item,
    sycl::constant_ptr<levelParams> d_levels_ptr, int matl,
    patchParams finePatch, gridParams gridP,
    sycl::int3 fineLevel_ROI_Lo,
    sycl::int3 fineLevel_ROI_Hi,
    const sycl::int3* __restrict__ regionLo,
    const sycl::int3* __restrict__ regionHi,
    unsigned int RT_flags_startCell,
    unsigned int RT_flags_endCell,
    int RT_flags_nDivQRays,
    int RT_flags_nFluxRays,
    bool RT_flags_CCRays,
    bool RT_flags_modifies_divQ,
    bool RT_flags_solveBoundaryFlux,
    bool RT_flags_solveDivQ,
    double RT_flags_threshold,
    bool RT_flags_allowReflect,
    bool RT_flags_usingFloats,
    int RT_flags_whichROI_algo,
    int RT_flags_rayDirSampleAlgo,
    int curTimeStep,
    GPUDataWarehouse* __restrict__ abskg_gdw,
    GPUDataWarehouse* __restrict__ sigmaT4_gdw,
    GPUDataWarehouse* __restrict__ cellType_gdw,
    GPUDataWarehouse* __restrict__ new_gdw) {

  int threadIdx_x = static_cast<int>(item.get_local_id(0));
  int gridDim_x   = static_cast<int>(item.get_group_range(0));
  int blockIdx_x  = static_cast<int>(item.get_group_linear_id()); // since it is just 1-D kernel

  int maxLevels = gridP.maxLevels;
  int fineL = maxLevels - 1;
  levelParams fineLevel = d_levels_ptr[fineL];

  // compute startCell and endCell relative to the block
  int startCell = RT_flags_startCell + ((RT_flags_endCell - RT_flags_startCell) / gridDim_x) * blockIdx_x;
  int endCell = RT_flags_startCell + ((RT_flags_endCell - RT_flags_startCell) / gridDim_x) * (blockIdx_x + 1);
  RT_flags_startCell = startCell;
  RT_flags_endCell = endCell;

  // sycl::ext::oneapi::experimental::printf("ThreadID: (%d, %d, %d): %d, %d\n",
  // 					  threadIdx_x, blockIdx_x, gridDim_x, RT_flags_startCell, RT_flags_endCell);

  //__________________________________
  //

  sycl::device_ptr<T> abskg_ptr[d_MAXLEVELS] = {nullptr};
  sycl::int3 abskg_size[d_MAXLEVELS];
  sycl::int3 abskg_offset[d_MAXLEVELS];
  sycl::device_ptr<T> sigmaT4OverPi_ptr[d_MAXLEVELS] = {nullptr};
  sycl::int3 sigmaT4OverPi_size[d_MAXLEVELS];
  sycl::int3 sigmaT4OverPi_offset[d_MAXLEVELS];
  sycl::device_ptr<int> cellType_ptr[d_MAXLEVELS] = {nullptr};
  sycl::int3 cellType_size[d_MAXLEVELS];
  sycl::int3 cellType_offset[d_MAXLEVELS];

  //__________________________________
  // coarse level data for the entire level
  for (int l = 0; l < maxLevels; ++l) {
    if (d_levels_ptr[l].hasFinerLevel) {
      if (RT_flags_usingFloats) {
        abskg_gdw->getLevel(item, abskg_ptr[l], abskg_size[l], abskg_offset[l], "abskgRMCRT", matl, l);
      } else {
        abskg_gdw->getLevel(item, abskg_ptr[l], abskg_size[l], abskg_offset[l], "abskg", matl, l);
      }
      sigmaT4_gdw->getLevel(item, sigmaT4OverPi_ptr[l], sigmaT4OverPi_size[l], sigmaT4OverPi_offset[l], "sigmaT4", matl, l);
      cellType_gdw->getLevel(item, cellType_ptr[l], cellType_size[l], cellType_offset[l], "cellType", matl, l);
    }
  }

  //__________________________________
  //  fine level data for the region of interest.
  //  ToDo:  replace get with getRegion() calls so
  //  so the halo can be > 0
  if (RT_flags_whichROI_algo == patch_based) {
    if (RT_flags_usingFloats) {
      abskg_gdw->get(item, abskg_ptr[fineL], abskg_size[fineL], abskg_offset[fineL], "abskgRMCRT", finePatch.ID, matl, fineL);
    } else {
      abskg_gdw->get(item, abskg_ptr[fineL], abskg_size[fineL], abskg_offset[fineL], "abskg", finePatch.ID, matl, fineL);
    }
    sigmaT4_gdw->get(item, sigmaT4OverPi_ptr[fineL], sigmaT4OverPi_size[fineL], sigmaT4OverPi_offset[fineL], "sigmaT4", finePatch.ID, matl, fineL);
    cellType_gdw->get(item, cellType_ptr[fineL], cellType_size[fineL], cellType_offset[fineL], "cellType", finePatch.ID, matl, fineL);
  }

  sycl::device_ptr<double> divQ_fine_ptr{nullptr};
  sycl::int3 divQ_fine_size(0,0,0);
  sycl::int3 divQ_fine_offset(0,0,0);
  sycl::device_ptr<GPUStencil7> boundFlux_fine_ptr{nullptr};
  sycl::int3 boundFlux_fine_size(0,0,0);
  sycl::int3 boundFlux_fine_offset(0,0,0);
  sycl::device_ptr<double> radiationVolQ_fine_ptr{nullptr};
  sycl::int3 radiationVolQ_fine_size(0,0,0);
  sycl::int3 radiationVolQ_fine_offset(0,0,0);

  //__________________________________
  //  fine level data for this patch

  if (RT_flags_modifies_divQ) {
    new_gdw->getModifiable(item, divQ_fine_ptr, divQ_fine_size, divQ_fine_offset, "divQ", finePatch.ID, matl, fineL);
    new_gdw->getModifiable(item, boundFlux_fine_ptr, boundFlux_fine_size, boundFlux_fine_offset, "RMCRTboundFlux", finePatch.ID, matl, fineL);
    new_gdw->getModifiable(item, radiationVolQ_fine_ptr, radiationVolQ_fine_size, radiationVolQ_fine_offset, "radiationVolq", finePatch.ID, matl, fineL);
  } else {
    // these should be allocateAntPut() calls
    new_gdw->get(item, divQ_fine_ptr, divQ_fine_size, divQ_fine_offset, "divQ", finePatch.ID, matl, fineL);
    new_gdw->get(item, boundFlux_fine_ptr, boundFlux_fine_size, boundFlux_fine_offset, "RMCRTboundFlux", finePatch.ID, matl, fineL);
    new_gdw->get(item, radiationVolQ_fine_ptr, radiationVolQ_fine_size, radiationVolQ_fine_offset, "radiationVolq", finePatch.ID, matl, fineL);

    //__________________________________
    // initialize Extra Cell Loop

    sycl::int3 finePatchSize = finePatch.hi - finePatch.lo;

    sycl::int3 c{0,0,0};
    for (unsigned short threadID = threadIdx_x + RT_flags_startCell;
	 threadID < RT_flags_endCell;
	 threadID += item.get_local_range(0)) {

      c.x() = (threadID % finePatchSize.x()) + finePatch.lo.x();
      c.y() = ((threadID % (finePatchSize.x() * finePatchSize.y())) / (finePatchSize.x())) + finePatch.lo.y();
      c.z() = (threadID / (finePatchSize.x() * finePatchSize.y())) + finePatch.lo.z();

      int divQ_fine_idx = c.z() - divQ_fine_offset.z() + divQ_fine_size.z() *
              (c.y() - divQ_fine_offset.y() + (c.x() - divQ_fine_offset.x()) * divQ_fine_size.y());
      int radiationVolQ_fine_idx = c.z() - radiationVolQ_fine_offset.z() + radiationVolQ_fine_size.z() *
              (c.y() - radiationVolQ_fine_offset.y() + (c.x() - radiationVolQ_fine_offset.x()) * radiationVolQ_fine_size.y());
      int boundFlux_fine_idx = c.z() - boundFlux_fine_offset.z() + boundFlux_fine_size.z() *
              (c.y() - boundFlux_fine_offset.y() + (c.x() - boundFlux_fine_offset.x()) * boundFlux_fine_size.y());

      divQ_fine_ptr[divQ_fine_idx] = 0.0;
      radiationVolQ_fine_ptr[radiationVolQ_fine_idx] = 0.0;
      boundFlux_fine_ptr[boundFlux_fine_idx] = 0;
    } // for (threadID < RT_flags_endCell)
  }

  // We're going to change thread to cell mappings, so make sure all vars have
  // been initialized before continuing
  // sycl::group_barrier(thread_block);

  //__________________________________
  //
  bool doLatinHyperCube = (RT_flags_rayDirSampleAlgo == LATIN_HYPER_CUBE);

  // This rand_i array is only needed for LATIN_HYPER_CUBE scheme
  // const int size = 500;
  int rand_i[d_MAX_RAYS]; // Give it a buffer room for many rays.
  // Hopefully this 500 will always be greater than the
  // number of rays.
  // TODO, a 4D array is probably better here (x,y,z,
  // ray#), saves on memory (no unused buffer)
  if (RT_flags_nFluxRays > d_MAX_RAYS || RT_flags_nDivQRays > d_MAX_RAYS) {
    sycl::ext::oneapi::experimental::printf(
      "\n\n\nASSERT!  rayTraceDataOnionKernel() - Cannot have more rays than the "
      "rand_i array size.  Flux rays: %d, divQ rays: %d, size of the "
      "array is.%d\n\n\n", RT_flags_nFluxRays, RT_flags_nDivQRays, d_MAX_RAYS);

    // We have to assert here, otherwise the upcoming math in
    // rayDirectionHyperCube_cellFaceDevice will generate nan values.
    assert(false);
  }

  oneapi::mkl::rng::device::philox4x32x10<1> engine(1234, 0);
  oneapi::mkl::rng::device::uniform<double> distr(0.0, 1.0);
  // setupRandNumsSeedAndSequences((dimGrid.x * dimGrid.y * dimGrid.z *
  //                                dimBlock.x * dimBlock.y * dimBlock.z),
  //                               finePatch.ID, curTimeStep);

  //______________________________________________________________________
  //           R A D I O M E T E R
  //______________________________________________________________________
  // TO BE FILLED IN

  //______________________________________________________________________
  //          B O U N D A R Y F L U X
  //______________________________________________________________________
  if (RT_flags_solveBoundaryFlux) {
    sycl::int3 dirIndexOrder[6];
    sycl::int3 dirSignSwap[6];

    //_____________________________________________
    //   Ordering for Surface Method
    // This block of code is used to properly place ray origins, and orient ray
    // directions onto the correct face.  This is necessary, because by default,
    // the rays are placed and oriented onto a default face, then require
    // adjustment onto the proper face.
    dirIndexOrder[EAST] = {2, 1, 0};
    dirIndexOrder[WEST] = {2, 1, 0};
    dirIndexOrder[NORTH] = {0, 2, 1};
    dirIndexOrder[SOUTH] = {0, 2, 1};
    dirIndexOrder[TOP] = {0, 1, 2};
    dirIndexOrder[BOT] = {0, 1, 2};

    // Ordering is slightly different from 6Flux since here, rays pass through
    // origin cell from the inside faces.
    dirSignSwap[EAST] = {-1, 1, 1};
    dirSignSwap[WEST] = {1, 1, 1};
    dirSignSwap[NORTH] = {1, -1, 1};
    dirSignSwap[SOUTH] = {1, 1, 1};
    dirSignSwap[TOP] = {1, 1, -1};
    dirSignSwap[BOT] = {1, 1, 1};

    sycl::int3 finePatchSize = finePatch.hi - finePatch.lo;

    sycl::int3 origin{0,0,0};
    for (unsigned short threadID = threadIdx_x + RT_flags_startCell;
	 threadID < RT_flags_endCell;
	 threadID += item.get_local_range(0)) {

      origin.x() = (threadID % finePatchSize.x()) + finePatch.lo.x();
      origin.y() = ((threadID % (finePatchSize.x() * finePatchSize.y())) / (finePatchSize.x())) + finePatch.lo.y();
      origin.z() = (threadID / (finePatchSize.x() * finePatchSize.y())) + finePatch.lo.z();

      // get a new set of random numbers
      if (doLatinHyperCube) {
        randVectorDevice(rand_i, RT_flags_nFluxRays, &distr, &engine);
      }

      // don't solve for fluxes in intrusions
      int cellType_idx = origin.z() - cellType_offset[fineL].z() + cellType_size[fineL].z() *
	(origin.y() - cellType_offset[fineL].y() + (origin.x() - cellType_offset[fineL].x()) * cellType_size[fineL].y());

      if ((cellType_ptr[fineL])[cellType_idx] == d_flowCell) {
        int boundFlux_fine_idx = origin.z() - boundFlux_fine_offset.z() + boundFlux_fine_size.z() *
	  (origin.y() - boundFlux_fine_offset.y() + (origin.x() - boundFlux_fine_offset.x()) * boundFlux_fine_size.y());
        boundFlux_fine_ptr[boundFlux_fine_idx] = 0; // FIXME: Already initialized?

        BoundaryFaces boundaryFaces;

        // which surrounding cells are boundaries
        (boundFlux_fine_ptr[boundFlux_fine_idx])[6] = has_a_boundaryDevice(
            origin, cellType_ptr[fineL], cellType_size[fineL], cellType_offset[fineL], boundaryFaces);

        sycl::double3 CC_pos = fineLevel.getCellPosition(origin);

//__________________________________
// Loop over boundary faces of the cell and compute incident radiative flux
#pragma unroll 1
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
#pragma unroll 1
          for (int iRay = 0; iRay < RT_flags_nFluxRays; iRay++) {
            sycl::double3 direction_vector;
            sycl::double3 rayOrigin;
            double cosTheta;

            if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
              rayDirectionHyperCube_cellFaceDevice(&distr, &engine,
						   origin, dirIndexOrder[RayFace], dirSignSwap[RayFace], iRay,
						   direction_vector, cosTheta, rand_i[iRay], iRay, RT_flags_nFluxRays);
            } else { // Naive Monte-Carlo sampling
              rayDirection_cellFaceDevice(&distr, &engine,
					  origin, dirIndexOrder[RayFace],
                                          dirSignSwap[RayFace], iRay,
                                          direction_vector, cosTheta);
            }

            rayLocation_cellFaceDevice(&distr, &engine, RayFace, finePatch.dx, CC_pos, rayOrigin);

            updateSumI_MLDevice(direction_vector, rayOrigin, origin, gridP,
                                fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo,
                                regionHi, sigmaT4OverPi_ptr, sigmaT4OverPi_size,
                                sigmaT4OverPi_offset, abskg_ptr, abskg_size,
                                abskg_offset, cellType_ptr, cellType_size,
                                cellType_offset, sumI, RT_flags_threshold, RT_flags_allowReflect, d_levels_ptr);

	    // must subtract sumI_prev, since sumI accumulates intensity
            sumProjI += cosTheta * (sumI - sumI_prev);

            sumCosTheta += cosTheta;

            sumI_prev = sumI;

          } // end of flux ray loop

	  // This operation corrects for error in the first
	  // moment over a half range of the solid angle (Modest
	  // Radiative Heat Transfer page 545 1rst edition)

	  sumProjI = sumProjI * (double)RT_flags_nFluxRays / sumCosTheta / 2.0;

          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          (boundFlux_fine_ptr[boundFlux_fine_idx])[face] = sumProjI * 2 * M_PI / (double)RT_flags_nFluxRays;

        } // boundary faces loop
      }   // end if checking for intrusions
    } // for loop (threadID)
  }

  //______________________________________________________________________
  //         S O L V E   D I V Q
  //______________________________________________________________________

  if (RT_flags_solveDivQ) {

    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    sycl::int3 finePatchSize = finePatch.hi - finePatch.lo;

    sycl::int3 origin{0,0,0};
    for (unsigned short threadID = threadIdx_x + RT_flags_startCell;
	 threadID < RT_flags_endCell;
	 threadID += item.get_local_range(0)) {

      origin.x() = (threadID % finePatchSize.x()) + finePatch.lo.x();
      origin.y() = ((threadID % (finePatchSize.x() * finePatchSize.y())) / (finePatchSize.x())) + finePatch.lo.y();
      origin.z() = (threadID / (finePatchSize.x() * finePatchSize.y())) + finePatch.lo.z();

      // don't compute in intrusions and walls
      int cellType_idx = origin.z() - cellType_offset[fineL].z() + cellType_size[fineL].z() *
	(origin.y() - cellType_offset[fineL].y() +
	 (origin.x() - cellType_offset[fineL].x()) * cellType_size[fineL].y());

      if ((cellType_ptr[fineL])[cellType_idx] != d_flowCell) {
        continue;
      }
      sycl::double3 CC_pos = d_levels_ptr[fineL].getCellPosition(origin);

      double sumI = 0;

      //__________________________________
      // ray loop
#pragma unroll 1
      for (int iRay = 0; iRay < RT_flags_nDivQRays; iRay++) {
        sycl::double3 ray_direction{0};
        if (doLatinHyperCube) { // Latin-Hyper-Cube sampling
          ray_direction = findRayDirectionHyperCubeDevice(&distr, &engine, RT_flags_nDivQRays, rand_i[iRay], iRay);
        } else { // Naive Monte-Carlo sampling
          ray_direction = findRayDirectionDevice(&distr, &engine);
        }

       // Compute the physical location of a ray's origin	
	sycl::double3 rayOrigin = (RT_flags_CCRays == false) ?
	  (CC_pos - 0.5 * d_levels_ptr[fineL].Dx + randDblDevice(&distr, &engine) * d_levels_ptr[fineL].Dx) : CC_pos;

        updateSumI_MLDevice(ray_direction, rayOrigin, origin, gridP,
                            fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo,
                            regionHi, sigmaT4OverPi_ptr, sigmaT4OverPi_size,
                            sigmaT4OverPi_offset, abskg_ptr, abskg_size,
                            abskg_offset, cellType_ptr, cellType_size,
                            cellType_offset, sumI, RT_flags_threshold, RT_flags_allowReflect,
			    d_levels_ptr);
      } // Ray loop


      //__________________________________
      //  Compute divQ
      int lhs_idx = origin.z() - divQ_fine_offset.z() +
	divQ_fine_size.z() * (origin.y() - divQ_fine_offset.y() +
			      (origin.x() - divQ_fine_offset.x()) * divQ_fine_size.y());
      int rhs1_idx = origin.z() - abskg_offset[fineL].z() +
	abskg_size[fineL].z() * (origin.y() - abskg_offset[fineL].y() +
				 (origin.x() - abskg_offset[fineL].x()) * abskg_size[fineL].y());
      int rhs2_idx = origin.z() - sigmaT4OverPi_offset[fineL].z() +
	sigmaT4OverPi_size[fineL].z() *
	(origin.y() - sigmaT4OverPi_offset[fineL].y() +
	 (origin.x() - sigmaT4OverPi_offset[fineL].x()) *
	 sigmaT4OverPi_size[fineL].y());

      divQ_fine_ptr[lhs_idx] = -4.0 * M_PI * (abskg_ptr[fineL])[rhs1_idx] *
	((sigmaT4OverPi_ptr[fineL])[rhs2_idx] - (sumI / RT_flags_nDivQRays));

      // radiationVolq is the incident energy per cell (W/m^3) and is necessary
      // when particle heat transfer models (i.e. Shaddix) are used
      lhs_idx = origin.z() - radiationVolQ_fine_offset.z() +
	radiationVolQ_fine_size.z() * (origin.y() - radiationVolQ_fine_offset.y() + (origin.x() - radiationVolQ_fine_offset.x()) *
				       radiationVolQ_fine_size.y());
      radiationVolQ_fine_ptr[lhs_idx] = 4.0 * M_PI * (sumI / RT_flags_nDivQRays);

    } // for (threadID)
  }   // solve divQ
}

//______________________________________________________________________
//
//______________________________________________________________________
__attribute__((always_inline)) sycl::double3
findRayDirectionDevice(oneapi::mkl::rng::device::uniform<double> *distr,
		       oneapi::mkl::rng::device::philox4x32x10<1> *engine) {
  // Random Points On Sphere
  // add fuzz to prevent infs in 1/dirVector calculation
  double plusMinus_one = 2.0 * randDblExcDevice(distr, engine) - 1.0 + DBL_EPSILON;
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one); // Radius of circle at z
  double theta = 2.0 * M_PI * randDblExcDevice(distr, engine);       // Uniform betwen 0-2Pi

  sycl::double3 dirVector;
  dirVector.x() = r * std::cos(theta); // Convert to cartesian coordinates
  dirVector.y() = r * std::sin(theta);
  dirVector.z() = plusMinus_one;

#if (FIXED_RAY_DIR == 1)
  dirVector = sycl::double3(0.707106781186548, 0.707106781186548, 0.) * SIGN;
#elif (FIXED_RAY_DIR == 2)
  dirVector = sycl::double3(0.707106781186548, 0.0, 0.707106781186548) * SIGN;
#elif (FIXED_RAY_DIR == 3)
  dirVector = sycl::double3(0.0, 0.707106781186548, 0.707106781186548) * SIGN;
#elif (FIXED_RAY_DIR == 4)
  dirVector =
      sycl::double3(0.707106781186548, 0.707106781186548, 0.707106781186548) *
      SIGN;
#elif (FIXED_RAY_DIR == 5)
  dirVector = sycl::double3(1, 0, 0) * SIGN;
#elif (FIXED_RAY_DIR == 6)
  dirVector = sycl::double3(0, 1, 0) * SIGN;
#elif (FIXED_RAY_DIR == 7)
  dirVector = sycl::double3(0, 0, 1) * SIGN;
#else
#endif

  return dirVector;
}

//______________________________________________________________________
//  Uses stochastically selected regions in polar and azimuthal space to
//  generate the Monte-Carlo directions. Samples Uniformly on a hemisphere
//  and as hence does not include the cosine in the sample.
//______________________________________________________________________

// device-only kernel
__attribute__((always_inline)) void
rayDirectionHyperCube_cellFaceDevice(
    oneapi::mkl::rng::device::uniform<double> *distr,
    oneapi::mkl::rng::device::philox4x32x10<1> *engine,
    const sycl::int3 &origin, const sycl::int3 &indexOrder,
    const sycl::int3 &signOrder, const int iRay, sycl::double3 &dirVector,
    double &cosTheta, const int bin_i, const int bin_j, const int nFluxRays) {

  // randomly sample within each randomly selected region (may not be needed,
  // alternatively choose center of subregion)
  cosTheta = (randDblExcDevice(distr, engine) + (double)bin_i) / (double)nFluxRays;

  double theta = sycl::acos(cosTheta); // polar angle for the hemisphere
  double phi = 2.0 * M_PI * (randDblExcDevice(distr, engine) + (double)bin_j) /
               (double)nFluxRays; // Uniform betwen 0-2Pi

  cosTheta = std::cos(theta);

  // Convert to Cartesian
  sycl::double3 tmp;
  tmp.x() = std::sin(theta) * std::cos(phi);
  tmp.y() = std::sin(theta) * std::sin(phi);
  tmp.z() = cosTheta;

  // Put direction vector as coming from correct face,
  dirVector.x() = tmp[indexOrder.x()] * signOrder.x();
  dirVector.y() = tmp[indexOrder.y()] * signOrder.y();
  dirVector.z() = tmp[indexOrder.z()] * signOrder.z();
}

//______________________________________________________________________
//

// device-only kernel
__attribute__((always_inline)) sycl::double3
findRayDirectionHyperCubeDevice(
    oneapi::mkl::rng::device::uniform<double> *distr,
    oneapi::mkl::rng::device::philox4x32x10<1> *engine,
    const int nDivQRays,
    const int bin_i,
    const int bin_j) {
  // Random Points On Sphere
  double plusMinus_one = 2.0 * (randDblExcDevice(distr, engine) + (double)bin_i) / nDivQRays - 1.0;

  // Radius of circle at z
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);

  // Uniform betwen 0-2Pi
  double phi = 2.0 * M_PI * (randDblExcDevice(distr, engine) + (double)bin_j) / nDivQRays;

  sycl::double3 dirVector; // Convert to cartesian
  dirVector.x() = r * std::cos(phi);
  dirVector.y() = r * std::sin(phi);
  dirVector.z() = plusMinus_one;

  return dirVector;
}

//______________________________________________________________________
//  Populate vector with integers which have been randomly shuffled.
//  This is sampling without replacement and can be used to in a
//  Latin-Hyper-Cube sampling scheme.  The algorithm used is the
//  modern Fisher-Yates shuffle.
//______________________________________________________________________
__attribute__((always_inline)) void
randVectorDevice(int int_array[], const int size,
		 oneapi::mkl::rng::device::uniform<double> *distr,
		 oneapi::mkl::rng::device::philox4x32x10<1> *engine) {
  // populate sequential array from 0 to size-1
  for (int i = 0; i < size; i++) {
    int_array[i] = i;
  }

  // fisher-yates shuffle starting with size-1
  for (int i = size - 1; i > 0; i--) {
    // Random number between 0 & i
    int rand_int = static_cast<int>( i * randDblDevice(distr, engine) );
    int swap = int_array[i];
    int_array[i] = int_array[rand_int];
    int_array[rand_int] = swap;
  }
}
//______________________________________________________________________
// Compute the Ray direction from a cell face
__attribute__((always_inline)) void
rayDirection_cellFaceDevice(oneapi::mkl::rng::device::uniform<double> *distr,
			    oneapi::mkl::rng::device::philox4x32x10<1> *engine,
			    const sycl::int3 &origin,
			    const sycl::int3 &indexOrder,
			    const sycl::int3 &signOrder, const int iRay,
			    sycl::double3 &directionVector,
			    double &cosTheta) {
  // Surface Way to generate a ray direction from the positive z face
  double phi = 2 * M_PI * randDblDevice(distr, engine); // azimuthal angle.  Range of 0 to 2pi
  double theta = sycl::acos(randDblDevice(distr, engine)); // polar angle for the hemisphere
  cosTheta = std::cos(theta);
  double sinTheta = std::sin(theta);

  // Convert to Cartesian
  sycl::double3 tmp;
  tmp[0] = sinTheta * std::cos(phi);
  tmp[1] = sinTheta * std::sin(phi);
  tmp[2] = cosTheta;

  // Put direction vector as coming from correct face,
  directionVector.x() = tmp[indexOrder[0]] * signOrder.x();
  directionVector.y() = tmp[indexOrder[1]] * signOrder.y();
  directionVector.z() = tmp[indexOrder[2]] * signOrder.z();
}

//______________________________________________________________________
//  Compute the Ray location from a cell face
// void rayLocation_cellFaceDevice(const sycl::int3 &origin,
//                                 const sycl::int3 &indexOrder,
//                                 const sycl::int3 &shift, const double &DyDx,
//                                 const double &DzDx, sycl::double3 &location) {
//   sycl::double3 tmp;
//   tmp.x() = randDblDevice();
//   tmp.y() = 0;
//   tmp.z() = randDblDevice() * DzDx;

//   // Put point on correct face
//   location.x() = tmp[indexOrder[0]] + (double)shift.x();
//   location.y() = tmp[indexOrder[1]] + (double)shift.y() * DyDx;
//   location.z() = tmp[indexOrder[2]] + (double)shift.z() * DzDx;

//   location.x() += (double)origin.x();
//   location.y() += (double)origin.y();
//   location.z() += (double)origin.z();
// }
//______________________________________________________________________
//
//  Compute the Ray location on a cell face
void rayLocation_cellFaceDevice(oneapi::mkl::rng::device::uniform<double> *distr,
				oneapi::mkl::rng::device::philox4x32x10<1> *engine,
				const int face,
				const sycl::double3 Dx,
                                const sycl::double3 CC_pos,
                                sycl::double3 &rayOrigin) {
  // left, bottom, back corner of the cell
  sycl::double3 cellOrigin = CC_pos - 0.5 * Dx;

  switch (face) {
  case WEST:
    rayOrigin.x() = cellOrigin.x();
    rayOrigin.y() = cellOrigin.y() + randDblDevice(distr, engine) * Dx.y();
    rayOrigin.z() = cellOrigin.z() + randDblDevice(distr, engine) * Dx.z();
    break;
  case EAST:
    rayOrigin.x() = cellOrigin.x() + Dx.x();
    rayOrigin.y() = cellOrigin.y() + randDblDevice(distr, engine) * Dx.y();
    rayOrigin.z() = cellOrigin.z() + randDblDevice(distr, engine) * Dx.z();
    break;
  case SOUTH:
    rayOrigin.x() = cellOrigin.x() + randDblDevice(distr, engine) * Dx.x();
    rayOrigin.y() = cellOrigin.y();
    rayOrigin.z() = cellOrigin.z() + randDblDevice(distr, engine) * Dx.z();
    break;
  case NORTH:
    rayOrigin.x() = cellOrigin.x() + randDblDevice(distr, engine) * Dx.x();
    rayOrigin.y() = cellOrigin.y() + Dx.y();
    rayOrigin.z() = cellOrigin.z() + randDblDevice(distr, engine) * Dx.z();
    break;
  case BOT:
    rayOrigin.x() = cellOrigin.x() + randDblDevice(distr, engine) * Dx.x();
    rayOrigin.y() = cellOrigin.y() + randDblDevice(distr, engine) * Dx.y();
    rayOrigin.z() = cellOrigin.z();
    break;
  case TOP:
    rayOrigin.x() = cellOrigin.x() + randDblDevice(distr, engine) * Dx.x();
    rayOrigin.y() = cellOrigin.y() + randDblDevice(distr, engine) * Dx.y();
    rayOrigin.z() = cellOrigin.z() + Dx.z();
    break;
  default:
    //      throw InternalError("Ray::rayLocation_cellFace,  Invalid FaceType
    //      Specified", __FILE__, __LINE__);
    return;
  }
}
//______________________________________________________________________
//
// device-only kernel
__attribute__((always_inline)) bool
has_a_boundaryDevice(const sycl::int3 &c, sycl::device_ptr<int> celltype_ptr,
                     sycl::int3& celltype_size, sycl::int3& celltype_offset,
                     BoundaryFaces &boundaryFaces) {

  sycl::int3 adj = c;
  int adj_idx = adj.z() - celltype_offset.z() + celltype_size.z() *
    (adj.y() - celltype_offset.y() + (adj.x() - celltype_offset.x()) * celltype_size.y());
  bool hasBoundary = false;

  adj.x() = c.x() - 1; // west

  // cell type of flow is -1, so when cellType+1 isn't false, we
  if (celltype_ptr[adj_idx] + 1) {
    boundaryFaces.addFace(WEST); // know we're at a boundary
    hasBoundary = true;
  }

  adj.x() += 2; // east

  if (celltype_ptr[adj_idx] + 1) {
    boundaryFaces.addFace(EAST);
    hasBoundary = true;
  }

  adj.x() -= 1;
  adj.y() = c.y() - 1; // south

  if (celltype_ptr[adj_idx] + 1) {
    boundaryFaces.addFace(SOUTH);
    hasBoundary = true;
  }

  adj.y() += 2; // north

  if (celltype_ptr[adj_idx] + 1) {
    boundaryFaces.addFace(NORTH);
    hasBoundary = true;
  }

  adj.y() -= 1;
  adj.z() = c.z() - 1; // bottom

  if (celltype_ptr[adj_idx] + 1) {
    boundaryFaces.addFace(BOT);
    hasBoundary = true;
  }

  adj.z() += 2; // top

  if (celltype_ptr[adj_idx] + 1) {
    boundaryFaces.addFace(TOP);
    hasBoundary = true;
  }

  return (hasBoundary);
}

//______________________________________________________________________
//
__attribute__((always_inline)) void
raySignStepDevice(sycl::double3 &sign, int cellStep[],
                  const sycl::double3 &inv_direction_vector) {
  // get new step and sign
  double me = 0;

  me = sycl::copysign((double)1.0, inv_direction_vector.x()); // +- 1
  sign.x() = sycl::fmax(0.0, me);                             // 0, 1
  cellStep[0] = static_cast<int>(me);

  me = sycl::copysign((double)1.0, inv_direction_vector.y()); // +- 1
  sign.y() = sycl::fmax(0.0, me);                             // 0, 1
  cellStep[1] = static_cast<int>(me);

  me = sycl::copysign((double)1.0, inv_direction_vector.z()); // +- 1
  sign.z() = sycl::fmax(0.0, me);                             // 0, 1
  cellStep[2] = static_cast<int>(me);
}

//______________________________________________________________________
//
__attribute__((always_inline))
bool containsCellDevice(const sycl::int3& low, const sycl::int3& high,
			const sycl::int3& cell, const int dir) {
  return low[dir] <= cell[dir] && high[dir] > cell[dir];
}

//______________________________________________________________________
//          // used by dataOnion it will be replaced
void reflect(double &fs, sycl::int3 &cur, sycl::int3 &prevCell,
             const double abskg, bool &in_domain, int &step, double &sign,
             double &ray_direction) {
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
//
template <class T>
__attribute__((always_inline)) void updateSumIDevice(
    levelParams level, sycl::double3 &ray_direction, sycl::double3 &ray_origin,
    const sycl::int3 &origin, const sycl::double3 &Dx, sycl::device_ptr<T> sigmaT4OverPi_ptr,
    sycl::int3 sigmaT4OverPi_size, sycl::int3 sigmaT4OverPi_offset,
    sycl::device_ptr<T> abskg_ptr, sycl::int3 abskg_size, sycl::int3 abskg_offset,
    sycl::device_ptr<int> celltype_ptr, sycl::int3 celltype_size, sycl::int3 celltype_offset,
    double &sumI, RMCRT_flags RT_flags) {
  sycl::int3 cur = origin;
  sycl::int3 prevCell = cur;
  // Step and sign for ray marching
  int step[3];        // Gives +1 or -1 based on sign
  sycl::double3 sign; //   is 0 for negative ray direction

  sycl::double3 inv_ray_direction = 1.0 / ray_direction;

  raySignStepDevice(sign, step, ray_direction);

  sycl::double3 CC_pos = level.getCellPosition(origin);

  // rayDx is the distance from bottom, left, back, corner of cell to ray
  sycl::double3 rayDx = ray_origin - (CC_pos - 0.5 * Dx);

  sycl::double3 tMax = (sign * Dx - rayDx) * inv_ray_direction;

  // Length of t to traverse one cell
  sycl::double3 tDelta;
  tDelta = sycl::abs(inv_ray_direction) * Dx;

  // Initializes the following values for each ray
  bool in_domain = true;
  double tMax_prev = 0;
  double intensity = 1.0;
  double fs = 1.0;
  int nReflect = 0; // Number of reflections
  double optical_thickness = 0;
  double expOpticalThick_prev = 1.0;
  double rayLength = 0.0;
  sycl::double3 ray_location = ray_origin;

#ifdef RAY_SCATTER
  double scatCoeff = RT_flags.sigmaScat; //[m^-1]  !! HACK !! This needs to come
                                         // from data warehouse
  if (scatCoeff == 0)
    scatCoeff = 1e-99; // avoid division by zero

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log(randDblExcDevice()) / scatCoeff;
#endif

  //+++++++Begin ray tracing+++++++++++++++++++
  // Threshold while loop
  while (intensity > RT_flags.threshold) {

    DIR dir = NONE;

    while (in_domain) {

      prevCell = cur;

      int sigmaT4OverPi_idx =
          prevCell.z() - sigmaT4OverPi_offset.z() +
          sigmaT4OverPi_size.z() * (prevCell.y() - sigmaT4OverPi_offset.y() +
                                    (prevCell.x() - sigmaT4OverPi_offset.x()) *
                                        sigmaT4OverPi_size.y());
      int abskg_idx =
          prevCell.z() - abskg_offset.z() +
          abskg_size.z() * (prevCell.y() - abskg_offset.y() +
                            (prevCell.x() - abskg_offset.x()) * abskg_size.y());

      double disMin = -9; // Represents ray segment length.

      //__________________________________
      //  Determine which cell the ray will enter next
      dir = NONE;
      if (tMax.x() < tMax.y()) {   // X < Y
        if (tMax.x() < tMax.z()) { // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if (tMax.y() < tMax.z()) { // Y < Z
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

      ray_location = ray_location + (disMin * ray_direction);

      int celltype_cur = cur.z() - celltype_offset.z() + celltype_size.z() *
	(cur.y() - celltype_offset.y() + (cur.x() - celltype_offset.x()) * celltype_size.y());
      in_domain = (celltype_ptr[celltype_cur] == d_flowCell);

      optical_thickness += abskg_ptr[abskg_idx] * disMin;

      RT_flags.nRaySteps++;

      // Eqn 3-15(see below reference) while
      // Third term inside the parentheses is accounted for in Inet. Chi is
      // accounted for in Inet calc.
      double expOpticalThick = sycl::exp(-optical_thickness);

      sumI += sigmaT4OverPi_ptr[sigmaT4OverPi_idx] *
              (expOpticalThick_prev - expOpticalThick) * fs;

      expOpticalThick_prev = expOpticalThick;

#ifdef RAY_SCATTER
      if ((rayLength > scatLength) && in_domain) {

        // get new scatLength for each scattering event
        scatLength = -log(randDblExcDevice()) / scatCoeff;

        ray_direction = findRayDirectionDevice();

        inv_ray_direction = 1.0 / ray_direction;

        // get new step and sign
        int stepOld = step[dir];
        raySignStepDevice(sign, step, ray_direction);

        // if sign[dir] changes sign, put ray back into prevCell (back
        // scattering) a sign change only occurs when the product of old and new
        // is negative
        cur = (step[dir] * stepOld < 0) ? prevCell : cur;

        sycl::double3 CC_pos = level.getCellPosition(cur);

        // rayDx is the distance from bottom, left, back, corner of cell to ray
        rayDx = ray_origin - (CC_pos - 0.5 * Dx);
        tMax = (sign * Dx - rayDx) * inv_ray_direction;

        // Length of t to traverse one cell
        tDelta = sycl::abs(inv_ray_direction) * Dx;
        tMax_prev = 0;
        rayLength = 0; // allow for multiple scattering events per ray
      }
#endif

    } // end domain while loop.

    //  wall emission 12/15/11
    int wallEmi_idx = cur.z() - abskg_offset.z() + abskg_size.z() *
      (cur.y() - abskg_offset.y() + (cur.x() - abskg_offset.x()) * abskg_size.y());
    double wallEmissivity = abskg_ptr[wallEmi_idx];

    // Ensure wall emissivity doesn't exceed one.
    wallEmissivity = (wallEmissivity > 1.0) ? 1.0 : wallEmissivity;

    intensity = sycl::exp(-optical_thickness);

    int tmp_idx = cur.z() - sigmaT4OverPi_offset.z() + sigmaT4OverPi_size.z() *
            (cur.y() - sigmaT4OverPi_offset.y() + (cur.x() - sigmaT4OverPi_offset.x()) * sigmaT4OverPi_size.y());
    sumI += wallEmissivity * sigmaT4OverPi_ptr[tmp_idx] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if (!RT_flags.allowReflect) {
      intensity = 0;
    }

    //__________________________________
    //  Reflections
    if ((intensity > RT_flags.threshold) && RT_flags.allowReflect) {
      int abs_idx = cur.z() - abskg_offset.z() + abskg_size.z() *
	(cur.y() - abskg_offset.y() + (cur.x() - abskg_offset.x()) * abskg_size.y());

      reflect(fs, cur, prevCell, abskg_ptr[abs_idx], in_domain, step[dir], sign[dir], ray_direction[dir]);
      ++nReflect;
    }

  } // threshold while loop.
} // end of updateSumIDevice function

//______________________________________________________________________
//  Multi-level
template <class T>
__attribute__((always_inline)) void updateSumI_MLDevice(
    sycl::double3 &ray_direction, sycl::double3 &ray_origin,
    const sycl::int3 &origin, gridParams gridP,
    const sycl::int3 &fineLevel_ROI_Lo,
    const sycl::int3 &fineLevel_ROI_Hi,
    const sycl::int3* __restrict__ regionLo,
    const sycl::int3* __restrict__ regionHi,
    sycl::device_ptr<T>* __restrict__ sigmaT4OverPi_ptr,
    sycl::int3* __restrict__ sigmaT4OverPi_size,
    sycl::int3* __restrict__ sigmaT4OverPi_offset,
    sycl::device_ptr<T>* __restrict__ abskg_ptr,
    sycl::int3* __restrict__ abskg_size,
    sycl::int3* __restrict__ abskg_offset,
    sycl::device_ptr<int>* __restrict__ cellType_ptr,
    sycl::int3* __restrict__ cellType_size,
    sycl::int3* __restrict__ cellType_offset,
    double &sumI, const double& RT_flags_threshold, const bool& RT_flags_allowReflect,
    const levelParams *__restrict__ d_levels) {

  int maxLevels = gridP.maxLevels; // for readability
  int L = maxLevels - 1;           // finest level
  int prevLev = L;

  sycl::int3 cur = origin;
  sycl::int3 prevCell = cur;

  // Step and sign for ray marching
  int step[3]; // Gives +1 or -1 based on sign
  sycl::double3 sign;
  sycl::double3 inv_ray_direction = 1.0 / ray_direction;

  raySignStepDevice(sign, step, inv_ray_direction);
  //__________________________________
  // define tMax & tDelta on all levels
  // go from finest to coarset level so you can compare
  // with 1L rayTrace results.
  sycl::double3 CC_posOrigin = d_levels[L].getCellPosition(origin);

  // rayDx is the distance from bottom, left, back, corner of cell to ray
  sycl::double3 Dx = d_levels[L].Dx;
  sycl::double3 rayDx = ray_origin - (CC_posOrigin - 0.5 * Dx);
  sycl::double3 tMaxV = (sign * Dx - rayDx) * inv_ray_direction;

  sycl::double3 tDelta[d_MAXLEVELS];
  for (int Lev = maxLevels - 1; Lev > -1; Lev--) {
    // Length of t to traverse one cell
    tDelta[Lev] = sycl::fabs(inv_ray_direction) * d_levels[Lev].Dx;
  }

  // Initializes the following values for each ray
  bool in_domain = true;
  sycl::double3 tMaxV_prev{0.0};
  double old_length = 0.0;

  double intensity = 1.0;
  double fs = 1.0;
  int nReflect = 0; // Number of reflections
  bool onFineLevel = true;
  double optical_thickness = 0;
  double expOpticalThick_prev = 1.0;
  sycl::double3 ray_location = ray_origin;
  sycl::double3 CC_pos = CC_posOrigin;

  //______________________________________________________________________
  //  Threshold  loop

  while (intensity > RT_flags_threshold) {
    DIR dir = NONE;

    while (in_domain) {
      prevCell = cur;
      prevLev = L;

      //__________________________________
      //  Determine the princple direction the ray is traveling
      //
      dir = NONE;
      if (tMaxV.x() < tMaxV.y()) {   // X < Y
        if (tMaxV.x() < tMaxV.z()) { // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if (tMaxV.y() < tMaxV.z()) { // Y < Z
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
      in_domain = gridP.domain_BB.inside(CC_pos); // position could be outside of domain

      bool ray_outside_ROI = (containsCellDevice(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir) == false);
      bool ray_outside_Region = (containsCellDevice(regionLo[L], regionHi[L], cur, dir) == false);

      bool jumpFinetoCoarserLevel = (onFineLevel && ray_outside_ROI && in_domain);
      bool jumpCoarsetoCoarserLevel = ((onFineLevel == false) && ray_outside_Region && (L > 0) && in_domain);

      if (jumpFinetoCoarserLevel) {
        cur = d_levels[L].mapCellToCoarser(cur);
        L = (d_levels[L].index - 1 > 0) ? d_levels[L].index : 0;
        onFineLevel = false;
      } else if (jumpCoarsetoCoarserLevel) {
        cur = d_levels[L].mapCellToCoarser(cur);
        L = (d_levels[L].index - 1 > 0) ? d_levels[L].index : 0;
      }

      //__________________________________
      //  update marching variables
      double distanceTraveled = (tMaxV[dir] - old_length);
      old_length = tMaxV[dir];
      tMaxV_prev = tMaxV;

      tMaxV[dir] = tMaxV[dir] + tDelta[L][dir];

      ray_location += (distanceTraveled * ray_direction);

      //__________________________________
      // when moving to a coarse level tmax will change only in the direction
      // the ray is moving
      if (jumpFinetoCoarserLevel || jumpCoarsetoCoarserLevel) {
        sycl::double3 dx = d_levels[L].Dx;
        double rayDx_Level = ray_location[dir] - (CC_pos[dir] - 0.5 * dx[dir]);
        double tMax_tmp = (sign[dir] * dx[dir] - rayDx_Level) * inv_ray_direction[dir];

        tMaxV = tMaxV_prev;
        tMaxV[dir] += tMax_tmp;
      }

      // if the cell isn't a flow cell then terminate the ray
      size_t cur_idx = cur.x() - cellType_offset[L].x() + cellType_size[L].x() * (cur.y() - cellType_offset[L].y() +
               (cur.z() - cellType_offset[L].z()) * cellType_size[L].y());
      in_domain = in_domain && ((cellType_ptr[L])[cur_idx] == d_flowCell);

      int abskg_prevcell_idx = prevCell.x() - abskg_offset[prevLev].x() +
          abskg_size[prevLev].x() * (prevCell.y() - abskg_offset[prevLev].y() +
               (prevCell.z() - abskg_offset[prevLev].z()) * abskg_size[prevLev].y());
      optical_thickness += (abskg_ptr[prevLev])[abskg_prevcell_idx] * distanceTraveled;

      double expOpticalThick = sycl::exp(-optical_thickness);

      int sigmaT4OverPi_prevcell_idx = prevCell.x() - sigmaT4OverPi_offset[prevLev].x() +
	sigmaT4OverPi_size[prevLev].x() *
	(prevCell.y() - sigmaT4OverPi_offset[prevLev].y() +
	 (prevCell.z() - sigmaT4OverPi_offset[prevLev].z()) *
	 sigmaT4OverPi_size[prevLev].y());
      sumI += (sigmaT4OverPi_ptr[prevLev])[sigmaT4OverPi_prevcell_idx] *
              (expOpticalThick_prev - expOpticalThick) * fs;

      expOpticalThick_prev = expOpticalThick;

    } // end domain while loop.  ++++++++++++++
    //__________________________________
    //
    int wallEmi_idx = cur.z() - abskg_offset[L].z() +
                         abskg_size[L].z() * (cur.y() - abskg_offset[L].y() +
                                              (cur.x() - abskg_offset[L].x()) *
                                                  abskg_size[L].y());
    double wallEmissivity = (abskg_ptr[L])[wallEmi_idx];

    // Ensure wall emissivity doesn't exceed one.
    wallEmissivity = (wallEmissivity > 1.0) ? 1.0 : wallEmissivity;

    intensity = sycl::exp(-optical_thickness);

    size_t tmp_idx =
        cur.z() - sigmaT4OverPi_offset[L].z() +
        sigmaT4OverPi_size[L].z() * (cur.y() - sigmaT4OverPi_offset[L].y() +
                                     (cur.x() - sigmaT4OverPi_offset[L].x()) *
                                         sigmaT4OverPi_size[L].y());
    sumI += wallEmissivity * (sigmaT4OverPi_ptr[L])[tmp_idx] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if (!RT_flags_allowReflect) {
      intensity = 0;
    }

    //__________________________________
    //  Reflections
    if ((intensity > RT_flags_threshold) && RT_flags_allowReflect) {
      size_t abs_idx = cur.z() - abskg_offset[L].z() +
                       abskg_size[L].z() * (cur.y() - abskg_offset[L].y() +
                                            (cur.x() - abskg_offset[L].x()) *
                                                abskg_size[L].y());

      reflect(fs, cur, prevCell, (abskg_ptr[L])[abs_idx], in_domain, step[dir],
              sign[dir], ray_direction[dir]);
      ++nReflect;
    }
  } // threshold while loop.
} // end of updateSumI_MLDevice function

//______________________________________________________________________
// Returns random number between 0 & 1.0 including 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation
//______________________________________________________________________
double randDblDevice(oneapi::mkl::rng::device::uniform<double> *distr,
		     oneapi::mkl::rng::device::philox4x32x10<1> *engine) {

#ifdef FIXED_RANDOM_NUM
  return 0.3;
#else
  double val = oneapi::mkl::rng::device::generate(*distr, *engine);
  return (double)val * (1.0/4294967295.0);  
#endif
}

//______________________________________________________________________
// Returns random number between 0 & 1.0 excluding 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation
//______________________________________________________________________

double randDblExcDevice(oneapi::mkl::rng::device::uniform<double> *distr,
			oneapi::mkl::rng::device::philox4x32x10<1> *engine) {
#ifdef FIXED_RANDOM_NUM
  return 0.3;
#else
  // call generate function to obtain scalar random number
  double val = oneapi::mkl::rng::device::generate(*distr, *engine);
  return ((double)val + 0.5) * (1.0/4294967295.0);  
#endif
}

//______________________________________________________________________
//  Each thread gets same seed, a different sequence number, no offset
//  This will create repeatable results.
void setupRandNumsSeedAndSequences(int numStates,
				   unsigned long long patchID,
                                   unsigned long long curTimeStep) {
  // Generate random numbers using curand_init().

  // Format is curand_init(seed, sequence, offset, state);

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
  // http://docs.nvidia.com/cuda/curand/device-api-overview.html#axzz4SPy8xMuj

  // For RMCRT we will take the tradeoff of possibly having statistically
  // correlated values over the 300 millisecond hit.

  // Generate what should be a unique seed.  To get a unique number the code
  // below computes a tID which is a combination of a patchID, threadID, and the
  // current timestep. This uses the left 20 bits from the patchID, the next 20
  // bits from the curTimeStep and the last 24 bits from the indexId.  Combined
  // that should be unique.

  // this is from CUDA
  // //Standard CUDA way of computing a threadID
  // int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  // int threadId = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  // unsigned long long tID = (((patchID & 0xFFFFF) << 44) | ((curTimeStep& 0xFFFFF) << 24) |  (threadId & 0xFFFFFF));
  // curand_init(tID, threadId, 0, &randNumStates[threadId]);

  //If you want to take the 300 millisecond hit, use this line below instead.
  //curand_init(1234, tID, 0, &randNumStates[threadId]);


  // TODO: for sycl
  // oneapi::mkl::rng::device::philox4x32x10<> engine(seed, 0);
  // oneapi::mkl::rng::device::uniform<> distr;

}

//______________________________________________________________________
//
template <class T>
void launchRayTraceKernel(DetailedTask *dtask, sycl::range<2> &dimGrid,
                          sycl::range<2> &dimBlock, const int matlIndx,
                          levelParams level, patchParams patch,
                          gpuStream_t *stream, RMCRT_flags RT_flags,
                          int curTimeStep, GPUDataWarehouse *abskg_gdw,
                          GPUDataWarehouse *sigmaT4_gdw,
                          GPUDataWarehouse *cellType_gdw,
                          GPUDataWarehouse *new_gdw) {

    auto rayTraceKernelEvent = stream->parallel_for(
      sycl::nd_range<2>(dimGrid * dimBlock, dimBlock), [=](auto item) {
        rayTraceKernel<T>(item, matlIndx, level, patch, RT_flags,
                          curTimeStep, abskg_gdw, sigmaT4_gdw,
                          cellType_gdw, new_gdw);
      });

}

//______________________________________________________________________
//
template <class T>
void launchRayTraceDataOnionKernel(
  DetailedTask *dtask, sycl::range<1> &dimGrid, sycl::range<1> &dimBlock,
  int matlIndex, patchParams patch, gridParams gridP, levelParams *levelP,
  sycl::int3 fineLevel_ROI_Lo, sycl::int3 fineLevel_ROI_Hi,
  gpuStream_t *stream, unsigned short deviceID, RMCRT_flags RT_flags, int curTimeStep,
  GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
  GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *new_gdw) {

  // copy regionLo & regionHi to device memory
  size_t size = d_MAXLEVELS * sizeof(sycl::int3);
  sycl::int3* dev_regionLo = static_cast<sycl::int3*>( dtask->addTempGpuMemoryToBeFreedOnCompletion(deviceID, size) );
  sycl::int3* dev_regionHi = static_cast<sycl::int3*>( dtask->addTempGpuMemoryToBeFreedOnCompletion(deviceID, size) );

  // More GPU stuff to allow kernel/copy overlapping
  sycl::int3 *myLo = new sycl::int3[d_MAXLEVELS];
  sycl::int3 *myHi = new sycl::int3[d_MAXLEVELS];
  dtask->addTempHostMemoryToBeFreedOnCompletion(myLo);
  dtask->addTempHostMemoryToBeFreedOnCompletion(myHi);

  for (int l = 0; l < gridP.maxLevels; ++l) {
    myLo[l] = levelP[l].regionLo; // never use levelP regionLo or hi in the kernel.
    myHi[l] = levelP[l].regionHi; // They are different on each patch
  }

  // cudaMemcpyAsync, HostToDevice
  auto evt_copy1 = stream->memcpy(dev_regionLo, myLo, size);
  auto evt_copy2 = stream->memcpy(dev_regionHi, myHi, size);

  //__________________________________
  // setup random number generator states on the device, 1 for each thread

  sycl::buffer<levelParams> dev_d_levels(levelP, sycl::range<1>(gridP.maxLevels));

  int RTflags_nDivQRays          = RT_flags.nDivQRays;
  int RTflags_nFluxRays          = RT_flags.nFluxRays;
  bool RTflags_CCRays            = RT_flags.CCRays;
  bool RTflags_modifies_divQ     = RT_flags.modifies_divQ;
  bool RTflags_solveBoundaryFlux = RT_flags.solveBoundaryFlux;
  bool RTflags_solveDivQ         = RT_flags.solveDivQ;
  double RTflags_threshold       = RT_flags.threshold;
  bool RTflags_allowReflect      = RT_flags.allowReflect;
  bool RTflags_usingFloats       = RT_flags.usingFloats;
  int RT_flags_whichROI_algo     = RT_flags.whichROI_algo;
  int RT_flags_rayDirSampleAlgo  = RT_flags.rayDirSampleAlgo;
  unsigned int RTflags_startCell = RT_flags.startCell;
  unsigned int RTflags_endCell   = RT_flags.endCell;

  auto rayTraceDataOnionKernelEvent = stream->submit([&](sycl::handler &cgh) {
    cgh.depends_on({evt_copy1, evt_copy2});

    // copy levelParams array to constant-memory on device
    sycl::accessor<levelParams, 1, sycl::access::mode::read,
		   sycl::access::target::constant_buffer>
      acc_d_levels(dev_d_levels, cgh);

    cgh.parallel_for(sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
		     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
		       rayTraceDataOnionKernel<T>(
			 item, sycl::constant_ptr<levelParams>(acc_d_levels),
			 matlIndex, patch, gridP,
			 fineLevel_ROI_Lo, fineLevel_ROI_Hi,
			 dev_regionLo, dev_regionHi,
			 RTflags_startCell,
			 RTflags_endCell,
			 RTflags_nDivQRays,
			 RTflags_nFluxRays,
			 RTflags_CCRays,
			 RTflags_modifies_divQ,
			 RTflags_solveBoundaryFlux,
			 RTflags_solveDivQ,
			 RTflags_threshold,
			 RTflags_allowReflect,
			 RTflags_usingFloats,
			 RT_flags_whichROI_algo,
			 RT_flags_rayDirSampleAlgo,
			 curTimeStep,
			 abskg_gdw, sigmaT4_gdw, cellType_gdw, new_gdw);
		     });
  });
}

//______________________________________________________________________
//  Explicit template instantiations
template void launchRayTraceKernel<double>(
    DetailedTask *dtask, sycl::range<2> &dimGrid, sycl::range<2> &dimBlock,
    const int matlIndx, levelParams level, patchParams patch,
    gpuStream_t *stream, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *new_gdw);

template void launchRayTraceKernel<float>(
    DetailedTask *dtask, sycl::range<2> &dimGrid, sycl::range<2> &dimBlock,
    const int matlIndx, levelParams level, patchParams patch,
    gpuStream_t *stream, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *celltype_gdw, GPUDataWarehouse *new_gdw);

//______________________________________________________________________
//
template void launchRayTraceDataOnionKernel<double>(
    DetailedTask *dtask, sycl::range<1> &dimGrid, sycl::range<1> &dimBlock,
    int matlIndex, patchParams patch, gridParams gridP, levelParams *levelP,
    sycl::int3 fineLevel_ROI_Lo, sycl::int3 fineLevel_ROI_Hi,
    gpuStream_t *stream, unsigned short deviceID, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *new_gdw);

template void launchRayTraceDataOnionKernel<float>(
    DetailedTask *dtask, sycl::range<1> &dimGrid, sycl::range<1> &dimBlock,
    int matlIndex, patchParams patch, gridParams gridP, levelParams *levelP,
    sycl::int3 fineLevel_ROI_Lo, sycl::int3 fineLevel_ROI_Hi,
    gpuStream_t *stream, unsigned short deviceID, RMCRT_flags RT_flags, int curTimeStep,
    GPUDataWarehouse *abskg_gdw, GPUDataWarehouse *sigmaT4_gdw,
    GPUDataWarehouse *cellType_gdw, GPUDataWarehouse *new_gdw);

} // end namespace Uintah
