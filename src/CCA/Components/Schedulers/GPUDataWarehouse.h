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

/* GPU DataWarehouse device & host access*/

#pragma once

#include <sci_defs/gpu_defs.h>

#include <Core/Disclosure/TypeDescription.h>

#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUPerPatch.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GPUStencil7.h>
#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Parallel/MasterLock.h>

#include <map>    //for host code only.
#include <memory> //for the shared_ptr code
#include <string>
#include <vector>

#define MAX_VARDB_ITEMS                                                        \
  10000000 // Due to how it's allocated, it will never use up this much space.
           // Only a very small fraction of it.
#define MAX_MATERIALSDB_ITEMS 20
#define MAX_LEVELDB_ITEMS 20 // TODO: Is this needed?
#define MAX_NAME_LENGTH 32   // How big a particular label can be.

#define LEVEL_PATCH_ID                                                         \
  = -99999999; // A sentinel value used for a patch ID when we're storing a
               // region of patches for a level instead of a regular patch.
namespace Uintah {

enum materialType {
  IDEAL_GAS = 0
  //  , HARD_SPHERE_GAS_EOS = 1
  //  , TST_EOS = 2
  //  , JWL_EOS = 3
  //  , JWLC_EOS = 4
  //  , MURNAGHAN_EOS = 5
  //  , BIRCH_MURNAGHAN_EOS = 6
  //  , GRUNEISEN_EOS = 7
  //  , TILLOTSON_EOS = 8
  //  , THOMSEN_HARTKA_WATER_EOS = 9
  //  , KNAUSS_SEA_WATER_EOS = 10
  //  , KUMARI_DASS_EOS = 11
};

class OnDemandDataWarehouse;
class TypeDescription;

class GPUDataWarehouse {

public:
  virtual ~GPUDataWarehouse(){};

  enum GhostType {
    None,
    AroundNodes,
    AroundCells,
    AroundFacesX,
    AroundFacesY,
    AroundFacesZ,
    AroundFaces,
    numGhostTypes // 7
  };

  struct materialItem {
    materialType material;
    char simulationType[MAX_NAME_LENGTH]; // ICE, MPM, etc.  Currently unused
  };

  // The dataItem can hold two kinds of data.
  // The first is information related to a regular data variable.
  // The second is information indicating how a ghost cell should be copied from
  // one var to another The two types of information are stored in one struct to
  // allow us to make the size of the GPU data warehouse dynamic.  For small
  // problems the GPUDW can be small, and large problems it can be large. The
  // biggest problem is that multiple CPU threads will be adding to the size of
  // the GPUDW IF we had two separate structs, then we would have to let both
  // grow independently and it would require two copies to the GPU DW instead of
  // one. So the solution is to allocate a large buffer of possible GPUDW data
  // in init_device(), one on the host RAM and device RAM Then on the CPU side a
  // thread running a task will collect various dataItems, and just before it
  // sends it to the GPU DW, it will dump its results into the host side buffer
  // (using locks). Then it can either copy in only as much of the GPUDW as
  // needed, instead of the entire buffer.

  struct VarItem {
    GhostType gtype;
    unsigned int numGhostCells;
    bool staging;
  };

  struct GhostItem {
    // This assumes the ghost cell is already in the GPU's memory
    // We only need to know the source patch, the destination patch
    // the material number, and the shared coordinates.
    int3 sharedLowCoordinates;
    int3 sharedHighCoordinates;

    // Wasatch has virtual patches, which come as a result of periodic boundary
    // cells, which wrap around on each other.  (Like a label wrapping around a
    // soup can, but for all boundary faces of the grid).  While they're
    // technically sharing the same coordinates (just wrapped around once), from
    // our perspective we need their actual indexes this offset helps us get
    // that.
    int3 virtualOffset;

    // So we can look up the size and offset information in the d_varDB
    int dest_varDB_index; // Will be set to -1 if this d_varDB item isn't a
                          // ghost cell item.
  };

  struct dataItem {

    char label[MAX_NAME_LENGTH]; // VarLabel name
    int domainID;                // a Patch ID (d_VarDB)
    int matlIndx;                // the material index
    int levelIndx;               // level the variable resides on (AMR)
    int3 var_offset;             // offset
    int3 var_size;               // dimensions of GPUGridVariable
    void *var_ptr;               // raw pointer to the memory
    unsigned int sizeOfDataType; // the memory size of a single data element.
    VarItem varItem; // If the item is holding variable data, remaining info is
                     // found in here
    GhostItem ghostItem; // If the item contains only ghost cell copying meta
                         // data, its info is found in here
  };

  struct contiguousArrayInfo {
    void *allocatedDeviceMemory;
    void *allocatedHostMemory;
    std::size_t sizeOfAllocatedMemory;
    std::size_t assignedOffset;
    std::size_t copiedOffset;
    // The default constructor
    contiguousArrayInfo() {
      allocatedDeviceMemory = nullptr;
      allocatedHostMemory = nullptr;
      sizeOfAllocatedMemory = 0;
      assignedOffset = 0; // To keep up to the point where data has been "put".
                          // Computes data will be assigned
      copiedOffset = 0;   // To keep up to the point where data will need to be
                          // copied.  Required data will be copied
    }
    // Overloaded constructor
    contiguousArrayInfo(double *allocatedDeviceMemory,
                        double *allocatedHostMemory,
                        std::size_t sizeOfAllocatedMemory) {
      this->allocatedDeviceMemory = allocatedDeviceMemory;
      this->allocatedHostMemory = allocatedHostMemory;
      this->sizeOfAllocatedMemory = sizeOfAllocatedMemory;
      assignedOffset = 0; // To keep up to the point where data has been "put"
      copiedOffset = 0;   // To keep up to the point where data has been copied.
    }
  };

  // This status is for concurrency.  This enum largely follows a model of
  // "action -> state". For example, allocating -> allocated.  The idea is that
  // only one thread should be able to claim moving into an action, and that
  // winner should be responsible for setting it into the state. When it hits
  // the state, other threads can utilize the variable.
  enum status {
    UNALLOCATED = 0x00000000,
    ALLOCATING = 0x00000001,
    ALLOCATED = 0x00000002,
    COPYING_IN = 0x00000004,
    VALID = 0x00000008, // For when a variable has its data, this excludes any
                        // knowledge of ghost cells.
    AWAITING_GHOST_COPY =
        0x00000010, // For when when we know a variable is awaiting ghost cell
                    // data It is possible for VALID bit set to 0 or 1 with this
                    // bit set, meaning we can know a variable is awaiting ghost
                    // copies but we don't know from this bit alone if the
                    // variable is valid yet.
    VALID_WITH_GHOSTS =
        0x00000020, // For when a variable has its data and it has its ghost
                    // cells Note: Change to just GHOST_VALID?  Meaning ghost
                    // cells could be valid but the non ghost part is unknown?
    DEALLOCATING =
        0x00000040, // TODO: REMOVE THIS WHEN YOU CAN, IT'S NOT OPTIMAL DESIGN.
    FORMING_SUPERPATCH =
        0x00000080, // As the name suggests, when a number of individual patches
                    // are being formed into a superpatch, there is a period of
                    // time which other threads should wait until all patches
                    // have been processed.
    SUPERPATCH =
        0x00000100, // Indicates this patch is allocated as part of a
                    // superpatch. At the moment superpatches is only
                    // implemented for entire domain levels.  But it seems to
                    // make the most sense to have another set of logic in
                    // level.cc which subdivides a level into superpatches. If
                    // this bit is set, you should find the lowest numbered
                    // patch ID first and start with concurrency reads/writes
                    // there.  (Doing this avoids the Dining Philosopher's
                    // problem.
    UNKNOWN = 0x00000200
  }; // Remove this when you can, unknown can be dangerous.
     // It's only here to help track some host variables

  // LEFT_SIXTEEN_BITS                         //Use the other 16 bits as a
  // usage counter. If it is zero we could deallocate.

  typedef int atomicDataStatus;

  //    0                   1                   2                   3
  //    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //   |    16-bit reference counter   |  unsued   | | | | | | | | | | |
  //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

  // left sixteen bits is a 16-bit integer reference counter.

  // Not allocated/Invalid = If the value is 0x00000000

  // Allocating                = bit 31 - 0x00000001
  // Allocated                 = bit 30 - 0x00000002
  // Copying in                = bit 29 - 0x00000004
  // Valid                     = bit 28 - 0x00000008
  // awaiting ghost data       = bit 27 - 0x00000010
  // Valid with ghost cells    = bit 26 - 0x00000020
  // Deallocating              = bit 25 - 0x00000040
  // Superpatch                = bit 24 - 0x00000080
  // Unknown                = bit 23 - 0x00000080

  // With this approach we can allow for multiple copy outs, but only one copy
  // in. We should never attempt to copy unless the status is odd (allocated) We
  // should never copy out if the status isn't valid.

  struct stagingVar {

    int3 device_offset;
    int3 device_size;
    // This so it can be used in an STL map
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    bool operator<(const stagingVar &rhs) const {
      if (this->device_offset.x < rhs.device_offset.x) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x &&
                 (this->device_offset.y < rhs.device_offset.y)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x &&
                 (this->device_offset.y == rhs.device_offset.y) &&
                 (this->device_offset.z < rhs.device_offset.z)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x &&
                 (this->device_offset.y == rhs.device_offset.y) &&
                 (this->device_offset.z == rhs.device_offset.z) &&
                 (this->device_size.x < rhs.device_size.x)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x &&
                 (this->device_offset.y == rhs.device_offset.y) &&
                 (this->device_offset.z == rhs.device_offset.z) &&
                 (this->device_size.x == rhs.device_size.x) &&
                 (this->device_size.y < rhs.device_size.y)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x &&
                 (this->device_offset.y == rhs.device_offset.y) &&
                 (this->device_offset.z == rhs.device_offset.z) &&
                 (this->device_size.x == rhs.device_size.x) &&
                 (this->device_size.y == rhs.device_size.y) &&
                 (this->device_size.z < rhs.device_size.z)) {
        return true;
      } else {
        return false;
      }
    }
#elif defined(HAVE_SYCL)
    bool operator<(const stagingVar &rhs) const {
      if (this->device_offset.x() < rhs.device_offset.x()) {
        return true;
      } else if (this->device_offset.x() == rhs.device_offset.x() &&
                 (this->device_offset.y() < rhs.device_offset.y())) {
        return true;
      } else if (this->device_offset.x() == rhs.device_offset.x() &&
                 (this->device_offset.y() == rhs.device_offset.y()) &&
                 (this->device_offset.z() < rhs.device_offset.z())) {
        return true;
      } else if (this->device_offset.x() == rhs.device_offset.x() &&
                 (this->device_offset.y() == rhs.device_offset.y()) &&
                 (this->device_offset.z() == rhs.device_offset.z()) &&
                 (this->device_size.x() < rhs.device_size.x())) {
        return true;
      } else if (this->device_offset.x() == rhs.device_offset.x() &&
                 (this->device_offset.y() == rhs.device_offset.y()) &&
                 (this->device_offset.z() == rhs.device_offset.z()) &&
                 (this->device_size.x() == rhs.device_size.x()) &&
                 (this->device_size.y() < rhs.device_size.y())) {
        return true;
      } else if (this->device_offset.x() == rhs.device_offset.x() &&
                 (this->device_offset.y() == rhs.device_offset.y()) &&
                 (this->device_offset.z() == rhs.device_offset.z()) &&
                 (this->device_size.x() == rhs.device_size.x()) &&
                 (this->device_size.y() == rhs.device_size.y()) &&
                 (this->device_size.z() < rhs.device_size.z())) {
        return true;
      } else {
        return false;
      }
    }
#endif // HAVE_CUDA, HAVE_SYCL
  };

  struct stagingVarInfo {
    void *device_ptr{nullptr}; // Where it is on the device
    size_t sizeInBytesDevicePtr; // This is needed for GPU memPool
    int varDB_index;
    atomicDataStatus atomicStatusInHostMemory;
    atomicDataStatus atomicStatusInGpuMemory;
  };

  // Only raw information about the data itself should go here.  Things that
  // should be shared if two patches are in the same superpatch sharing the same
  // data pointer.  (For example, both could have numGhostCells = 1, it wouldn't
  // make sense for one to have numGhostCells = 1 and another = 2 if they're
  // sharing the same data pointer.
  class varInfo {
  public:
    varInfo() {
      atomicStatusInHostMemory = UNALLOCATED;
      atomicStatusInGpuMemory = UNALLOCATED;
      gtype = GhostType::None;
    }
    void *device_ptr{nullptr}; // Where it is on the device
    size_t sizeInBytesDevicePtr; // This is needed for GPU memPool
    int3 device_offset{
        0, 0, 0}; // TODO, split this into a device_low and a device_offset.
                  // Device_low goes here but device_offset should NOT go here
                  // (multiple patches may share a dataInfo, but they should
                  // have distinct offsets
    int3 device_size{0, 0, 0};
    unsigned int sizeOfDataType{0};
    GhostType gtype;
    unsigned int numGhostCells{0};
    atomicDataStatus
        atomicStatusInHostMemory; // Shared_ptr because patches in a superpatch
                                  // share the pointer.
    atomicDataStatus
        atomicStatusInGpuMemory; // TODO, merge into the one above it.

    std::map<stagingVar, stagingVarInfo>
        stagingVars; // When ghost cells in the GPU need to go to another memory
                     // space we will be creating temporary contiguous arrays to
                     // hold that information.  After many iterations of other
                     // attempts, it seems creating a map of staging vars is the
                     // cleanest way to go for data in GPU memory.
  };

  class allVarPointersInfo {
  public:
    allVarPointersInfo() { var = std::make_shared<varInfo>(); }
    std::shared_ptr<varInfo> var;
    int3 device_offset{0, 0, 0};
    int varDB_index{-1}; // Where this also shows up in the varDB.  We can use
                         // this to get the rest of the information we need.
  };

  struct labelPatchMatlLevel {
    std::string label;
    int patchID;
    int matlIndx;
    int levelIndx;
    labelPatchMatlLevel(const char *label, int patchID, int matlIndx,
                        int levelIndx) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndx = matlIndx;
      this->levelIndx = levelIndx;
    }
    // This so it can be used in an STL map
    bool operator<(const labelPatchMatlLevel &right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label &&
                 (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label &&
                 (this->patchID == right.patchID) &&
                 (this->matlIndx < right.matlIndx)) {
        return true;
      } else if (this->label == right.label &&
                 (this->patchID == right.patchID) &&
                 (this->matlIndx == right.matlIndx) &&
                 (this->levelIndx < right.levelIndx)) {
        return true;
      } else {
        return false;
      }
    }
  };

  //______________________________________________________________________
  // GPU GridVariable methods
  HOST_DEVICE void getStagingVar(const GPUGridVariableBase &var,
                                 char const *label, int patchID, int matlIndx,
                                 int levelIndx, int3 offset, int3 size);
  HOST_DEVICE bool stagingVarExists(char const *label, int patchID,
                                    int matlIndx, int levelIndx,
                                    const int3 &offset, const int3 &size);

#ifdef HAVE_SYCL // APIs only makes sense for SYCL-side
  // device-side calls

  // GPUGridVariableBase
  template <typename T, int DIM>
  SYCL_EXTERNAL void
  get(sycl::nd_item<DIM> &item, sycl::device_ptr<T> GPUGridVariableBase_ptr,
      sycl::int3 &GPUGridVariableBase_size,
      sycl::int3 &GPUGridVariableBase_offset, const char *label,
      const int patchID, const int8_t matlIndx, const int8_t levelIndx = 0) {
    // device code
    GPUDataWarehouse::dataItem *ditem = getItem(item, label, patchID, matlIndx, levelIndx);
    if (ditem != nullptr) {
      GPUGridVariableBase_ptr = static_cast<T *>(ditem->var_ptr);
      GPUGridVariableBase_size = ditem->var_size;
      GPUGridVariableBase_offset = ditem->var_offset;
    }
  }
  // GPUReductionVariableBase, GPUPerPatchBase
  template <typename T, int DIM>
  SYCL_EXTERNAL void
  get(sycl::nd_item<DIM> &item, sycl::device_ptr<T> gpu_ptr,
      const char *label, const int patchID,
      const int8_t matlIndx, const int8_t levelIndx = 0) {
    // device code
    GPUDataWarehouse::dataItem *ditem = getItem(item, label, patchID, matlIndx, levelIndx);
    if (ditem != nullptr) {
      gpu_ptr = static_cast<T *>(ditem->var_ptr);
    }
  }

  template <typename T, int DIM>
  SYCL_EXTERNAL void
  getModifiable(sycl::nd_item<DIM> &item,
                sycl::device_ptr<T> GPUGridVariableBase_ptr,
                sycl::int3 &GPUGridVariableBase_size,
                sycl::int3 &GPUGridVariableBase_offset,
                char const *label, const int patchID,
                const int8_t matlIndx,
                const int8_t levelIndx = 0) {
    // device code
    GPUDataWarehouse::dataItem *ditem = getItem(item, label, patchID, matlIndx, levelIndx);
    if (ditem != nullptr) {
      GPUGridVariableBase_ptr = static_cast<T *>(ditem->var_ptr);
      GPUGridVariableBase_size = ditem->var_size;
      GPUGridVariableBase_offset = ditem->var_offset;
    }
  }
  // GPUReductionVariableBase, GPUPerPatchBase
  template <typename T, int DIM>
  SYCL_EXTERNAL void
  getModifiable(sycl::nd_item<DIM> &item, sycl::device_ptr<T> gpu_ptr,
                char const *label, const int patchID, const int8_t matlIndx,
                const int8_t levelIndx = 0) {
    // device code
    GPUDataWarehouse::dataItem *ditem = getItem(item, label, patchID, matlIndx, levelIndx);
    if (ditem != nullptr) {
      gpu_ptr = static_cast<T *>(ditem->var_ptr);
    }
  }

  template <typename T, int DIM>
  SYCL_EXTERNAL void getLevel(sycl::nd_item<DIM> &item,
                              sycl::device_ptr<T> GPUGridVariableBase_ptr,
                              sycl::int3 &GPUGridVariableBase_size,
                              sycl::int3 &GPUGridVariableBase_offset,
                              const char *label, const int8_t matlIndx,
                              const int8_t levelIndx) {
    // device code
    get(item, GPUGridVariableBase_ptr, GPUGridVariableBase_size,
        GPUGridVariableBase_offset, label, -99999999, matlIndx, levelIndx);
  }

  // host-side calls
  template <typename T>
  void get(const T &var, char const *label, const int patchID,
           const int8_t matlIndx, const int8_t levelIndx = 0) {
    // host code
    varLock->lock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    auto it = varPointers->find(lpml);
    if (it != varPointers->end()) {
      allVarPointersInfo vp = it->second;
      if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
          var.setArray3(vp.var->device_offset, vp.var->device_size,
                        vp.var->device_ptr);
        }
      else {
        var.setData(vp.var->device_ptr);
      }
    }
    varLock->unlock();
  }

  template <typename T>
  void getModifiable(const T &var, char const *label, const int patchID,
                     const int8_t matlIndx, const int8_t levelIndx = 0) {
    // host code
    varLock->lock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    auto it = varPointers->find(lpml);
    if (it != varPointers->end()) {
      allVarPointersInfo vp = it->second;
      if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
          var.setArray3(vp.var->device_offset, vp.var->device_size,
                        vp.var->device_ptr);
        }
      else {
        var.setData(vp.var->device_ptr);
      }
    }
    varLock->unlock();
  }

  void put(void *GPUGridVariableBase_ptr,
           const sycl::int3 &GPUGridVariableBase_size,
           const sycl::int3 &GPUGridVariableBase_offset,
           const std::size_t allocMemSize,
           std::size_t sizeOfDataType,
           char const *label, int patchID, int matlIndx, int levelIndx = 0,
           bool staging = false, GhostType gtype = None, int numGhostCells = 0);
  // SYCL varient: GPUReductionVariableBase, GPUPerPatchBase
  void put(void *GPUPerPatchBase_ptr, std::size_t sizeOfDataType,
           char const *label, int patchID, int matlIndx, int levelIndx = 0);
  void put(GPUGridVariableBase &var, std::size_t sizeOfDataType,
           char const *label, int patchID, int matlIndx, int levelIndx = 0,
           bool staging = false, GhostType gtype = None, int numGhostCells = 0);
  void put(GPUReductionVariableBase &var, std::size_t sizeOfDataType,
           char const *label, int patchID, int matlIndx, int levelIndx = 0);
  void put(GPUPerPatchBase &var, std::size_t sizeOfDataType, char const *label,
           int patchID, int matlIndx, int levelIndx = 0);

  void allocateAndPut(const Uintah::TypeDescription::Type &type,
                      void *&GPUGridVariableBase_ptr,
                      sycl::int3 &GPUGridVariableBase_size,
                      sycl::int3 &GPUGridVariableBase_offset, char const *label,
                      int patchID, int matlIndx, int levelIndx, bool staging,
                      const sycl::int3 &low, const sycl::int3 &high,
                      std::size_t sizeOfDataType, GhostType gtype = None,
                      int numGhostCells = 0);
  // for both GPUGridVariableBase, GPUPerPatchBase
  void allocateAndPut(const Uintah::TypeDescription::Type &type,
                      void *&GPUReductionPerPatchVariableBase,
                      char const *label, int patchID, int matlIndx,
                      int levelIndx, std::size_t sizeOfDataType);
  void allocateAndPut(GPUGridVariableBase &var, char const *label, int patchID,
                      int matlIndx, int levelIndx, bool staging,
                      const sycl::int3 &low, const sycl::int3 &high,
                      std::size_t sizeOfDataType, GhostType gtype = None,
                      int numGhostCells = 0);
  void allocateAndPut(GPUReductionVariableBase &var, char const *label,
                      int patchID, int matlIndx, int levelIndx,
                      std::size_t sizeOfDataType);
  void allocateAndPut(GPUPerPatchBase &var, char const *label, int patchID,
                      int matlIndx, int levelIndx, std::size_t sizeOfDataType);

  bool transferFrom(gpuStream_t *stream, GPUDataWarehouse *from,
                    char const *label, int patchID, int matlIndx,
                    int levelIndx);

#else // for CUDA, HIP and/or CPU

  template <typename T>
  HOST_DEVICE void get(const T &var, char const *label, const int patchID,
                       const int8_t matlIndx, const int8_t levelIndx = 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // device code
    GPUDataWarehouse::dataItem *item = getItem(label, patchID, matlIndx, levelIndx);
    if (item) {
      if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
        var.setArray3(item->var_offset, item->var_size, item->var_ptr);
      } else if constexpr (std::is_same_v<T, GPUReductionVariableBase> ||
                           std::is_same_v<T, GPUPerPatchBase>) {
        var.setData(item->var_ptr);
      }
    }
#else
    // host code
    varLock->lock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      allVarPointersInfo vp = varPointers->at(lpml);
      if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
        var.setArray3(vp.var->device_offset, vp.var->device_size,
                      vp.var->device_ptr);
      } else if constexpr (std::is_same_v<T, GPUReductionVariableBase> ||
                           std::is_same_v<T, GPUPerPatchBase>) {
        var.setData(vp.var->device_ptr);
      }
    }
    varLock->unlock();
#endif
  }

  template <typename T>
  HOST_DEVICE void getModifiable(T &var, char const *label, const int patchID,
                                 const int8_t matlIndx,
                                 const int8_t levelIndx = 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // device code
    GPUDataWarehouse::dataItem *item = getItem(label, patchID, matlIndx, levelIndx);
    if (item) {
      if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
        var.setArray3(item->var_offset, item->var_size, item->var_ptr);
      } else if constexpr (std::is_same_v<T, GPUReductionVariableBase> ||
                           std::is_same_v<T, GPUPerPatchBase>) {
        var.setData(item->var_ptr);
      }
    }
#else
    // host code
    varLock->lock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
    if (it != varPointers->end()) {
      if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
        var.setArray3(it->second.var->device_offset, it->second.var->device_size,
                      it->second.var->device_ptr);
      } else if constexpr (std::is_same_v<T, GPUReductionVariableBase> ||
                           std::is_same_v<T, GPUPerPatchBase>) {
        var.setData(it->second.var->device_ptr);
      }
    }
    varLock->unlock();
#endif
  }

  void put(GPUGridVariableBase &var, std::size_t sizeOfDataType,
           char const *label, int patchID, int matlIndx, int levelIndx = 0,
           bool staging = false, GhostType gtype = None, int numGhostCells = 0);
  void put(GPUReductionVariableBase &var, std::size_t sizeOfDataType,
           char const *label, int patchID, int matlIndx, int levelIndx = 0);
  void put(GPUPerPatchBase &var, std::size_t sizeOfDataType, char const *label,
           int patchID, int matlIndx, int levelIndx = 0);

  void allocateAndPut(GPUGridVariableBase &var, char const *label, int patchID,
                      int matlIndx, int levelIndx, bool staging,
                      const int3 &low, const int3 &high,
                      std::size_t sizeOfDataType, GhostType gtype = None,
                      int numGhostCells = 0);
  void allocateAndPut(GPUReductionVariableBase &var, char const *label,
                      int patchID, int matlIndx, int levelIndx,
                      std::size_t sizeOfDataType);
  void allocateAndPut(GPUPerPatchBase &var, char const *label, int patchID,
                      int matlIndx, int levelIndx, std::size_t sizeOfDataType);

  bool transferFrom(gpuStream_t *stream, GPUGridVariableBase &var_source,
                    GPUDataWarehouse *from, char const *label, int patchID,
                    int matlIndx, int levelIndx);
#endif


  HOST_DEVICE void getLevel(const GPUGridVariableBase &var, char const *label,
                            const int8_t matlIndx, const int8_t levelIndx) {
    get(var, label, -99999999, matlIndx, levelIndx);
  }

  void copySuperPatchInfo(char const *label, int superPatchBaseID,
                          int superPatchDestinationID, int matlIndx,
                          int levelIndx);

  void putUnallocatedIfNotExists(char const *label, int patchID, int matlIndx,
                                 int levelIndx, bool staging,
                                 const int3 &offset, const int3 &size);
  void copyItemIntoTaskDW(GPUDataWarehouse *hostSideGPUDW, char const *label,
                          int patchID, int matlIndx, int levelIndx,
                          bool staging, int3 offset, int3 size);

  void allocate(const char *indexID, std::size_t size);

  //______________________________________________________________________
  // GPU DataWarehouse support methods
  // HOST_DEVICE bool existContiguously(char const* label, int patchID, int
  // matlIndx, int levelIndx, bool staging, int3 host_size, int3 host_offset);
  // HOST_DEVICE bool existsLevelDB( char const* name, int matlIndx, int
  // levelIndx);       // levelDB HOST_DEVICE bool removeLevelDB( char const*
  // name, int matlIndx, int levelIndx);
  bool remove(char const *label, int patchID, int matlIndx, int levelIndx);
  void *getPlacementNewBuffer();
  void syncto_device(gpuStream_t *gpu_stream);
  void clear();
  void deleteSelfOnDevice();
  GPUDataWarehouse *getdevice_ptr() { return d_device_copy; };
  void setDebug(bool s) { d_debug = s; }

  //______________________________________________________________________
  // Additional support methods
  void putMaterials(std::vector<std::string> materials);
  HOST_DEVICE materialType getMaterial(int i) const;
  HOST_DEVICE int getNumMaterials() const;
  void putGhostCell(char const *label, int sourcePatchID, int destPatchID,
                    int matlIndx, int levelIndx, bool sourceStaging,
                    bool deststaging, const int3 &varOffset,
                    const int3 &varSize, const int3 &sharedLowCoordinates,
                    const int3 &sharedHighCoordinates,
                    const int3 &virtualOffset);

  bool areAllStagingVarsValid(char const *label, int patchID, int matlIndx,
                              int levelIndx);

  // atomicDataStatus getStatus(std::shared_ptr<atomicDataStatus>&
  // status);
  std::string getDisplayableStatusCodes(atomicDataStatus &status);
  void getStatusFlagsForVariableOnGPU(
      bool &correctSize, bool &allocating, bool &allocated, bool &copyingIn,
      bool &validOnGPU, bool &gatheringGhostCells,
      bool &validWithGhostCellsOnGPU, bool &deallocating,
      bool &formingSuperPatch, bool &superPatch, char const *label,
      const int patchID, const int matlIndx, const int levelIndx,
      const int3 &offset, const int3 &size);

  bool compareAndSwapAllocating(atomicDataStatus &status);
  bool compareAndSwapAllocate(atomicDataStatus &status);
  bool checkAllocated(atomicDataStatus &status);
  bool checkValid(atomicDataStatus &status);

  bool isAllocatedOnGPU(char const *label, int patchID, int matlIndx,
                        int levelIndx);
  bool isAllocatedOnGPU(char const *label, int patchID, int matlIndx,
                        int levelIndx, int3 offset, int3 size);
  bool isValidOnGPU(char const *label, int patchID, int matlIndx,
                    int levelIndx);
  bool compareAndSwapSetValidOnGPU(char const *const label, const int patchID,
                                   const int matlIndx, const int levelIndx);
  bool compareAndSwapSetValidOnGPUStaging(char const *label, int patchID,
                                          int matlIndx, int levelIndx,
                                          const int3 &offset, const int3 &size);
  bool dwEntryExistsOnCPU(char const *label, int patchID, int matlIndx,
                          int levelIndx);
  bool isValidOnCPU(char const *label, const int patchID, const int matlIndx,
                    const int levelIndx);
  bool compareAndSwapSetValidOnCPU(char const *const label, int patchID,
                                   int matlIndx, int levelIndx);

  bool compareAndSwapAwaitingGhostDataOnGPU(char const *label, int patchID,
                                            int matlIndx, int levelIndx);
  bool compareAndSwapCopyingIntoGPU(char const *label, int patchID,
                                    int matlIndx, int levelIndx);
  bool compareAndSwapCopyingIntoCPU(char const *label, int patchID,
                                    int matlIndx, int levelIndx);
  bool compareAndSwapCopyingIntoGPUStaging(char const *label, int patchID,
                                           int matlIndx, int levelIndx,
                                           int3 offset, int3 size);
  bool isValidWithGhostsOnGPU(char const *label, int patchID, int matlIndx,
                              int levelIndx);
  void setValidWithGhostsOnGPU(char const *label, int patchID, int matlIndx,
                               int levelIndx);

  bool compareAndSwapFormASuperPatchGPU(char const *label, int patchID,
                                        int matlIndx, int levelIndx);
  bool compareAndSwapSetSuperPatchGPU(char const *label, int patchID,
                                      int matlIndx, int levelIndx);
  bool isSuperPatchGPU(char const *label, int patchID, int matlIndx,
                       int levelIndx);
  void setSuperPatchLowAndSize(char const *const label, const int patchID,
                               const int matlIndx, const int levelIndx,
                               const int3 &low, const int3 &size);
  bool compareAndSwapDeallocating(atomicDataStatus &status);
  bool compareAndSwapDeallocate(atomicDataStatus &status);

  // This and the function below go through the d_ghostCellData array and copies
  // data into the correct destination GPU var.  This would be the final step of
  // a GPU ghost cell transfer.
  void copyGpuGhostCellsToGpuVarsInvoker(gpuStream_t *stream);
#ifdef HAVE_SYCL
  void copyGpuGhostCellsToGpuVars(sycl::nd_item<3> &item);
#else
  __device__ void copyGpuGhostCellsToGpuVars();
#endif
  bool ghostCellCopiesNeeded();
  void getSizes(int3 &low, int3 &high, int3 &siz, GhostType &gtype,
                int &numGhostCells, char const *label, int patchID,
                int matlIndx, int levelIndx = 0);

  void init_device(std::size_t objectSizeInBytes, unsigned int maxdVarDBItems);
  void init(int id, std::string internalName);
  void cleanup();

private:
#ifdef HAVE_SYCL
  template <int DIM>
  dataItem *getItem(sycl::nd_item<DIM> &item, const char *label,
                    const int patchID, const int8_t matlIndx,
                    const int8_t levelIndx) {

    // This upcoming __syncthreads is needed.  With CUDA function calls are
    // inlined.
    //  If you don't have it this upcoming __syncthreads here's what I think can
    //  happen:

    // * The correct index was found by one of the threads.
    // * The last __syncthreads is called, all threads met up there.
    // * Some threads in the block then make a second "function" call and reset
    // index to -1
    // * Meanwhile, those other threads were still in the first "function" call
    // and hadn't
    //   yet processed if (index == -1).  They now run that line.  And see index
    //   is now -1.  That's bad.

    sycl::group work_grp = item.get_group();
    // So to prevent this scenario, we have one more __syncthreads listed
    // immediately below.
    sycl::group_barrier(work_grp); // sync before get

    short numThreads = work_grp.get_local_range().size();
    int i = work_grp.get_local_linear_id();

    // Have every thread try to find the label/patchId/matlIndx is a match in
    // array.  This is a parallel approach so that instead of doing a simple
    // sequential search with one thread, we can let every thread search for it.
    // Only the winning thread gets to write to shared data.

    sycl::local_ptr<int> index =
        sycl::ext::oneapi::group_local_memory_for_overwrite<int>(work_grp);
    *index = -1;

    sycl::group_barrier(work_grp);

    while (i < d_numVarDBItems) {
      short strmatch = 0;
      char const *s1 = label;
      char const *s2 = &(d_varDB[i].label[0]);

      // a one-line strcmp.  This should keep branching down to a minimum.
      while (!(strmatch = *(unsigned char *)s1 - *(unsigned char *)s2) &&
             *s1++ && *s2++)
        ;

      // only one thread will ever match this.
      // And nobody on the device side should ever access "staging" variables.
      if (strmatch == 0) {
        if (patchID == -99999999 && d_varDB[i].matlIndx == matlIndx &&
            d_varDB[i].levelIndx == levelIndx &&
            d_varDB[i].varItem.staging == false &&
            d_varDB[i].ghostItem.dest_varDB_index == -1) {
          *index = i;
        } else if (d_varDB[i].domainID == patchID &&
                   d_varDB[i].matlIndx == matlIndx &&
                   d_varDB[i].varItem.staging == false &&
                   d_varDB[i].ghostItem.dest_varDB_index == -1) {
          *index = i;
        }
      }
      i = i + numThreads;
    }

    sycl::group_barrier(work_grp);
    return &d_varDB[*index];
  }
#else // CUDA, HIP
  __device__ dataItem *getItem(char const *label, const int patchID,
                               const int8_t matlIndx, const int8_t levelIndx);
#endif

  std::map<labelPatchMatlLevel, allVarPointersInfo> *varPointers;

  Uintah::MasterLock *allocateLock;
  Uintah::MasterLock *varLock;

  char _internalName[80];

  materialItem d_materialDB[MAX_MATERIALSDB_ITEMS];
  dataItem d_levelDB[MAX_LEVELDB_ITEMS];
  int d_numVarDBItems;
  int d_numMaterials;
  int numGhostCellCopiesNeeded;
  GPUDataWarehouse *d_device_copy; // The pointer to the copy of this object in the GPU.
  bool d_dirty;       // if this changes, we have to recopy the GPUDW.
  int d_device_id;
  bool d_debug;
  std::size_t objectSizeInBytes;
  unsigned int d_maxdVarDBItems; // How many items we can add to d_varDB before
                                 // we run out of capacity.
  void *
      placementNewBuffer; // For task DWs, we want to seraliaze and size this
                          // object as small as possible. So we create a buffer,
                          // and keep track of the start of that buffer here.

  // These STL data structures being here do not pose a problem for the CUDA
  // compiler

  std::map<std::string, contiguousArrayInfo> *contiguousArrays;

  dataItem d_varDB
      [MAX_VARDB_ITEMS]; // For the device side.  The variable database. It's a
                         // very large buffer. Important note: We should never
                         // transfer a full GPUDataWarehouse object as is.
                         // Instead we should use malloc and only use a section
                         // of it.  See here for more information:
                         // http://www.open-std.org/Jtc1/sc22/wg14/www/docs/dr_051.html
                         // Doing it this way allows to only need one malloc
                         // instead of two.  And the object is serialized
                         // allowing for a single copy if needed instead of
                         // worrying about two copies (one for the GPU DW and
                         // one for the array). A thread first accumulates all
                         // items that will go into this array, gets a count and
                         // data size, and then only that amount of data is
                         // copied to the GPU's memory.
                         //***This must be the last data member of the class***
                         // This follows C++ 98 standards "Nonstatic data
                         // members of a (non-union) class declared without an
                         // intervening access-specifier are allocated so that
                         // later members have higher addresses within a class
                         // object. The order of allocation of nonstatic data
                         // members separated by an access-specifier is
                         // unspecified (11.1). Implementation alignment
                         // requirements might cause two adjacent members not to
                         // be allocated immediately after each other; so might
                         // requirements for space for managing virtual
                         // functions (10.3) and virtual base classes (10.1)."
};

} // end namespace Uintah
