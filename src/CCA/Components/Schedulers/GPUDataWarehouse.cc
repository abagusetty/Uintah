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

/* GPU DataWarehouse device & host access*/

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <CCA/Components/Schedulers/GPUMemoryPool.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <CCA/Components/Schedulers/SYCLScheduler.hpp>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>

//______________________________________________________________________
//
bool GPUDataWarehouse::stagingVarExists(char const *label, int patchID,
                                        int matlIndx, int levelIndx,
                                        const sycl::int3& offset, const sycl::int3& size) {
  // host code
  varLock->lock();
  bool retval = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);

  if (it != varPointers->end()) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it =
        it->second.var->stagingVars.find(sv);
    retval = (staging_it != it->second.var->stagingVars.end());
  }
  varLock->unlock();
  return retval;
}

//______________________________________________________________________
//
void GPUDataWarehouse::getStagingVar(const GPUGridVariableBase &var,
                                     char const *label, int patchID,
                                     int matlIndx, int levelIndx, sycl::int3 offset,
                                     sycl::int3 size) {
  // host code
  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it =
        it->second.var->stagingVars.find(sv);
    if (staging_it != it->second.var->stagingVars.end()) {
      var.setArray3(offset, size, staging_it->second.device_ptr);
    } else {
      printf("GPUDataWarehouse::getStagingVar() - Didn't find a staging "
             "variable from the device for label %s patch %d matl %d level %d "
             "offset (%d, %d, %d) size (%d, %d, %d).",
             label, patchID, matlIndx, levelIndx, offset.x(), offset.y(),
             offset.z(), size.x(), size.y(), size.z());
      exit(-1);
    }
  }
  varLock->unlock();
}

//______________________________________________________________________
//

void GPUDataWarehouse::getLevel(const GPUGridVariableBase &var,
                                char const *label, int8_t matlIndx,
                                int8_t levelIndx) {
  // host code
  get(var, label, -99999999, matlIndx, levelIndx);
}

//______________________________________________________________________
//

// template<typename T, typename std::enable_if_t<
//                        std::is_same_v<T, GPUGridVariableBase> ||
//                        std::is_same_v<T, GPUReductionVariableBase> ||
//                        std::is_same_v<T, GPUPerPatchBase>> >
// void GPUDataWarehouse::get(const T &var, char const *label,
//                            const int patchID, const int8_t matlIndx,
//                            const int8_t levelIndx) {
//   // host code
//   varLock->lock();
//   labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
//   auto it = varPointers->find(lpml);
//   if (it != varPointers->end()) {
//     allVarPointersInfo vp = it->second;
//     if constexpr (std::is_same_v<T, GPUGridVariableBase>) {
//       var.setArray3(vp.var->device_offset, vp.var->device_size,
//                     vp.var->device_ptr);
//     }
//     else {
//       var.setData(vp.var->device_ptr);
//     }
//   }
//   varLock->unlock();
// }


void GPUDataWarehouse::get(const GPUGridVariableBase &var, char const *label,
                           const int patchID, const int8_t matlIndx,
                           const int8_t levelIndx) {
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    allVarPointersInfo vp = it->second;
    var.setArray3(vp.var->device_offset, vp.var->device_size,
                  vp.var->device_ptr);
  }
  varLock->unlock();
}

void GPUDataWarehouse::get(const GPUReductionVariableBase &var,
                           char const *label, const int patchID,
                           const int8_t matlIndx, const int8_t levelIndx) {
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    allVarPointersInfo vp = it->second;
    var.setData(vp.var->device_ptr);
  }
  varLock->unlock();
}

void GPUDataWarehouse::get(const GPUPerPatchBase &var, char const *label,
                           const int patchID, const int8_t matlIndx,
                           const int8_t levelIndx) {
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    allVarPointersInfo vp = it->second;
    var.setData(vp.var->device_ptr);
  }
  varLock->unlock();
}

//______________________________________________________________________
//

void GPUDataWarehouse::getModifiable(GPUGridVariableBase &var,
                                     char const *label, const int patchID,
                                     const int8_t matlIndx,
                                     const int8_t levelIndx) {
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    var.setArray3(it->second.var->device_offset, it->second.var->device_size,
                  it->second.var->device_ptr);
  }
  varLock->unlock();
}

//______________________________________________________________________
//

void GPUDataWarehouse::getModifiable(GPUReductionVariableBase &var,
                                     char const *label, const int patchID,
                                     const int8_t matlIndx,
                                     const int8_t levelIndx) {
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    allVarPointersInfo vp = it->second;
    var.setData(vp.var->device_ptr);
  }
  varLock->unlock();
}

void GPUDataWarehouse::getModifiable(GPUPerPatchBase &var, char const *label,
                                     const int patchID, const int8_t matlIndx,
                                     const int8_t levelIndx) {
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    allVarPointersInfo vp = it->second;
    var.setData(vp.var->device_ptr);
  }
  varLock->unlock();
}
//______________________________________________________________________
// This method assumes the base patch in a superpatch region has already been
// allocated. This is a shallow copy.  It copies all datawarehouse metadata
// entries (except the status) from that item into this patch's item in the GPU
// DW.
void GPUDataWarehouse::copySuperPatchInfo(char const *label,
                                          int superPatchBaseID,
                                          int superPatchDestinationID,
                                          int matlIndx, int levelIndx) {

  if (superPatchBaseID == superPatchDestinationID) {
    // don't handle shallow copying itself
    return;
  }
  // Possible TODO: Add in offsets so the variable could be accessed in a
  // non-superpatch manner.

  labelPatchMatlLevel lpml_source(label, superPatchBaseID, matlIndx, levelIndx);
  labelPatchMatlLevel lpml_dest(label, superPatchDestinationID, matlIndx,
                                levelIndx);

  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator source_iter =
      varPointers->find(lpml_source);
  if (source_iter != varPointers->end()) {
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator dest_iter =
        varPointers->find(lpml_dest);
    if (dest_iter != varPointers->end()) {

      // They now share the variable.  The magic of this happens because the var
      // is a C++ shared_ptr
      // TODO: They don't share the same offset.  When offsets are added in,
      // this should be updated to manage offsets.
      dest_iter->second.var = source_iter->second.var;

    } else {
      printf("ERROR: GPUDataWarehouse::copySuperPatchInfo() - Didn't find a "
             "the destination ID at %d to copy into label %s patch %d matl %d "
             "level %d\n",
             superPatchDestinationID, label, superPatchDestinationID, matlIndx,
             levelIndx);
      varLock->unlock();
      exit(-1);
    }
  } else {
    printf(
        "ERROR: GPUDataWarehouse::copySuperPatchInfo() - Didn't find a base "
        "superPatch ID at %d to copy into label %s patch %d matl %d level %d\n",
        superPatchBaseID, label, superPatchDestinationID, matlIndx, levelIndx);
    varLock->unlock();
    exit(-1);
  }
  varLock->unlock();
}

//______________________________________________________________________
//
void GPUDataWarehouse::put(void* GPUGridVariableBase_ptr,
                           sycl::int3& GPUGridVariableBase_size,
                           sycl::int3& GPUGridVariableBase_offset,
                           std::size_t sizeOfDataType,
                           char const *label, int patchID, int matlIndx,
                           int levelIndx, bool staging, GhostType gtype,
                           int numGhostCells, void *host_ptr) {
  varLock->lock();

  sycl::int3 var_offset{0, 0, 0}; // offset
  sycl::int3 var_size{0, 0, 0};   // dimensions of GPUGridVariable
  void *var_ptr{nullptr};   // raw pointer to the memory

  var_offset = GPUGridVariableBase_offset;
  var_size = GPUGridVariableBase_size;
  var_ptr = GPUGridVariableBase_ptr;
  //var.getArray3(var_offset, var_size, var_ptr);

  // See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto iter = varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  // sanity checks
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put()  Can't use put() for a host-side "
           "GPU DW without it first existing in the internal database.\n");
    exit(-1);
  } else if (staging) {
    stagingVar sv;
    sv.device_offset = var_offset;
    sv.device_size = var_size;
    staging_it = iter->second.var->stagingVars.find(sv);
    if (staging_it == iter->second.var->stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::put()  Can't use put() for a "
             "host-side GPU DW without this staging var first existing in the "
             "internal database.\n");
      exit(-1);
    }
  }

  if (staging == false) {

    iter->second.varDB_index = -1;
    iter->second.var->device_ptr = var_ptr;
    iter->second.var->device_offset = var_offset;
    iter->second.var->device_size = var_size;
    iter->second.var->sizeOfDataType = sizeOfDataType;
    iter->second.var->gtype = gtype;
    iter->second.var->numGhostCells = numGhostCells;
    iter->second.var->host_contiguousArrayPtr = host_ptr;
    iter->second.var->atomicStatusInHostMemory = UNKNOWN;

  } else { // if (staging == true)

    staging_it->second.device_ptr = var_ptr;
    staging_it->second.host_contiguousArrayPtr = host_ptr;
    staging_it->second.varDB_index = -1;
    staging_it->second.atomicStatusInHostMemory = UNKNOWN;

    // Update the non-staging var's sizeOfDataType.  The staging var uses this
    // number. It's possible that a staging var can exist and an empty
    // placeholder non-staging var also exist, if so, then then empty
    // placeholder non-staging var won't have correct data type size. So we grab
    // it here.
    iter->second.var->sizeOfDataType = sizeOfDataType;
  }

  varLock->unlock();
}

void GPUDataWarehouse::put(GPUGridVariableBase &var, std::size_t sizeOfDataType,
                           char const *label, int patchID, int matlIndx,
                           int levelIndx, bool staging, GhostType gtype,
                           int numGhostCells, void *host_ptr) {
  varLock->lock();

  sycl::int3 var_offset{0, 0, 0}; // offset
  sycl::int3 var_size{0, 0, 0};   // dimensions of GPUGridVariable
  void *var_ptr{nullptr};   // raw pointer to the memory

  var.getArray3(var_offset, var_size, var_ptr);

  // See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter =
      varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  // sanity checks
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side "
           "GPU DW without it first existing in the internal database.\n");
    exit(-1);
  } else if (staging) {
    stagingVar sv;
    sv.device_offset = var_offset;
    sv.device_size = var_size;
    staging_it = iter->second.var->stagingVars.find(sv);
    if (staging_it == iter->second.var->stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a "
             "host-side GPU DW without this staging var first existing in the "
             "internal database.\n");
      exit(-1);
    }
  }

  if (staging == false) {

    iter->second.varDB_index = -1;
    iter->second.var->device_ptr = var_ptr;
    iter->second.var->device_offset = var_offset;
    iter->second.var->device_size = var_size;
    iter->second.var->sizeOfDataType = sizeOfDataType;
    iter->second.var->gtype = gtype;
    iter->second.var->numGhostCells = numGhostCells;
    iter->second.var->host_contiguousArrayPtr = host_ptr;
    iter->second.var->atomicStatusInHostMemory = UNKNOWN;

  } else { // if (staging == true)

    staging_it->second.device_ptr = var_ptr;
    staging_it->second.host_contiguousArrayPtr = host_ptr;
    staging_it->second.varDB_index = -1;
    staging_it->second.atomicStatusInHostMemory = UNKNOWN;

    // Update the non-staging var's sizeOfDataType.  The staging var uses this
    // number. It's possible that a staging var can exist and an empty
    // placeholder non-staging var also exist, if so, then then empty
    // placeholder non-staging var won't have correct data type size. So we grab
    // it here.
    iter->second.var->sizeOfDataType = sizeOfDataType;
  }

  varLock->unlock();
}
//______________________________________________________________________
// This method puts an empty placeholder entry into the GPUDW database and marks
// it as unallocated
void GPUDataWarehouse::putUnallocatedIfNotExists(char const *label, int patchID,
                                                 int matlIndx, int levelIndx,
                                                 bool staging, const sycl::int3& offset,
                                                 const sycl::int3& size) {

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);

  // If it's a normal non-staging variable, check if doesn't exist.  If so, add
  // an "unallocated" entry. If it's a staging variable, then still check if the
  // non-staging part exists.  A staging must exist within a non-staging
  // variable. A scenario where this can get a staging variable without a
  // non-staging variable is receiving data from neighbor nodes. For example,
  // suppose node A has patch 0, and node B has patch 1, and A's patch 0 needs
  // ghost cells from B's patch 1.  Node A will receive those ghost cells, but
  // they will be marked as belonging to patch 1.  Since A doesn't have the
  // regular non-staging var for patch 1, we make an empty placeholder for patch
  // 1 so A can have a staging var to hold the ghost cell for patch 1.

  if (it == varPointers->end()) {
    // Do not place size information.  The Data Warehouse should not declare its
    // current size until after the allocation is complete. Further, no
    // scheduler thread should attempt to determine an entry's size until the
    // allocated flag has been marked as true.
    allVarPointersInfo vp;

    vp.varDB_index = -1;
    vp.var->device_ptr = nullptr;
    vp.var->atomicStatusInHostMemory = UNKNOWN;
    vp.var->atomicStatusInGpuMemory = UNALLOCATED;
    vp.var->host_contiguousArrayPtr = nullptr;
    vp.var->sizeOfDataType = 0;

    std::pair<std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator, bool>
        ret = varPointers->insert(
            std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type(lpml,
                                                                          vp));
    if (!ret.second) {
      printf("ERROR:\nGPUDataWarehouse::putUnallocatedIfNotExists( ) Failure "
             "inserting into varPointers map.\n");
      varLock->unlock();
      exit(-1);
    }
    it = ret.first;
  }

  if (staging) {
    std::map<stagingVar, stagingVarInfo>::iterator staging_it;

    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.var->stagingVars.find(sv);
    if (staging_it == it->second.var->stagingVars.end()) {
      stagingVarInfo svi;
      svi.varDB_index = -1;
      svi.device_ptr = nullptr;
      svi.host_contiguousArrayPtr = nullptr;
      svi.atomicStatusInHostMemory = UNKNOWN;
      svi.atomicStatusInGpuMemory = UNALLOCATED;

      std::pair<stagingVar, stagingVarInfo> p = std::make_pair(sv, svi);

      it->second.var->stagingVars.insert(p);
    }
  }
  varLock->unlock();
}

//______________________________________________________________________
//
void GPUDataWarehouse::allocateAndPut(const TypeDescription::Type& type,
                                      void*& GPUGridVariableBase_ptr,
                                      sycl::int3& GPUGridVariableBase_size,
                                      sycl::int3& GPUGridVariableBase_offset,
                                      char const *label, int patchID,
                                      int matlIndx, int levelIndx, bool staging,
                                      const sycl::int3& low, const sycl::int3& high,
                                      std::size_t sizeOfDataType, GhostType gtype,
                                      int numGhostCells) {

  // Allocate space on the GPU and declare a variable onto the GPU.

  // Check if it exists prior to allocating memory for it.
  // If it has already been allocated, just use that.
  // If it hasn't, this is lock free and the first thread to request allocating
  // gets to allocate If another thread sees that allocating is in process, it
  // loops and waits until the allocation complete.

  bool allocationNeeded = false;
  sycl::int3 size = high - low;
  sycl::int3 offset = low;

  // This variable may not yet exist.  But we want to declare we're allocating
  // it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, staging,
                            offset, size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  if (staging) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.var->stagingVars.find(sv);
  }

  varLock->unlock();

  // Locking not needed from here on in this method.  STL maps ensure that
  // iterators point to correct values even if other threads add nodes.  We just
  // can't remove values, but that shouldn't ever happen.

  // This prepares the var with the offset and size.  Any possible allocation
  // will come later. If it needs to go into the database, that will also come
  // later
  void *addr{nullptr};
  GPUGridVariableBase_size = size;
  GPUGridVariableBase_offset = offset;
  GPUGridVariableBase_ptr = addr;
  //var.setArray3(offset, size, addr);

  // Now see if we allocate the variable or use a previous existing allocation.
  if (staging == false) {

    // See if someone has stated they are allocating it
    allocationNeeded =
        compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);

    if (!allocationNeeded) {
      // Someone else is allocating it or it has already been allocated. Wait
      // until they are done.
      bool allocated = false;
      while (!allocated) {
        allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
      }

      // Sanity check to ensure we have correct size information.
      varLock->lock();
      it = varPointers->find(lpml);
      varLock->unlock();

      if (it->second.var->device_offset.x() == low.x() &&
          it->second.var->device_offset.y() == low.y() &&
          it->second.var->device_offset.z() == low.z() &&
          it->second.var->device_size.x() == size.x() &&
          it->second.var->device_size.y() == size.y() &&
          it->second.var->device_size.z() == size.z()) {

        // Space for this var already exists.  Use that and return.

        // Have this var use the existing memory address.
        GPUGridVariableBase_offset = it->second.var->device_offset;
        GPUGridVariableBase_size = it->second.var->device_size;
        GPUGridVariableBase_ptr = it->second.var->device_ptr;
        // var.setArray3(it->second.var->device_offset,
        //               it->second.var->device_size, it->second.var->device_ptr);
      } else if (it->second.var->device_offset.x() <= low.x() &&
                 it->second.var->device_offset.y() <= low.y() &&
                 it->second.var->device_offset.z() <= low.z() &&
                 it->second.var->device_size.x() >= size.x() &&
                 it->second.var->device_size.y() >= size.y() &&
                 it->second.var->device_size.z() >= size.z()) {
        // It fits inside.  Just use it.

        GPUGridVariableBase_offset = it->second.var->device_offset;
        GPUGridVariableBase_size = it->second.var->device_size;
        GPUGridVariableBase_ptr = it->second.var->device_ptr;
        // var.setArray3(it->second.var->device_offset,
        //               it->second.var->device_size, it->second.var->device_ptr);
      } else {
        printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  Variable in "
               "database but of the wrong size.  This shouldn't ever happen. "
               "This needs low (%d, %d, %d) and size (%d, %d, %d), but in the "
               "database it is low (%d, %d, %d) and size (%d, %d, %d)\n",
               label, low.x(), low.y(), low.z(), size.x(), size.y(), size.z(),
               it->second.var->device_offset.x(),
               it->second.var->device_offset.y(),
               it->second.var->device_offset.z(),
               it->second.var->device_size.x(), it->second.var->device_size.y(),
               it->second.var->device_size.z());
        exit(-1);
      }
    }
  } else {

    // it's a staging variable
    if (staging_it != it->second.var->stagingVars.end()) {

      // This variable exists in the database, no need to "put" it in again.
      // See if someone has stated they are allocating it
      allocationNeeded =
          compareAndSwapAllocating(staging_it->second.atomicStatusInGpuMemory);

      if (!allocationNeeded) {
        // We need the pointer.  We can't move on until we get the pointer.
        // Ensure that it has been allocated (just not allocating). Another
        // thread may have been assigned to allocate it but not completed that
        // action.  If that's the case, wait until it's done so we can get the
        // pointer.
        bool allocated = false;
        while (!allocated) {
          allocated = checkAllocated(staging_it->second.atomicStatusInGpuMemory);
        }
        // Have this var use the existing memory address.
        GPUGridVariableBase_offset = offset;
        GPUGridVariableBase_size = size;
        GPUGridVariableBase_ptr = staging_it->second.device_ptr;
        //var.setArray3(offset, size, staging_it->second.device_ptr);
      }
    }
  }

  // Now allocate it
  if (allocationNeeded) {
    OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);

    std::size_t memSize = 0;
    switch(type) {
    case TypeDescription::double_type : {
      memSize = GPUGridVariableBase_size.x() * GPUGridVariableBase_size.y() * GPUGridVariableBase_size.z() * sizeof(double);
      break;
    }
    case TypeDescription::float_type : {
      memSize = GPUGridVariableBase_size.x() * GPUGridVariableBase_size.y() * GPUGridVariableBase_size.z() * sizeof(float);
      break;
    }
    case TypeDescription::int_type : {
      memSize = GPUGridVariableBase_size.x() * GPUGridVariableBase_size.y() * GPUGridVariableBase_size.z() * sizeof(int);
      break;
    }
    case TypeDescription::Stencil7 : {
      memSize = GPUGridVariableBase_size.x() * GPUGridVariableBase_size.y() * GPUGridVariableBase_size.z() * sizeof(GPUStencil7);
      break;
    }
    default : {
      SCI_THROW(InternalError("createGPUGridVariable, unsupported GPUGridVariable type: ", __FILE__, __LINE__));
    }
    }

    addr = GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(d_device_id, memSize);

    // Also update the var object itself (i.e., addr)
    GPUGridVariableBase_offset = offset;
    GPUGridVariableBase_size = size;
    GPUGridVariableBase_ptr = addr;
    //var.setArray3(offset, size, addr);

    // Put all remaining information about the variable into the the database.
    put(GPUGridVariableBase_ptr, GPUGridVariableBase_size, GPUGridVariableBase_offset,
        sizeOfDataType, label, patchID, matlIndx, levelIndx, staging,
        gtype, numGhostCells);

    // Now that we have the pointer and that it has been inserted into the
    // database, Update the status from allocating to allocated
    if (!staging) {
      compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
    } else {
      compareAndSwapAllocate(staging_it->second.atomicStatusInGpuMemory);
    }
  }
}

void GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var,
                                      char const *label, int patchID,
                                      int matlIndx, int levelIndx, bool staging,
                                      const sycl::int3& low, const sycl::int3& high,
                                      std::size_t sizeOfDataType, GhostType gtype,
                                      int numGhostCells) {

  // Allocate space on the GPU and declare a variable onto the GPU.

  // Check if it exists prior to allocating memory for it.
  // If it has already been allocated, just use that.
  // If it hasn't, this is lock free and the first thread to request allocating
  // gets to allocate If another thread sees that allocating is in process, it
  // loops and waits until the allocation complete.

  bool allocationNeeded = false;
  sycl::int3 size = high - low;
  sycl::int3 offset = low;

  // This variable may not yet exist.  But we want to declare we're allocating
  // it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, staging,
                            offset, size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  if (staging) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.var->stagingVars.find(sv);
  }

  varLock->unlock();

  // Locking not needed from here on in this method.  STL maps ensure that
  // iterators point to correct values even if other threads add nodes.  We just
  // can't remove values, but that shouldn't ever happen.

  // This prepares the var with the offset and size.  Any possible allocation
  // will come later. If it needs to go into the database, that will also come
  // later
  void *addr = nullptr;
  var.setArray3(offset, size, addr);

  // Now see if we allocate the variable or use a previous existing allocation.
  if (staging == false) {

    // See if someone has stated they are allocating it
    allocationNeeded =
        compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);

    if (!allocationNeeded) {
      // Someone else is allocating it or it has already been allocated. Wait
      // until they are done.
      bool allocated = false;
      while (!allocated) {
        allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
      }

      // Sanity check to ensure we have correct size information.
      varLock->lock();
      it = varPointers->find(lpml);
      varLock->unlock();

      if (it->second.var->device_offset.x() == low.x() &&
          it->second.var->device_offset.y() == low.y() &&
          it->second.var->device_offset.z() == low.z() &&
          it->second.var->device_size.x() == size.x() &&
          it->second.var->device_size.y() == size.y() &&
          it->second.var->device_size.z() == size.z()) {

        // Space for this var already exists.  Use that and return.

        // Have this var use the existing memory address.
        var.setArray3(it->second.var->device_offset,
                      it->second.var->device_size, it->second.var->device_ptr);
      } else if (it->second.var->device_offset.x() <= low.x() &&
                 it->second.var->device_offset.y() <= low.y() &&
                 it->second.var->device_offset.z() <= low.z() &&
                 it->second.var->device_size.x() >= size.x() &&
                 it->second.var->device_size.y() >= size.y() &&
                 it->second.var->device_size.z() >= size.z()) {
        // It fits inside.  Just use it.

        var.setArray3(it->second.var->device_offset,
                      it->second.var->device_size, it->second.var->device_ptr);
      } else {
        printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  Variable in "
               "database but of the wrong size.  This shouldn't ever happen. "
               "This needs low (%d, %d, %d) and size (%d, %d, %d), but in the "
               "database it is low (%d, %d, %d) and size (%d, %d, %d)\n",
               label, low.x(), low.y(), low.z(), size.x(), size.y(), size.z(),
               it->second.var->device_offset.x(),
               it->second.var->device_offset.y(),
               it->second.var->device_offset.z(),
               it->second.var->device_size.x(), it->second.var->device_size.y(),
               it->second.var->device_size.z());
        exit(-1);
      }
    }
  } else {

    // it's a staging variable
    if (staging_it != it->second.var->stagingVars.end()) {

      // This variable exists in the database, no need to "put" it in again.
      // See if someone has stated they are allocating it
      allocationNeeded =
          compareAndSwapAllocating(staging_it->second.atomicStatusInGpuMemory);

      if (!allocationNeeded) {
        // We need the pointer.  We can't move on until we get the pointer.
        // Ensure that it has been allocated (just not allocating). Another
        // thread may have been assigned to allocate it but not completed that
        // action.  If that's the case, wait until it's done so we can get the
        // pointer.
        bool allocated = false;
        while (!allocated) {
          allocated =
              checkAllocated(staging_it->second.atomicStatusInGpuMemory);
        }
        // Have this var use the existing memory address.
        var.setArray3(offset, size, staging_it->second.device_ptr);
      }
    }
  }

  // Now allocate it
  if (allocationNeeded) {
    OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);
    addr = GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(d_device_id, var.getMemSize());

    // Also update the var object itself (i.e., addr)
    var.setArray3(offset, size, addr);

    // Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging,
        gtype, numGhostCells);

    // Now that we have the pointer and that it has been inserted into the
    // database, Update the status from allocating to allocated
    if (!staging) {
      compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
    } else {
      compareAndSwapAllocate(staging_it->second.atomicStatusInGpuMemory);
    }
  }
}

//______________________________________________________________________
// This method is meant to take an entry from the host side DW and copy it  into
// the task datawarehouse whose job is to  eventually live GPU side.
void GPUDataWarehouse::copyItemIntoTaskDW(GPUDataWarehouse *hostSideGPUDW,
                                          char const *label, int patchID,
                                          int matlIndx, int levelIndx,
                                          bool staging, sycl::int3 offset,
                                          sycl::int3 size) {

  if (d_device_copy == nullptr) {
    printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - This method "
           "should only be called from a task data warehouse.\n");
    exit(-1);
  }

  varLock->lock();
  if (d_numVarDBItems == MAX_VARDB_ITEMS) {
    printf("ERROR: Out of GPUDataWarehouse space");
    varLock->unlock();
    exit(-1);
  }
  varLock->unlock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  stagingVar sv;
  sv.device_offset = offset;
  sv.device_size = size;

  // Get the iterator(s) from the host side GPUDW.
  hostSideGPUDW->varLock->lock();

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator
      hostSideGPUDW_iter = hostSideGPUDW->varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator hostSideGPUDW_staging_iter;
  if (staging) {
    hostSideGPUDW_staging_iter = hostSideGPUDW_iter->second.var->stagingVars.find(sv);
    if (hostSideGPUDW_staging_iter == hostSideGPUDW_iter->second.var->stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - No staging var "
             "was found for for %s patch %d material %d level %d offset (%d, "
             "%d, %d) size (%d, %d, %d) in the DW located at %p\n",
             label, patchID, matlIndx, levelIndx, offset.x(), offset.y(),
             offset.z(), size.x(), size.y(), size.z(), hostSideGPUDW);
      varLock->unlock();
      exit(-1);
    }
  }

  hostSideGPUDW->varLock->unlock();

  varLock->lock();

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);
  // sanity check
  if (iter != varPointers->end() && !staging) {
    printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - This task "
           "datawarehouse already had an entry for %s patch %d material %d "
           "level %d\n",
           label, patchID, matlIndx, levelIndx);
    varLock->unlock();
    exit(-1);
  }

  // If it's staging, there should already be a non-staging var in the host-side
  // GPUDW (even if it's just a placeholder)

  // Inserting into this task DW, it is a requirement that non-staging variables
  // get inserted first then any staging variables can come in later.  This
  // won't handle any scenario where a staging variable is requested into the
  // task DW without a non-staging variable already existing here.

  // TODO: Replace with an atomic counter.
  int d_varDB_index = d_numVarDBItems;
  d_numVarDBItems++;

  int i = d_varDB_index;

  if (!staging) {

    // Create a new allVarPointersInfo object, copying over the offset.
    allVarPointersInfo vp;
    vp.device_offset = hostSideGPUDW_iter->second.device_offset;

    // Give it a d_varDB index
    vp.varDB_index = d_varDB_index;

    // insert it in
    varPointers->insert(
        std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type(lpml,
                                                                      vp));

    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);

    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx = levelIndx;
    d_varDB[i].sizeOfDataType = hostSideGPUDW_iter->second.var->sizeOfDataType;
    d_varDB[i].varItem.gtype = hostSideGPUDW_iter->second.var->gtype;
    d_varDB[i].varItem.numGhostCells =
        hostSideGPUDW_iter->second.var->numGhostCells;
    d_varDB[i].varItem.staging = staging;
    d_varDB[i].ghostItem.dest_varDB_index =
        -1; // Signify that this d_varDB item is NOT meta data to copy a ghost
            // cell.
    d_varDB[i].var_offset = hostSideGPUDW_iter->second.var->device_offset;
    d_varDB[i].var_size = hostSideGPUDW_iter->second.var->device_size;
    d_varDB[i].var_ptr = hostSideGPUDW_iter->second.var->device_ptr;

  } else {

    if (iter == varPointers->end()) {
      // A staging item was requested but there's no regular variable for it to
      // piggy back in. So create an empty placeholder regular variable.

      // Create a new allVarPointersInfo object, copying over the offset.
      allVarPointersInfo vp;
      vp.device_offset = hostSideGPUDW_iter->second.device_offset;

      // Empty placeholders won't be placed in the d_varDB array.
      vp.varDB_index = -1;

      // insert it in
      std::pair<std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator,
                bool>
          ret = varPointers->insert(
              std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type(
                  lpml, vp));
      if (!ret.second) {
        printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW( ) Failure "
               "inserting into varPointers map.\n");
        varLock->unlock();
        exit(-1);
      }
      iter = ret.first;
    }

    // copy the item
    stagingVarInfo svi = hostSideGPUDW_staging_iter->second;

    // Give it a d_varDB index
    svi.varDB_index = d_varDB_index;

    // insert it in
    std::map<stagingVar, stagingVarInfo>::iterator staging_iter =
        iter->second.var->stagingVars.find(sv);
    if (staging_iter != iter->second.var->stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW( ) This staging var "
             "already exists in this task DW\n");
    }
    std::pair<stagingVar, stagingVarInfo> p = std::make_pair(sv, svi);
    iter->second.var->stagingVars.insert(p);

    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx = levelIndx;
    d_varDB[i].sizeOfDataType = hostSideGPUDW_iter->second.var->sizeOfDataType;
    d_varDB[i].varItem.gtype = hostSideGPUDW_iter->second.var->gtype;
    d_varDB[i].varItem.numGhostCells =
        hostSideGPUDW_iter->second.var->numGhostCells;
    d_varDB[i].varItem.staging = staging;
    d_varDB[i].ghostItem.dest_varDB_index =
        -1; // Signify that this d_varDB item is NOT meta data to copy a ghost
            // cell.
    d_varDB[i].var_offset = hostSideGPUDW_staging_iter->first.device_offset;
    d_varDB[i].var_size = hostSideGPUDW_staging_iter->first.device_size;
    d_varDB[i].var_ptr = hostSideGPUDW_staging_iter->second.device_ptr;
  }

  d_dirty = true;
  varLock->unlock();
}

//______________________________________________________________________
//
void GPUDataWarehouse::putContiguous(
    GPUGridVariableBase &var, const char *indexID, char const *label,
    int patchID, int matlIndx, int levelIndx, bool staging, sycl::int3 low, sycl::int3 high,
    std::size_t sizeOfDataType, GridVariableBase *gridVar, bool stageOnHost) {
  // abb: NO-OP, implementation removed since not being used
}

//______________________________________________________________________
//
void GPUDataWarehouse::allocate(const char *indexID, std::size_t size) {
  // abb: NO-OP, implementation removed since not being used
}

//______________________________________________________________________
//
void GPUDataWarehouse::copyHostContiguousToHost(GPUGridVariableBase &device_var,
                                                GridVariableBase *host_var,
                                                char const *label, int patchID,
                                                int matlIndx, int levelIndx) {}

//______________________________________________________________________
//
void GPUDataWarehouse::put(GPUReductionVariableBase &var, std::size_t sizeOfDataType,
                           char const *label, int patchID, int matlIndx,
                           int levelIndx, void *host_ptr) {

  varLock->lock();

  void *var_ptr; // raw pointer to the memory
  var.getData(var_ptr);

  // See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter =
      varPointers->find(lpml);

  // sanity check
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side "
           "GPU DW without it first existing in the internal database.\n");
    exit(-1);
  }

  iter->second.varDB_index = -1;
  iter->second.var->device_ptr = var_ptr;
  iter->second.var->sizeOfDataType = sizeOfDataType;
  iter->second.var->gtype = None;
  iter->second.var->numGhostCells = 0;
  iter->second.var->host_contiguousArrayPtr = host_ptr;
  iter->second.var->atomicStatusInHostMemory = UNKNOWN;
  sycl::int3 zeroValue{0};
  iter->second.var->device_offset = zeroValue;
  iter->second.var->device_size = zeroValue;

  // previously set, do not set here
  // iter->second.var->atomicStatusInGpuMemory =

  varLock->unlock();
}
//______________________________________________________________________
//
// SYCL varient: same for GPUReductionVariableBase & GPUPerPatchBase
void GPUDataWarehouse::put(void* gpu_ptr, std::size_t sizeOfDataType,
                           char const *label, int patchID, int matlIndx,
                           int levelIndx, void *host_ptr) {

  varLock->lock();
  void *var_ptr{nullptr}; // raw pointer to the memory
  var_ptr = gpu_ptr;
  //var.getData(var_ptr);

  // See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter =
      varPointers->find(lpml);

  // sanity check
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side "
           "GPU DW without it first existing in the internal database for %s "
           "patch %d matl %d.\n",
           label, patchID, matlIndx);
    exit(-1);
  }

  iter->second.varDB_index = -1;
  iter->second.var->device_ptr = var_ptr;
  iter->second.var->sizeOfDataType = sizeOfDataType;
  iter->second.var->gtype = None;
  iter->second.var->numGhostCells = 0;
  iter->second.var->host_contiguousArrayPtr = host_ptr;
  iter->second.var->atomicStatusInHostMemory = UNKNOWN;
  sycl::int3 zeroValue{0};
  iter->second.var->device_offset = zeroValue;
  iter->second.var->device_size = zeroValue;

  // previously set, do not set here
  // iter->second.atomicStatusInGputMemory =

  varLock->unlock();
}

void GPUDataWarehouse::put(GPUPerPatchBase &var, std::size_t sizeOfDataType,
                           char const *label, int patchID, int matlIndx,
                           int levelIndx, void *host_ptr) {

  varLock->lock();
  void *var_ptr; // raw pointer to the memory
  var.getData(var_ptr);

  // See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter =
      varPointers->find(lpml);

  // sanity check
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side "
           "GPU DW without it first existing in the internal database for %s "
           "patch %d matl %d.\n",
           label, patchID, matlIndx);
    exit(-1);
  }

  iter->second.varDB_index = -1;
  iter->second.var->device_ptr = var_ptr;
  iter->second.var->sizeOfDataType = sizeOfDataType;
  iter->second.var->gtype = None;
  iter->second.var->numGhostCells = 0;
  iter->second.var->host_contiguousArrayPtr = host_ptr;
  iter->second.var->atomicStatusInHostMemory = UNKNOWN;
  sycl::int3 zeroValue{0};
  iter->second.var->device_offset = zeroValue;
  iter->second.var->device_size = zeroValue;

  // previously set, do not set here
  // iter->second.atomicStatusInGputMemory =

  varLock->unlock();
}

//______________________________________________________________________
//
// same for GPUReductionVariableBase, GPUPerPatchBase
void GPUDataWarehouse::allocateAndPut(const TypeDescription::Type& type,
                                      void*& GPUReductionPerPatchVariableBase,
                                      char const *label, int patchID,
                                      int matlIndx, int levelIndx,
                                      std::size_t sizeOfDataType) {

  // Allocate space on the GPU and declare a variable onto the GPU.
  // This method does NOT stage everything in a big array.

  // Check if it exists prior to allocating memory for it.
  // If it has already been allocated, just use that.
  // If it hasn't, this is lock free and the first thread to request allocating
  // gets to allocate If another thread sees that allocating is in process, it
  // loops and waits until the allocation complete.

  bool allocationNeeded = false;
  sycl::int3 size{0, 0, 0};
  sycl::int3 offset{0, 0, 0};
  // This variable may not yet exist.  But we want to declare we're allocating
  // it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset, size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  varLock->unlock();

  void *addr = nullptr;

  // Now see if we allocate the variable or use a previous existing allocation.
  // See if someone has stated they are allocating it
  allocationNeeded = compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);
  if (!allocationNeeded) {
    // Someone else is allocating it or it has already been allocated.
    // Space for this var already exists.  Use that and return.

    // We need the pointer.  We can't move on until we get the pointer.
    // Ensure that it has been allocated (just not allocating). Another thread
    // may have been assigned to allocate it but not completed that action.  If
    // that's the case, wait until it's done so we can get the pointer.
    bool allocated = false;
    while (!allocated) {
      allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
      addr = it->second.var->device_ptr;
    }
    // Have this var use the existing memory address.
    GPUReductionPerPatchVariableBase = addr;
    //var.setData(addr);
  } else {
    // We are the first task to request allocation.  Do it.
    OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);

    std::size_t memSize = 0;
    switch(type) {
    case TypeDescription::double_type : {
      memSize = sizeof(double);
      break;
    }
    case TypeDescription::float_type : {
      memSize = sizeof(float);
      break;
    }
    case TypeDescription::int_type : {
      memSize = sizeof(int);
      break;
    }
    case TypeDescription::Stencil7 : {
      memSize = sizeof(GPUStencil7);
      break;
    }
    default : {
      SCI_THROW(InternalError("createGPUGridVariable, unsupported GPUGridVariable type: ", __FILE__, __LINE__));
    }
    }

    addr = GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(d_device_id, memSize);

    // Also update the var object itself
    GPUReductionPerPatchVariableBase = addr;
    //var.setData(addr);

    // Put all remaining information about the variable into the the database.
    put(GPUReductionPerPatchVariableBase, sizeOfDataType, label, patchID, matlIndx, levelIndx);

    // Now that the database knows of this and other threads can see the device
    // pointer, update the status from allocating to allocated
    compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
  }
}


void GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase &var,
                                      char const *label, int patchID,
                                      int matlIndx, int levelIndx,
                                      std::size_t sizeOfDataType) {

  // Allocate space on the GPU and declare a variable onto the GPU.
  // This method does NOT stage everything in a big array.

  // Check if it exists prior to allocating memory for it.
  // If it has already been allocated, just use that.
  // If it hasn't, this is lock free and the first thread to request allocating
  // gets to allocate If another thread sees that allocating is in process, it
  // loops and waits until the allocation complete.

  bool allocationNeeded = false;
  sycl::int3 size = sycl::int3(0, 0, 0);
  sycl::int3 offset = sycl::int3(0, 0, 0);
  // This variable may not yet exist.  But we want to declare we're allocating
  // it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset,
                            size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);

  varLock->unlock();

  void *addr = nullptr;

  // Now see if we allocate the variable or use a previous existing allocation.
  // See if someone has stated they are allocating it
  allocationNeeded = compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);
  if (!allocationNeeded) {
    // Someone else is allocating it or it has already been allocated.
    // Space for this var already exists.  Use that and return.

    // We need the pointer.  We can't move on until we get the pointer.
    // Ensure that it has been allocated (just not allocating). Another thread
    // may have been assigned to allocate it but not completed that action.  If
    // that's the case, wait until it's done so we can get the pointer.
    bool allocated = false;
    while (!allocated) {
      allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
      addr = it->second.var->device_ptr;
    }
    // Have this var use the existing memory address.
    var.setData(addr);
  } else {
    // We are the first task to request allocation.  Do it.
    OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);
    std::size_t memSize = var.getMemSize();
    addr = GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(d_device_id, memSize);

    // Also update the var object itself
    var.setData(addr);

    // Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);

    // Now that the database knows of this and other threads can see the device
    // pointer, update the status from allocating to allocated
    compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
  }
}

//______________________________________________________________________
//
void GPUDataWarehouse::allocateAndPut(GPUPerPatchBase &var, char const *label,
                                      int patchID, int matlIndx, int levelIndx,
                                      std::size_t sizeOfDataType) {

  // Allocate space on the GPU and declare a variable onto the GPU.
  // This method does NOT stage everything in a big array.

  // Check if it exists prior to allocating memory for it.
  // If it has already been allocated, just use that.
  // If it hasn't, this is lock free and the first thread to request allocating
  // gets to allocate If another thread sees that allocating is in process, it
  // loops and waits until the allocation complete.

  bool allocationNeeded = false;
  sycl::int3 size = sycl::int3(0, 0, 0);
  sycl::int3 offset = sycl::int3(0, 0, 0);
  // This variable may not yet exist.  But we want to declare we're allocating
  // it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset,
                            size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);

  varLock->unlock();

  void *addr = nullptr;

  // Now see if we allocate the variable or use a previous existing allocation.
  // See if someone has stated they are allocating it
  allocationNeeded = compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);
  if (!allocationNeeded) {
    // Someone else is allocating it or it has already been allocated.
    // Space for this var already exists.  Use that and return.

    // We need the pointer.  We can't move on until we get the pointer.
    // Ensure that it has been allocated (just not allocating). Another thread
    // may have been assigned to allocate it but not completed that action.  If
    // that's the case, wait until it's done so we can get the pointer.
    bool allocated = false;
    while (!allocated) {
      allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
      addr = it->second.var->device_ptr;
    }
    // Have this var use the existing memory address.
    var.setData(addr);
  } else {
    // We are the first task to request allocation.  Do it.
    OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);
    std::size_t memSize = var.getMemSize();
    addr = GPUMemoryPool::getInstance().allocateGpuSpaceFromPool(d_device_id, memSize);

    // Also update the var object itself
    var.setData(addr);

    // Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);
    // Now that the database knows of this and other threads can see the device
    // pointer, update the status from allocating to allocated
    compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
  }
}

//______________________________________________________________________
//

bool GPUDataWarehouse::remove(char const *label, int patchID, int matlIndx,
                              int levelIndx) {
  return false;
}

//______________________________________________________________________
//
void GPUDataWarehouse::init(int id, std::string internalName) {

  d_device_id = id;
  strncpy(_internalName, internalName.c_str(), sizeof(_internalName));
  objectSizeInBytes = 0;
  d_maxdVarDBItems = 0;

  allocateLock = new Uintah::MasterLock{};
  varLock = new Uintah::MasterLock{};
  varPointers = new std::map<labelPatchMatlLevel, allVarPointersInfo>;
  contiguousArrays = new std::map<std::string, contiguousArrayInfo>;

  // other data members are initialized in the constructor
  d_numVarDBItems = 0;
  d_numMaterials = 0;
  d_debug = false;
  // d_numGhostCells = 0;
  d_device_copy = nullptr;
  d_dirty = true;
  objectSizeInBytes = 0;
  // resetdVarDB();
  numGhostCellCopiesNeeded = 0;
}

//______________________________________________________________________
//
void GPUDataWarehouse::cleanup() {
  delete allocateLock;
  delete varLock;
  delete varPointers;
  delete contiguousArrays;
}

//______________________________________________________________________
//
void GPUDataWarehouse::init_device(std::size_t objectSizeInBytes,
                                   unsigned int d_maxdVarDBItems) {
  this->objectSizeInBytes = objectSizeInBytes;
  this->d_maxdVarDBItems = d_maxdVarDBItems;

  auto& streamPool = GPUStreamPool<>::getInstance();
  //this->d_device_copy = sycl::malloc_device<GPUDataWarehouse>(1, *(streamPool.getDefaultGpuStreamFromPool(d_device_id)));
  void* temp = sycl::malloc_device(objectSizeInBytes, *(streamPool.getDefaultGpuStreamFromPool(d_device_id)));
  this->d_device_copy = (GPUDataWarehouse*)temp;

  d_dirty = true;
}

//______________________________________________________________________
//
void GPUDataWarehouse::syncto_device(gpuStream_t* gpu_stream) {
  if (!d_device_copy) {
    printf("ERROR:\nGPUDataWarehouse::syncto_device()\nNo device copy\n");
    exit(-1);
  }
  varLock->lock();

  if (d_dirty) {
    OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);
    // Even though this is in a writeLock state on the CPU, the nature of
    // multiple threads each with their own stream copying to a GPU means that
    // one stream might seemingly go out of order.  This is ok for two reasons.
    // 1) Nothing should ever be *removed* from a gpu data warehouse 2)
    // Therefore, it doesn't matter if streams go out of order, each thread will
    // still ensure it copies exactly what it needs.  Other streams may write
    // additional data to the gpu data warehouse, but cpu threads will only
    // access their own data, not data copied in by other cpu threada via
    // streams.

    // This approach does NOT require CUDA pinned memory.
    // unsigned int sizeToCopy = sizeof(GPUDataWarehouse);

    gpu_stream->memcpy(d_device_copy, this, objectSizeInBytes);

    d_dirty = false;
  }

  varLock->unlock();
}

//______________________________________________________________________
//
void GPUDataWarehouse::clear() {

  OnDemandDataWarehouse::uintahSetGpuDevice(d_device_id);

  varLock->lock();

  auto& memPool = GPUMemoryPool::getInstance();
  for (auto varIter : varPointers) {
    // clear out all the staging vars, if any
    std::map<stagingVar, stagingVarInfo>::iterator stagingIter;
    for (auto stagingIter : varIter->second.var->stagingVars) {
      if (compareAndSwapDeallocating(
              stagingIter->second.atomicStatusInGpuMemory)) {
        // The counter hit zero, so lets deallocate the var.

        if (memPool.freeGpuSpaceToPool(d_device_id, stagingIter->second.device_ptr, stagingIter->second.sizeInBytesDevicePtr)) {
          stagingIter->second.device_ptr = nullptr;
          stagingIter->second.device_ptr = nullptr;
          compareAndSwapDeallocate(stagingIter->second.atomicStatusInGpuMemory);
        } else {
          printf("ERROR:\nGPUDataWarehouse::clear(), for a staging variable, "
                 "couldn't find in the GPU memory pool the space starting at "
                 "address %p\n",
                 stagingIter->second.device_ptr);
          varLock->unlock();
          exit(-1);
        }
      }
    }

    varIter->second.var->stagingVars.clear();

    // clear out the regular vars

    // See if it's a placeholder var for staging vars.  This happens if the
    // non-staging var had a device_ptr of nullptr, and it was only in the
    // varPointers map to only hold staging vars
    if (compareAndSwapDeallocating(
            varIter->second.var->atomicStatusInGpuMemory)) {
      if (varIter->second.var->device_ptr) {

        if (memPool.freeGpuSpaceToPool(
                d_device_id, varIter->second.var->device_ptr)) {
          varIter->second.var->device_ptr = nullptr;
          compareAndSwapDeallocate(
              varIter->second.var->atomicStatusInGpuMemory);
        } else {
          printf("ERROR:\nGPUDataWarehouse::clear(), for a non-staging "
                 "variable, couldn't find in the GPU memory pool the space "
                 "starting at address %p\n",
                 varIter->second.var->device_ptr);
          varLock->unlock();
          exit(-1);
        }
      }
    }
  }
  varPointers->clear();

  varLock->unlock();

  init(d_device_id, _internalName);
}

//______________________________________________________________________
//
void GPUDataWarehouse::deleteSelfOnDevice() {
  // Note: in SYCL no need to set the GPU active device before calling any SYCL APIs, unlike CUDA/HIP
  if (d_device_copy) {
    auto& streamPool = GPUStreamPool<>::getInstance();
    sycl::free(d_device_copy, *(streamPool.getDefaultGpuStreamFromPool(d_device_id)));
    d_device_copy = nullptr;
  }
}

//______________________________________________________________________
// These material methods below needs more work.  They haven't been tested.
void GPUDataWarehouse::putMaterials(std::vector<std::string> materials) {

  varLock->lock();
  // see if a thread has already supplied this datawarehouse with the material
  // data
  int numMaterials = materials.size();

  if (d_numMaterials != numMaterials) {
    // nobody has given us this material data yet, so lets add it in from the
    // beginning.

    if (numMaterials > MAX_MATERIALSDB_ITEMS) {
      printf("ERROR: out of GPUDataWarehouse space for materials");
      exit(-1);
    }
    for (int i = 0; i < numMaterials; i++) {
      if (strcmp(materials.at(i).c_str(), "ideal_gas") == 0) {
        d_materialDB[i].material = IDEAL_GAS;
      } else {
        printf(
            "ERROR:  This material has not yet been coded for GPU support\n.");
        exit(-1);
      }
    }
    d_numMaterials = numMaterials;
  }

  varLock->unlock();
}

//______________________________________________________________________
//
int GPUDataWarehouse::getNumMaterials() const { return d_numMaterials; }

//______________________________________________________________________
//
materialType GPUDataWarehouse::getMaterial(int i) const {
  if (i >= d_numMaterials) {
    printf("ERROR: Attempting to access material past bounds\n");
    assert(0);
  }
  return d_materialDB[i].material;
}

//______________________________________________________________________
// TODO: This is too slow.  It needs work.
void GPUDataWarehouse::copyGpuGhostCellsToGpuVars(sycl::nd_item<3> &item) {

  // Copy all ghost cells from their source to their destination.
  // The ghost cells could either be only the data that needs to be copied,
  // or it could be on an edge of a bigger grid var.
  // I believe the x,y,z coordinates of everything should match.

  // This could probably be made more efficient by using only perhaps one block,
  // copying float 4s, and doing it with instruction level parallelism.
  int numThreads = item.get_local_range(2) * item.get_local_range(1) *
                   item.get_local_range(0);
  int blockID = item.get_group(2) +
                item.get_group(1) * item.get_group_range(2) +
                item.get_group_range(2) * item.get_group_range(1) *
                    item.get_group(0); // blockID on the grid
  int threadID =
      item.get_local_id(2) +
      item.get_local_range().get(2) * item.get_local_id(1) +
      (item.get_local_range().get(2) * item.get_local_range().get(1)) *
          item.get_local_id(0); // threadID in the block
  int totalThreads = numThreads * item.get_group_range(2) *
                     item.get_group_range(1) * item.get_group_range(0);
  int assignedCellID;

  // go through every ghost cell var we need
  for (int i = 0; i < d_numVarDBItems; i++) {
    // some things in d_varDB are meta data for simulation variables
    // other things in d_varDB are meta data for how to copy ghost cells.
    // Make sure we're only dealing with ghost cells here
    if (d_varDB[i].ghostItem.dest_varDB_index != -1) {
      assignedCellID = blockID * numThreads + threadID;
      int destIndex = d_varDB[i].ghostItem.dest_varDB_index;

      sycl::int3 ghostCellSize = d_varDB[i].ghostItem.sharedHighCoordinates -
                           d_varDB[i].ghostItem.sharedLowCoordinates;

      // while there's still work to do (this assigned ID is still within the
      // ghost cell)
      while (assignedCellID <
             ghostCellSize.x() * ghostCellSize.y() * ghostCellSize.z()) {
        int z = assignedCellID / (ghostCellSize.x() * ghostCellSize.y());
        int temp = assignedCellID % (ghostCellSize.x() * ghostCellSize.y());
        int y = temp / ghostCellSize.x();
        int x = temp % ghostCellSize.x();

        assignedCellID += totalThreads;

        // if we're in a valid x,y,z space for the variable.  (It's unlikely
        // every cell will perfectly map onto every available thread.)
        if (x < ghostCellSize.x() && y < ghostCellSize.y() &&
            z < ghostCellSize.z()) {

          // offset them to their true array coordinates, not relative
          // simulation cell coordinates When using virtual addresses, the
          // virtual offset is always applied to the source, but the destination
          // is correct.
          int x_source_real = x +
                              d_varDB[i].ghostItem.sharedLowCoordinates.x() -
                              d_varDB[i].ghostItem.virtualOffset.x() -
                              d_varDB[i].var_offset.x();
          int y_source_real = y +
                              d_varDB[i].ghostItem.sharedLowCoordinates.y() -
                              d_varDB[i].ghostItem.virtualOffset.y() -
                              d_varDB[i].var_offset.y();
          int z_source_real = z +
                              d_varDB[i].ghostItem.sharedLowCoordinates.z() -
                              d_varDB[i].ghostItem.virtualOffset.z() -
                              d_varDB[i].var_offset.z();
          // count over array slots.
          int sourceOffset =
              x_source_real +
              d_varDB[i].var_size.x() *
                  (y_source_real + z_source_real * d_varDB[i].var_size.y());

          int x_dest_real = x + d_varDB[i].ghostItem.sharedLowCoordinates.x() -
                            d_varDB[destIndex].var_offset.x();
          int y_dest_real = y + d_varDB[i].ghostItem.sharedLowCoordinates.y() -
                            d_varDB[destIndex].var_offset.y();
          int z_dest_real = z + d_varDB[i].ghostItem.sharedLowCoordinates.z() -
                            d_varDB[destIndex].var_offset.z();

          int destOffset =
              x_dest_real +
              d_varDB[destIndex].var_size.x() *
                  (y_dest_real + z_dest_real * d_varDB[destIndex].var_size.y());

          // copy all 8 bytes of a double in one shot
          if (d_varDB[i].sizeOfDataType == sizeof(double)) {
            *((double *)(d_varDB[destIndex].var_ptr) + destOffset) =
                *((double *)(d_varDB[i].var_ptr) + sourceOffset);
          }
          // or copy all 4 bytes of an int in one shot.
          else if (d_varDB[i].sizeOfDataType == sizeof(int)) {
            *(((int *)d_varDB[destIndex].var_ptr) + destOffset) =
                *((int *)(d_varDB[i].var_ptr) + sourceOffset);
            // Copy each byte until we've copied all for this data type.
          } else {
            for (int j = 0; j < d_varDB[i].sizeOfDataType; j++) {
              *(((char *)d_varDB[destIndex].var_ptr) +
                (destOffset * d_varDB[destIndex].sizeOfDataType + j)) =
                  *(((char *)d_varDB[i].var_ptr) +
                    (sourceOffset * d_varDB[i].sizeOfDataType + j));
            }
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//
void copyGpuGhostCellsToGpuVarsKernel(GPUDataWarehouse *gpudw,
                                      sycl::nd_item<3> &item) {
  gpudw->copyGpuGhostCellsToGpuVars(item);
}

//______________________________________________________________________
//
void GPUDataWarehouse::copyGpuGhostCellsToGpuVarsInvoker(gpuStream_t* stream) {
  // see if this GPU datawarehouse has ghost cells in it.
  if (numGhostCellCopiesNeeded > 0) {
    // call a kernel which gets the copy process started.

    sycl::range<3> dimBlock(1, 16, 32);
    sycl::range<3> dimGrid(1, 1, 1);
    // Give each ghost copying kernel 32 * 16 = 512 threads to copy
    // (32x32 was too much for a smaller laptop GPU and the Uintah build server
    // in debug mode)

    stream->parallel_for(
      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
      [=, d_device_copy = this->d_device_copy](sycl::nd_item<3> item) {
	copyGpuGhostCellsToGpuVarsKernel(d_device_copy, item);
      });
  }
}

//______________________________________________________________________
//
bool GPUDataWarehouse::ghostCellCopiesNeeded() {
  // see if this GPU datawarehouse has ghost cells in it.
  return (numGhostCellCopiesNeeded > 0);
}

//______________________________________________________________________
//
void GPUDataWarehouse::putGhostCell(char const *label, int sourcePatchID,
                                    int destPatchID, int matlIndx,
                                    int levelIndx, bool sourceStaging,
                                    bool destStaging,
                                    const sycl::int3& varOffset,
                                    const sycl::int3& varSize,
                                    const sycl::int3& sharedLowCoordinates,
                                    const sycl::int3& sharedHighCoordinates,
                                    const sycl::int3& virtualOffset) {

  // Add information describing a ghost cell that needs to be copied internally
  // from one chunk of data to the destination.  This covers a GPU -> same GPU
  // copy scenario.
  varLock->lock();
  unsigned int i = d_numVarDBItems;
  if (i > d_maxdVarDBItems) {
    printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). Exceeded maximum "
           "d_varDB entries.  Index is %d and max items is %d\n",
           label, i, d_maxdVarDBItems);
    varLock->unlock();
    exit(-1);
  }
  int index = -1;
  d_numVarDBItems++;
  numGhostCellCopiesNeeded++;
  d_varDB[i].ghostItem.sharedLowCoordinates = sharedLowCoordinates;
  d_varDB[i].ghostItem.sharedHighCoordinates = sharedHighCoordinates;
  d_varDB[i].ghostItem.virtualOffset = virtualOffset;

  // look up the source index and the destination index for these.
  // it may be an entire variable (in which case staging is false)
  // or it may be a staging variable.
  labelPatchMatlLevel lpml_source(label, sourcePatchID, matlIndx, levelIndx);
  if (!sourceStaging) {
    if (varPointers->find(lpml_source) != varPointers->end()) {
      index = varPointers->at(lpml_source).varDB_index;
    }
  } else {
    // Find the variable that contains the region in which our ghost cells
    // exist. Usually the sharedLowCoordinates and sharedHighCoordinates
    // correspond exactly to the size of the staging variable. (TODO ? But
    // sometimes the ghost data is found within larger staging variable. Not
    // sure if there is a use case for this yet)
    stagingVar sv;
    sv.device_offset = varOffset;
    sv.device_size = varSize;

    std::map<stagingVar, stagingVarInfo>::iterator staging_it =
        varPointers->at(lpml_source).var->stagingVars.find(sv);
    if (staging_it != varPointers->at(lpml_source).var->stagingVars.end()) {

      index = staging_it->second.varDB_index;

    } else {
      int nStageVars = varPointers->at(lpml_source).var->stagingVars.size();
      printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). Number of staging "
             "vars for this var: %d, No staging variable found exactly "
             "matching all of the following: label %s patch %d matl %d level "
             "%d offset (%d, %d, %d) size (%d, %d, %d) on DW at %p.\n",
             label, nStageVars, label, sourcePatchID, matlIndx, levelIndx,
             sv.device_offset.x(), sv.device_offset.y(), sv.device_offset.z(),
             sv.device_size.x(), sv.device_size.y(), sv.device_size.z(), this);
      varLock->unlock();
      exit(-1);
    }
    // Find the d_varDB entry for this specific one.
  }

  if (index < 0) {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell, label %s, source patch ID "
           "%d, matlIndx %d, levelIndex %d staging %s not found in GPU DW %p\n",
           label, sourcePatchID, matlIndx, levelIndx,
           sourceStaging ? "true" : "false", this);
    varLock->unlock();
    exit(-1);
  }

  d_varDB[i].var_offset = d_varDB[index].var_offset;
  d_varDB[i].var_size = d_varDB[index].var_size;
  d_varDB[i].var_ptr = d_varDB[index].var_ptr;
  d_varDB[i].sizeOfDataType = d_varDB[index].sizeOfDataType;

  // Find where we are sending the ghost cell data to
  labelPatchMatlLevel lpml_dest(label, destPatchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml_dest);
  if (it != varPointers->end()) {
    if (destStaging) {
      // TODO: Do the same thing as the source.
      // If the destination is staging, then the shared coordinates are also the
      // ghost coordinates.
      stagingVar sv;
      sv.device_offset = sharedLowCoordinates;
      sv.device_size = sharedHighCoordinates - sharedLowCoordinates;

      std::map<stagingVar, stagingVarInfo>::iterator staging_it =
          it->second.var->stagingVars.find(sv);
      if (staging_it != it->second.var->stagingVars.end()) {
        d_varDB[i].ghostItem.dest_varDB_index = staging_it->second.varDB_index;
      } else {
        printf("\nERROR:\nGPUDataWarehouse::putGhostCell() didn't find a "
               "staging variable from the device for offset (%d, %d, %d) and "
               "size (%d, %d, %d).\n",
               sharedLowCoordinates.x(), sharedLowCoordinates.y(),
               sharedLowCoordinates.z(), sv.device_size.x(), sv.device_size.y(),
               sv.device_size.z());
        varLock->unlock();
        exit(-1);
      }

    } else {
      d_varDB[i].ghostItem.dest_varDB_index = it->second.varDB_index;
    }
  } else {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell(), label: %s destination "
           "patch ID %d, matlIndx %d, levelIndex %d, staging %s not found in "
           "GPU DW variable database\n",
           label, destPatchID, matlIndx, levelIndx,
           destStaging ? "true" : "false");
    varLock->unlock();
    exit(-1);
  }

  d_dirty = true;
  varLock->unlock();
}

//______________________________________________________________________
//
void GPUDataWarehouse::getSizes(int3 &low, int3 &high, int3 &siz,
                                GhostType &gtype, int &numGhostCells,
                                char const *label, int patchID, int matlIndx,
                                int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo info = varPointers->at(lpml);
    low = info.device_offset;
    high = info.var->device_size + info.var->device_offset;
    siz = info.var->device_size;
    gtype = info.var->gtype;
    numGhostCells = info.var->numGhostCells;
  }
  varLock->unlock();
}
//______________________________________________________________________
// Deep copies (not shallow copies or moves) an entry from one data warehouse to
// another.
// (Note: A deep copy is a full copy of data from one variable's memory
// space to another variable's memory space
// A shallow copy is just a pointer copy and a ref counting.
// A move is a true std::move() reseating.) RMCRT and Arches
// often keep a variable in the old data warehouse alive by copying it to the
// new data warehouse. It can't be a move (it may be needed to use data from the
// old and the new) It can't be a shallow copy (it may be needed to modify the
// new and also use the old) So it must be a deep copy. Both the source and
// destination variables must be in the GPU data warehouse, both must be listed
// as "allocated". If these are not the case, the transferFrom doesn't proceed.
// Both must have the same variable sizes.  If this is not the case, the program
// will exit. If all above conditions are met, then it will do a device to
// device memcopy call. *Important*: For this to work, it needs a GPU stream.
// GPU streams are stored per task, every Uintah task is assigned a possible
// stream to use.  To get the stream you have to request it from the
// detailedTask object. Normal CPU task callback functions do not have access to
// the detailedTask object, but it is possible to extend the callack function
// parameter list so that it does.  See UnifiedSchedulerTest::timeAdvanceUnified
// as an example. *Also important*: For this to work, the destination variable
//*MUST* be listed as a computes in the task that's calling transferFrom(). That
// allows for the computes data to have been preallocated ahead of time by the
// scheduler. Uintah's scheduler is fine if it is able to allocate the space, so
// that it can allow the task developer to write data into space it created.  If
// it was a computes, then this method can copy data into the computes memory,
// and when the task which called transferFrom is done, the scheduler will mark
// this computes variable as VALID. Note: A shallow copy method has been
// requested by the Arches team.  That hasn't been implemented yet.  It would
// require ref counting a variable, and perhaps some sanity checks to ensure a
// shallow copied variable is not called a computes and then later listed as a
// modifies.

// This function is called from OnDemandDataWarehouse.cc only
bool GPUDataWarehouse::transferFrom(gpuStream_t* stream,
                                    GPUDataWarehouse *from, char const *label,
                                    int patchID, int matlIndx, int levelIndx) {

  from->varLock->lock();

  // lock both data warehouses, no way to lock free this section,
  // you could get the dining philosophers problem.
  this->varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator source_it =
      from->varPointers->find(lpml);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator dest_it =
      this->varPointers->find(lpml);
  int proceed = true;
  if (source_it == from->varPointers->end()) {
    // It may just be there wasn't any requires in the GPU to begin with, so
    // don't bother attempting to copy. printf("GPU source not found\n");
    proceed = false;
  } else if (dest_it == this->varPointers->end()) {
    // It may just be there wasn't any computes in the GPU to begin with, so
    // don't bother attempting to copy. printf("GPU dest not found in DW at %p
    // for variable %s patch %d matl %d level %d\n", this, label, patchID,
    // matlIndx, levelIndx);
    proceed = false;
  } else if (((__sync_fetch_and_or(
                   &(source_it->second.var->atomicStatusInGpuMemory), 0) &
               ALLOCATED) != ALLOCATED)) {
    // It may just be there wasn't any computes in the GPU to begin with, so
    // don't bother attempting to copy. printf("GPU source not allocated for
    // variable %s patch %d matl %d level %d, it has status codes %s\n",  label,
    // patchID, matlIndx, levelIndx,
    // getDisplayableStatusCodes(source_it->second.atomicStatusInGpuMemory).c_str());
    proceed = false;

    // Is this  a problem?  We know of this variable in the data warehouse, but
    // we have no space for it. printf("Error: GPUDataWarehouse::transferFrom()
    // - No source variable device space found.  Cannot proceed with deep copy.
    // Exiting...\n"); exit(-1);
  } else if (((__sync_fetch_and_or(
                   &(dest_it->second.var->atomicStatusInGpuMemory), 0) &
               ALLOCATED) != ALLOCATED)) {
    // printf("GPU destination not allocated for variable %s patch %d matl %d
    // level %d\n",  label, patchID, matlIndx, levelIndx); It may just be there
    // wasn't any computes in the GPU to begin with, so don't bother attempting
    // to copy.
    proceed = false;

    // Is a problem?  We know of this variable in the data warehouse, but we
    // have no space for it. printf("Error: GPUDataWarehouse::transferFrom() -
    // No destination variable device space found.  Cannot proceed with deep
    // copy.  Exiting...\n"); exit(-1);
  }
  if (!proceed) {
    from->varLock->unlock();
    this->varLock->unlock();
    return false;
  }

  if (!(source_it->second.var->device_offset.x() == dest_it->second.var->device_offset.x() &&
        source_it->second.var->device_offset.y() == dest_it->second.var->device_offset.y() &&
        source_it->second.var->device_offset.z() == dest_it->second.var->device_offset.z() &&
        source_it->second.var->device_size.x()   == dest_it->second.var->device_size.x() &&
        source_it->second.var->device_size.y()   == dest_it->second.var->device_size.y() &&
        source_it->second.var->device_size.z()   == dest_it->second.var->device_size.z())) {

    printf(
        "Error: GPUDataWarehouse::transferFrom() - The source and destination "
        "variables exists for variable %s patch %d matl %d level %d, but the "
        "sizes don't match.  Cannot proceed with deep copy.  Exiting...\n",
        label, patchID, matlIndx, levelIndx);
    printf("The source size is (%d, %d, %d) with offset (%d, %d, %d) and "
           "device size is (%d, %d, %d) with offset (%d, %d, %d)\n",
           source_it->second.var->device_size.x(),
           source_it->second.var->device_size.y(),
           source_it->second.var->device_size.z(),
           source_it->second.var->device_offset.x(),
           source_it->second.var->device_offset.y(),
           source_it->second.var->device_offset.z(),
           dest_it->second.var->device_size.x(),
           dest_it->second.var->device_size.y(),
           dest_it->second.var->device_size.z(),
           dest_it->second.var->device_offset.x(),
           dest_it->second.var->device_offset.y(),
           dest_it->second.var->device_offset.z());

    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);

  } else if (!(source_it->second.var->device_ptr)) {
    // A couple more santiy checks, this may be overkill...
    printf("Error: GPUDataWarehouse::transferFrom() - No source variable "
           "pointer found for variable %s patch %d matl %d level %d\n",
           label, patchID, matlIndx, levelIndx);
    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);

  } else if (!(dest_it->second.var->device_ptr)) {
    printf("Error: GPUDataWarehouse::transferFrom() - No destination variable "
           "pointer found for variable %s patch %d matl %d level %d\n",
           label, patchID, matlIndx, levelIndx);
    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);

  }

  // We shouldn't need to allocate space on either the source or the destination.
  // The source should have been listed as a "requires", and the  destination
  // should have been listed as a "computes" for the task. And this solves a mess
  // of problems, mainly dealing with when it is listed as allocated and when
  // it's listed as valid.

  // cudaMemcpyDeviceToDevice
  // This is sort of device-to-device copy in SYCL on the same device.
  stream->memcpy(dest_it->second.var->device_ptr,
                 source_it->second.var->device_ptr,
                 source_it->second.var->device_size.x() *
                 source_it->second.var->device_size.y() *
                 source_it->second.var->device_size.z() *
                 source_it->second.var->sizeOfDataType);

  from->varLock->unlock();
  this->varLock->unlock();

  // Let the caller know we found and transferred something.
  return true;
}

//______________________________________________________________________
// Go through all staging vars for a var. See if they are all marked as valid.
bool GPUDataWarehouse::areAllStagingVarsValid(char const *label, int patchID,
                                              int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  if (it != varPointers->end()) {
    for (std::map<stagingVar, stagingVarInfo>::iterator staging_it =
             it->second.var->stagingVars.begin();
         staging_it != it->second.var->stagingVars.end(); ++staging_it) {
      if (!checkValid(staging_it->second.atomicStatusInGpuMemory)) {
        varLock->unlock();
        return false;
      }
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// Simply performs an atomic fetch on the status variable.
// typedef int atomicDataStatus;
// atomicDataStatus
// GPUDataWarehouse::getStatus(atomicDataStatus& status) {
//  return __sync_or_and_fetch(&(status), 0);
//}

//______________________________________________________________________
//
std::string
GPUDataWarehouse::getDisplayableStatusCodes(atomicDataStatus &status) {
  atomicDataStatus varStatus = __sync_or_and_fetch(&status, 0);
  std::string retval = "";
  if (varStatus == 0) {
    retval += "Unallocated ";
  } else {
    if ((varStatus & ALLOCATING) == ALLOCATING) {
      retval += "Allocating ";
    }
    if ((varStatus & ALLOCATED) == ALLOCATED) {
      retval += "Allocated ";
    }
    if ((varStatus & COPYING_IN) == COPYING_IN) {
      retval += "Copying-in ";
    }
    if ((varStatus & VALID) == VALID) {
      retval += "Valid ";
    }
    if ((varStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY) {
      retval += "Awaiting-ghost-copy ";
    }
    if ((varStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS) {
      retval += "Valid-with-ghosts ";
    }
    if ((varStatus & DEALLOCATING) == DEALLOCATING) {
      retval += "Deallocating ";
    }
    if ((varStatus & FORMING_SUPERPATCH) == FORMING_SUPERPATCH) {
      retval += "Forming-superpatch ";
    }
    if ((varStatus & SUPERPATCH) == SUPERPATCH) {
      retval += "Superpatch ";
    }
    if ((varStatus & UNKNOWN) == UNKNOWN) {
      retval += "Unknown ";
    }
  }
  // trim whitespace
  retval.erase(std::find_if(retval.rbegin(), retval.rend(),
                            std::not1(std::ptr_fun<int, int>(std::isspace)))
                   .base(),
               retval.end());
  return retval;
}

//______________________________________________________________________
//

void GPUDataWarehouse::getStatusFlagsForVariableOnGPU(
    bool &correctSize, bool &allocating, bool &allocated, bool &copyingIn,
    bool &validOnGPU, bool &gatheringGhostCells, bool &validWithGhostCellsOnGPU,
    bool &deallocating, bool &formingSuperPatch, bool &superPatch,
    char const *label, const int patchID, const int matlIndx,
    const int levelIndx, const sycl::int3 &offset, const sycl::int3 &size) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);

  if (varPointers->find(lpml) != varPointers->end()) {
    // check the sizes
    allVarPointersInfo vp = varPointers->at(lpml);
    sycl::int3 device_offset = vp.var->device_offset;
    sycl::int3 device_size = vp.var->device_size;
    // DS 12132019: GPU Resize fix. Changed condition == to <= (and >=). If
    // device variable is greater than host, its ok.
    correctSize =
        (device_offset.x() <= offset.x() && device_offset.y() <= offset.y() &&
         device_offset.z() <= offset.z() && device_size.x() >= size.x() &&
         device_size.y() >= size.y() && device_size.z() >= size.z());

    // get the value
    atomicDataStatus varStatus =
        __sync_or_and_fetch(&(vp.var->atomicStatusInGpuMemory), 0);

    allocating = ((varStatus & ALLOCATING) == ALLOCATING);
    allocated = ((varStatus & ALLOCATED) == ALLOCATED);
    copyingIn = ((varStatus & COPYING_IN) == COPYING_IN);
    validOnGPU = ((varStatus & VALID) == VALID);
    gatheringGhostCells =
        ((varStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY);
    validWithGhostCellsOnGPU =
        ((varStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS);
    deallocating = ((varStatus & DEALLOCATING) == DEALLOCATING);
    formingSuperPatch =
        ((varStatus & FORMING_SUPERPATCH) == FORMING_SUPERPATCH);
    superPatch = ((varStatus & SUPERPATCH) == SUPERPATCH);

  } else {
    correctSize = false;
    allocating = false;
    allocated = false;
    copyingIn = false;
    validOnGPU = false;
    gatheringGhostCells = false;
    validWithGhostCellsOnGPU = false;
    formingSuperPatch = false;
    superPatch = false;
  }

  varLock->unlock();
}

//______________________________________________________________________
// returns false if something else already allocated space and we don't have to.
// returns true if we are the ones to allocate the space.
// performs operations with atomic compare and swaps
bool GPUDataWarehouse::compareAndSwapAllocating(atomicDataStatus &status) {

  bool allocating = false;

  while (!allocating) {

    // get the value
    atomicDataStatus oldVarStatus = __sync_or_and_fetch(&status, 0);

    unsigned int refCounter = (oldVarStatus >> 16);

    // if it's allocated, return true
    if (refCounter >= 1) {
      // Something else already took care of it, and it has moved beyond the
      // allocating state into something else.
      return false;
    } else if ((oldVarStatus & UNALLOCATED) != UNALLOCATED) {
      // Sanity check.  The ref counter was zero, but the variable isn't
      // unallocated. We can't have this.
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Something "
             "wrongly modified the atomic status while setting the allocated "
             "flag\n");
      exit(-1);

    } else {
      // Attempt to claim we'll allocate it.  If not go back into our loop and
      // recheck
      short refCounter = 1;
      atomicDataStatus newVarStatus =
          (refCounter << 16) |
          (oldVarStatus & 0xFFFF); // Place in the reference counter and save
                                   // the right 16 bits.
      newVarStatus =
          newVarStatus | ALLOCATING; // It's possible to preserve a flag, such
                                     // as copying in ghost cells.
      allocating =
          __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}

//______________________________________________________________________
// Sets the allocated flag on a variables atomicDataStatus
// This is called after an allocating process completes.  *Only* the thread that
// got a true from compareAndSwapAllocating() should immediately call this.
bool GPUDataWarehouse::compareAndSwapAllocate(atomicDataStatus &status) {

  bool allocated = false;

  // get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch(&status, 0);
  if ((oldVarStatus & ALLOCATING) == 0) {
    // A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Can't "
           "allocate a status if it wasn't previously marked as allocating.\n");
    exit(-1);
  } else if ((oldVarStatus & ALLOCATED) == ALLOCATED) {
    // A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Can't "
           "allocate a status if it's already allocated\n");
    exit(-1);
  } else {
    // Attempt to claim we'll allocate it.  Create what we want the status to
    // look like by turning off allocating and turning on allocated. Note: No
    // need to turn off UNALLOCATED, it's defined as all zero bits. But the
    // below is kept in just for readability's sake.
    atomicDataStatus newVarStatus = oldVarStatus & ~UNALLOCATED;
    newVarStatus = newVarStatus & ~ALLOCATING;
    newVarStatus = newVarStatus | ALLOCATED;

    // If we succeeded in our attempt to claim to allocate, this returns true.
    // If we failed, thats a real problem, and we crash the problem below.
    allocated =
        __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
  }
  if (!allocated) {
    // Another sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Something "
           "wrongly modified the atomic status while setting the allocated "
           "flag\n");
    exit(-1);
  }
  return allocated;
}

//______________________________________________________________________
// Simply determines if a variable has been marked as allocated.
bool GPUDataWarehouse::checkAllocated(atomicDataStatus &status) {

  return ((__sync_or_and_fetch(&status, 0) & ALLOCATED) == ALLOCATED);
}

//______________________________________________________________________
//
bool GPUDataWarehouse::compareAndSwapDeallocating(atomicDataStatus &status) {

  bool deallocating = false;

  while (!deallocating) {

    // get the value
    atomicDataStatus oldVarStatus = __sync_or_and_fetch(&status, 0);

    unsigned int refCounter = (oldVarStatus >> 16);
    if (refCounter == 0 || ((oldVarStatus & DEALLOCATING) == DEALLOCATING) ||
        ((oldVarStatus & 0xFFFF) == UNALLOCATED) ||
        ((oldVarStatus & UNKNOWN) == UNKNOWN)) {
      // There's nothing to deallocate, or something else already deallocated it
      // or is deallocating it. So this thread won't do it.
      return false;
    } else if (refCounter == 1) {
      // Ref counter is 1, we can deallocate it.
      // Leave the refCounter at 1.
      atomicDataStatus newVarStatus =
          (refCounter << 16) |
          (oldVarStatus & 0xFFFF); // Place in the reference counter and save
                                   // the right 16 bits.
      newVarStatus =
          newVarStatus | DEALLOCATING; // Set it to deallocating so nobody else
                                       // can attempt to use it
      bool successfulUpdate =
          __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
      if (successfulUpdate) {
        // Need to deallocate, let the caller know it.
        deallocating = true;
      }
    } else if (refCounter > 1) {
      // Something else is using this variable, don't deallocate, just decrement
      // the counter
      refCounter--;
      atomicDataStatus newVarStatus =
          (refCounter << 16) | (oldVarStatus & 0xFFFF);
      bool successfulUpdate =
          __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
      if (successfulUpdate) {
        // No need to deallocate, let the caller know it.
        return false;
      }
    } else {
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocating( )  This "
             "variable's ref counter was 0, but its status said it was in use. "
             " This shouldn't happen\n");
      exit(-1);
    }
  }
  return true;
}

//______________________________________________________________________
// Sets the allocated flag on a variables atomicDataStatus
// This is called after a deallocating process completes.  *Only* the thread
// that got a true from
// compareAndSwapDeallocating() should immediately call this.
bool GPUDataWarehouse::compareAndSwapDeallocate(atomicDataStatus &status) {

  bool allocated = false;

  // get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch(&status, 0);
  unsigned int refCounter = (oldVarStatus >> 16);

  if ((oldVarStatus & DEALLOCATING) == 0) {
    // A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Can't "
           "deallocate a status if it wasn't previously marked as "
           "deallocating.\n");
    exit(-1);
  } else if ((oldVarStatus & 0xFFFF) == UNALLOCATED) {
    // A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Can't "
           "deallocate a status if it's already deallocated\n");
    exit(-1);
  } else if (refCounter != 1) {
    // A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Attemping "
           "to deallocate a variable but the ref counter isn't the required "
           "value of 1\n");
    exit(-1);
  } else {
    // Attempt to claim we'll deallocate it.  Create what we want the status to
    // look like by turning off all status flags (indicating unallocated), it
    // should also zero out the reference counter.
    atomicDataStatus newVarStatus = UNALLOCATED;

    // If we succeeded in our attempt to claim to deallocate, this returns true.
    // If we failed, thats a real problem, and we crash the problem below.
    allocated =
        __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
  }
  if (!allocated) {
    // Another sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Something "
           "wrongly modified the atomic status while trying set the status "
           "flags to unallocated\n");
    exit(-1);
  }
  return allocated;
}

//______________________________________________________________________
// Simply determines if a variable has been marked as valid.
bool GPUDataWarehouse::checkValid(atomicDataStatus &status) {

  return ((__sync_or_and_fetch(&status, 0) & VALID) == VALID);
}

//______________________________________________________________________
//
bool GPUDataWarehouse::isAllocatedOnGPU(char const *label, int patchID,
                                        int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal =
        ((__sync_fetch_and_or(
              &(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) &
          ALLOCATED) == ALLOCATED);
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//
bool GPUDataWarehouse::isAllocatedOnGPU(char const *label, int patchID,
                                        int matlIndx, int levelIndx,
                                        sycl::int3 offset, sycl::int3 size) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    // cout << "In isAllocatedOnGPU - For patchID " << patchID << " for the
    // status is " <<
    // getDisplayableStatusCodes(varPointers->at(lpml).atomicStatusInGpuMemory)
    // << endl;
    bool retVal =
        ((__sync_fetch_and_or(
              &(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) &
          ALLOCATED) == ALLOCATED);
    if (retVal) {
      // now check the sizes
      sycl::int3 device_offset = varPointers->at(lpml).var->device_offset;
      sycl::int3 device_size = varPointers->at(lpml).var->device_size;
      retVal =
          (device_offset.x() == offset.x() && device_offset.y() == offset.y() &&
           device_offset.z() == offset.z() && device_size.x() == size.x() &&
           device_size.y() == size.y() && device_size.z() == size.z());
    }
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//
bool GPUDataWarehouse::isValidOnGPU(char const *label, int patchID,
                                    int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal =
        ((__sync_fetch_and_or(
              &(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) &
          VALID) == VALID);
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
bool GPUDataWarehouse::compareAndSwapSetValidOnGPU(char const *const label,
                                                   const int patchID,
                                                   const int matlIndx,
                                                   const int levelIndx) {
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
        varPointers->find(lpml);
    if (it != varPointers->end()) {
      atomicDataStatus *status = &(it->second.var->atomicStatusInGpuMemory);
      atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID) == VALID) {
        // Something else already took care of it.  So this task won't manage
        // it.
        varLock->unlock();
        return false;
      } else {
        // Attempt to claim we'll manage the ghost cells for this variable.  If
        // the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
        newVarStatus = newVarStatus | VALID;
        settingValid =
            __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR\nGPUDataWarehouse::compareAndSwapSetValidOnGPU() - Unknown "
             "variable %s on GPUDataWarehouse\n",
             label);
      exit(-1);
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
bool GPUDataWarehouse::compareAndSwapSetValidOnGPUStaging(
    char const *label, int patchID, int matlIndx, int levelIndx, const sycl::int3& offset,
    const sycl::int3& size) {
  varLock->lock();
  bool settingValidOnStaging = false;
  while (!settingValidOnStaging) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
        varPointers->find(lpml);

    if (it != varPointers->end()) {

      stagingVar sv;
      sv.device_offset = offset;
      sv.device_size = size;

      std::map<stagingVar, stagingVarInfo>::iterator staging_it =
          it->second.var->stagingVars.find(sv);

      if (staging_it != it->second.var->stagingVars.end()) {
        atomicDataStatus *status =
            &(staging_it->second.atomicStatusInGpuMemory);
        atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);

        if ((oldVarStatus & VALID) == VALID) {
          // Something else already took care of it.  So this task won't manage
          // it.
          varLock->unlock();
          return false;
        } else {
          // Attempt to claim we'll manage the ghost cells for this variable. If
          // the claim fails go back into our loop and recheck
          atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
          newVarStatus = newVarStatus | VALID;
          settingValidOnStaging =
              __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
        }

      } else {
        varLock->unlock();
        printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetValidOnGPUStaging( "
               ")  Staging variable %s not found.\n",
               label);
        exit(-1);
      }
    } else {
      varLock->unlock();
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetValidOnGPUStaging( )  "
             "Variable %s not found.\n",
             label);
      exit(-1);
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// We have an entry for this item in the GPU DW, and it's not unknown. Therefore
// if this returns true it means this GPU DW specifically knows something about
// the state of this variable. (The reason for the unknown check is currently
// when a var is added to the GPUDW, we also need to state what we know about
// its data in host memory.  Since it doesn't know, it marks it as unknown,
// meaning, the host side DW is possibly managing the data.)
bool GPUDataWarehouse::dwEntryExistsOnCPU(char const *label, int patchID,
                                          int matlIndx, int levelIndx) {

  varLock->lock();
  bool retVal = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  if (it != varPointers->end()) {
    if ((it->second.var->atomicStatusInHostMemory & UNKNOWN) != UNKNOWN) {

      retVal = true;
    }
  }
  varLock->unlock();
  return retVal;
}

//______________________________________________________________________
//
bool GPUDataWarehouse::isValidOnCPU(char const *label, const int patchID,
                                    const int matlIndx, const int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {

    bool retVal =
        ((__sync_fetch_and_or(
              &(varPointers->at(lpml).var->atomicStatusInHostMemory), 0) &
          VALID) == VALID);
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
// TODO: This needs to be turned into a compare and swap operation
//______________________________________________________________________
bool GPUDataWarehouse::compareAndSwapSetValidOnCPU(char const *const label,
                                                   const int patchID,
                                                   const int matlIndx,
                                                   const int levelIndx) {
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
        varPointers->find(lpml);
    if (it != varPointers->end()) {
      atomicDataStatus *status = &(it->second.var->atomicStatusInHostMemory);
      atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID) == VALID) {
        // Something else already took care of it.  So this task won't manage
        // it.
        varLock->unlock();
        return false;
      } else {
        // Attempt to claim we'll manage the ghost cells for this variable.  If
        // the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
        newVarStatus = newVarStatus | VALID;
        settingValid =
            __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR\nGPUDataWarehouse::compareAndSwapSetValidOnCPU() - Unknown "
             "variable %s on GPUDataWarehouse\n",
             label);
      exit(-1);
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// returns false if something else already changed a valid variable to valid
// awaiting ghost data returns true if we are the ones to manage this variable's
// ghost data.
bool GPUDataWarehouse::compareAndSwapAwaitingGhostDataOnGPU(char const *label,
                                                            int patchID,
                                                            int matlIndx,
                                                            int levelIndx) {

  bool allocating = false;

  varLock->lock();
  while (!allocating) {
    // get the address
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      atomicDataStatus *status =
          &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
      atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
      if (((oldVarStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY) ||
          ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
        // Something else already took care of it.  So this task won't manage
        // it.
        varLock->unlock();
        return false;
      } else {
        // Attempt to claim we'll manage the ghost cells for this variable.  If
        // the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus | AWAITING_GHOST_COPY;
        allocating =
            __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapAwaitingGhostDataOnGPU( "
             ")  Variable %s not found.\n",
             label);
      exit(-1);
      return false;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// returns false if something else already claimed to copy or has copied data
// into the GPU. returns true if we are the ones to manage this variable's ghost
// data.
bool GPUDataWarehouse::compareAndSwapCopyingIntoGPU(char const *label,
                                                    int patchID, int matlIndx,
                                                    int levelIndx) {

  atomicDataStatus *status = nullptr;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  if (it != varPointers->end()) {
    status = &(it->second.var->atomicStatusInGpuMemory);
  } else {
    varLock->unlock();
    exit(-1);
    return false;
  }
  varLock->unlock();

  bool copyingin = false;
  while (!copyingin) {
    atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
    if (oldVarStatus == UNALLOCATED) {
      varLock->unlock();
      exit(-1);
    }
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
        ((oldVarStatus & VALID) == VALID) ||
        ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
      // Something else already took care of it.  So this task won't manage it.
      varLock->unlock();
      return false;
    } else {
      // Attempt to claim we'll manage the ghost cells for this variable.  If
      // the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copyingin =
          __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}

//______________________________________________________________________
// returns false if something else already claimed to copy or has copied data
// into the CPU. returns true if we are the ones to manage this variable's ghost
// data.
bool GPUDataWarehouse::compareAndSwapCopyingIntoCPU(char const *label,
                                                    int patchID, int matlIndx,
                                                    int levelIndx) {

  atomicDataStatus *status = nullptr;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  if (varPointers->find(lpml) != varPointers->end()) {
    status = &(it->second.var->atomicStatusInHostMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoCPU( )  "
           "Variable %s not found.\n",
           label);
    exit(-1);
    return false;
  }
  varLock->unlock();

  bool copyingin = false;
  while (!copyingin) {
    // get the address
    atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
        ((oldVarStatus & VALID) == VALID) ||
        ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
      // Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      // Attempt to claim we'll manage the ghost cells for this variable.  If
      // the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      newVarStatus = newVarStatus & ~UNKNOWN;
      copyingin =
          __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}

//______________________________________________________________________
// returns false if something else already claimed to copy or has copied data
// into the GPU. returns true if we are the ones to manage this variable's ghost
// data.
bool GPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging(
    char const *label, int patchID, int matlIndx, int levelIndx, sycl::int3 offset,
    sycl::int3 size) {

  atomicDataStatus *status;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);

  if (it != varPointers->end()) {

    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it =
        it->second.var->stagingVars.find(sv);
    if (staging_it != it->second.var->stagingVars.end()) {
      status = &(staging_it->second.atomicStatusInGpuMemory);
    } else {
      varLock->unlock();
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( ) "
             " Staging variable %s not found.\n",
             label);
      exit(-1);
      return false;
    }
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( )  "
           "Variable %s not found.\n",
           label);
    exit(-1);
    return false;
  }
  varLock->unlock();

  bool copyingin = false;

  while (!copyingin) {
    // get the address
    atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
    if (oldVarStatus == UNALLOCATED) {
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( ) "
             " Variable %s is unallocated.\n",
             label);
      exit(-1);
    } else if ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS) {
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( ) "
             " Variable %s is marked as valid with ghosts, that should never "
             "happen with staging vars.\n",
             label);
      exit(-1);
    } else if (((oldVarStatus & COPYING_IN) == COPYING_IN) /*||
                                                             ((oldVarStatus & VALID) == VALID)*/) {
      // DS 06032020: Commented "((oldVarStatus & VALID) == VALID)" condition as
      // a temporary fix for the defect: When a variable is modified on GPU,
      // ValidWithGhost status is reverted to allow gathering of ghost cells
      // again for the next requires dependency. But as of now there is no
      // mechanism to mark staging variables invalid. As a result, although the
      // ghost cells on the main variable are invalidated, staging variables
      // still have valid status and hold old values and because of valid status
      // prepareDeviceVars does not issue fresh H2D copy for the staging
      // variable. The permanent fix is to find out and inactivate staging
      // variables of neighboring patches. But that needs more time. So this
      // temporary fix to ignore valid status as of now which will cause few
      // redundant h2d copies, but will make code work.
      //      printf("compareAndSwapCopyingIntoGPUStaging: %s %d COPYING_IN: %d
      //      VALID: %d\n", label, patchID, (oldVarStatus & COPYING_IN) ==
      //      COPYING_IN, (oldVarStatus & VALID) == VALID );

      // Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      // Attempt to claim we'll manage the ghost cells for this variable.  If
      // the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      newVarStatus = newVarStatus & ~VALID; // DS 06032020: temp fix
      copyingin =
          __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }

  return true;
}

//______________________________________________________________________
//
bool GPUDataWarehouse::isValidWithGhostsOnGPU(char const *label, int patchID,
                                              int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  if (it != varPointers->end()) {
    bool retVal =
        ((__sync_fetch_and_or(&(it->second.var->atomicStatusInGpuMemory), 0) &
          VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS);
    varLock->unlock();
    return retVal;
  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
// TODO: This needs to be turned into a compare and swap operation
void GPUDataWarehouse::setValidWithGhostsOnGPU(char const *label, int patchID,
                                               int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it =
      varPointers->find(lpml);
  if (it != varPointers->end()) {
    // UNKNOWN
    // make sure the valid is still turned on
    // do not set VALID here because one thread can gather the main patch and
    // other gathers ghost cells if ghost cells is finished first, setting valid
    // here causes task to start even though other thread copying the main patch
    // is not completed. race condition. Removed VALID_WITH_GHOSTS with from
    // compareAndSwapCopyingInto* add extra condition to check valid AND valid
    // with ghost both in UnifiedSchedular::allGPUProcessingVarsReady
    //__sync_or_and_fetch(&(it->second.var->atomicStatusInGpuMemory), VALID);

    // turn off AWAITING_GHOST_COPY
    __sync_and_and_fetch(&(it->second.var->atomicStatusInGpuMemory),
                         ~AWAITING_GHOST_COPY);

    // turn on VALID_WITH_GHOSTS
    __sync_or_and_fetch(&(it->second.var->atomicStatusInGpuMemory),
                        VALID_WITH_GHOSTS);

    varLock->unlock();
  } else {
    varLock->unlock();
    exit(-1);
  }
}

//______________________________________________________________________
// returns true if successful if marking a variable as a superpatch.  False
// otherwise. Can only turn an unallocated variable into a superpatch.
bool GPUDataWarehouse::compareAndSwapFormASuperPatchGPU(char const *label,
                                                        int patchID,
                                                        int matlIndx,
                                                        int levelIndx) {

  bool compareAndSwapSucceeded = false;

  // get the status
  atomicDataStatus *status = nullptr;
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    status = &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapFormASuperPatchGPU( )  "
           "Variable %s patch %d material %d levelIndx %d not found.\n",
           label, patchID, matlIndx, levelIndx);
    exit(-1);
    return false;
  }
  varLock->unlock();

  while (!compareAndSwapSucceeded) {

    atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);

    if ((oldVarStatus & FORMING_SUPERPATCH) == FORMING_SUPERPATCH ||
        ((oldVarStatus & SUPERPATCH) == SUPERPATCH)) {
      // Something else already took care of it.  So this task won't manage it.
      return false;
    } else if (((oldVarStatus & ALLOCATING) == ALLOCATING) ||
               ((oldVarStatus & ALLOCATED) == ALLOCATED) ||
               ((oldVarStatus & ALLOCATING) == ALLOCATING) ||
               ((oldVarStatus & COPYING_IN) == COPYING_IN) ||
               ((oldVarStatus & VALID) == VALID) ||
               ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS) ||
               ((oldVarStatus & DEALLOCATING) == DEALLOCATING)) {
      // Note, we DO allow a variable to be set as AWAITING_GHOST_COPY before
      // anything else.

      // At the time of implementation this scenario shouldn't ever happen.  If
      // so it means Someone is requesting to take a variable already in memory
      // that's not a superpatch and turn it into a superpatch.  It would
      // require some kind of special deep copy mechanism
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapFormASuperPatchGPU( )  "
             "Variable %s cannot be turned into a superpatch, it's in use "
             "already with status %s.\n",
             label, getDisplayableStatusCodes(oldVarStatus).c_str());
      exit(-1);
      return false;
    } else {
      atomicDataStatus newVarStatus = oldVarStatus | FORMING_SUPERPATCH;
      compareAndSwapSucceeded =
          __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }

  atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);

  return true;
}

//______________________________________________________________________
// Sets the allocated flag on a variables atomicDataStatus
// This is called after a forming a superpatch process completes.  *Only* the
// thread that got to set FORMING_SUPERPATCH can set SUPERPATCH.  Further, no
// other thread should modify the atomic status
// compareAndSwapFormASuperPatchGPU() should immediately call this.
bool GPUDataWarehouse::compareAndSwapSetSuperPatchGPU(char const *label,
                                                      int patchID, int matlIndx,
                                                      int levelIndx) {

  bool superpatched = false;

  // get the status
  atomicDataStatus *status = nullptr;
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    status = &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetSuperPatchGPU( )  "
           "Variable %s patch %d material %d levelIndx %d not found.\n",
           label, patchID, matlIndx, levelIndx);
    exit(-1);
    return false;
  }
  varLock->unlock();

  const atomicDataStatus oldVarStatus = __sync_or_and_fetch(status, 0);
  if ((oldVarStatus & FORMING_SUPERPATCH) == 0) {
    // A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetSuperPatchGPU( )  Can't "
           "set a superpatch status if it wasn't previously marked as forming "
           "a superpatch.\n");
    exit(-1);
  } else {
    // Attempt to claim forming it into a superpatch.
    atomicDataStatus newVarStatus = oldVarStatus;
    newVarStatus = newVarStatus & ~FORMING_SUPERPATCH;
    newVarStatus = newVarStatus | SUPERPATCH;

    // If we succeeded in our attempt to claim to deallocate, this returns true.
    // If we failed, thats a real problem, and we crash below.
    // printf("current status is %s oldVarStatus is %s newVarStatus is %s\n",
    // getDisplayableStatusCodes(status)
    superpatched =
        __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
  }
  if (!superpatched) {
    // Another sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetSuperPatchGPU( )  "
           "Something modified the atomic status between the phases of forming "
           "a superpatch and setting a superpatch.  This shouldn't happen\n");
    exit(-1);
  }
  return superpatched;
}

//______________________________________________________________________
//
bool GPUDataWarehouse::isSuperPatchGPU(char const *label, int patchID,
                                       int matlIndx, int levelIndx) {
  bool retVal = false;
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    retVal = ((__sync_fetch_and_or(
                   &(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) &
               SUPERPATCH) == SUPERPATCH);
  }
  varLock->unlock();
  return retVal;
}

//______________________________________________________________________
//
void GPUDataWarehouse::setSuperPatchLowAndSize(
    char const *const label, const int patchID, const int matlIndx,
    const int levelIndx, const sycl::int3 &low, const sycl::int3 &size) {
  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  auto it = varPointers->find(lpml);
  if (it == varPointers->end()) {
    printf("ERROR: GPUDataWarehouse::setSuperPatchLowAndSize - Didn't find a "
           "variable for label %s patch %d matl %d level %d\n",
           label, patchID, matlIndx, levelIndx);
    varLock->unlock();
    exit(-1);
  }
  it->second.var->device_offset = low;
  it->second.var->device_size = size;
  varLock->unlock();
}

//______________________________________________________________________
//

// void GPUDataWarehouse::print() {}
// void *GPUDataWarehouse::getPlacementNewBuffer() { return placementNewBuffer; }
