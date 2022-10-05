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

#pragma once

#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

namespace Uintah {

  struct pool_tag {};
  using pool_monitor = Uintah::CrowdMonitor<pool_tag>;
  
  class GPUMemoryPool;

  class GPUMemoryPool {

  public:

    struct gpuMemoryPoolDevicePtrValue {
      unsigned int timestep;
      std::size_t       size;
    };
    struct gpuMemoryPoolDevicePtrItem {
      unsigned int  device_id;
      void*         ptr;

      gpuMemoryPoolDevicePtrItem(unsigned int device_id, void* ptr) {
	this->device_id = device_id;
	this->ptr = ptr;
      }

      //This so it can be used in an STL map
      bool operator<(const gpuMemoryPoolDevicePtrItem& right) const {
	if (this->device_id < right.device_id) {
	  return true;
	} else if ((this->device_id == right.device_id) && (this->ptr < right.ptr)) {
	  return true;
	} else {
	  return false;
	}
      }
    };

    template <typename T>
    struct gpuMemoryPoolDeviceSizeValue {
      T* ptr;
    };
    struct gpuMemoryPoolDeviceSizeItem {

      unsigned short device_id;
      std::size_t         deviceSize;

      gpuMemoryPoolDeviceSizeItem(unsigned short device_id, std::size_t deviceSize) {
	this->device_id = device_id;
	this->deviceSize = deviceSize;
      }
      //This so it can be used in an STL map
      bool operator<(const gpuMemoryPoolDeviceSizeItem& right) const {
	if (this->device_id < right.device_id) {
	  return true;
	} else if ((this->device_id == right.device_id) && (this->deviceSize < right.deviceSize)) {
	  return true;
	} else {
	  return false;
	}
      }
    };

    template <typename T>
    static T* allocateGpuSpaceFromPool(unsigned short device_id, std::size_t count) {
      // Right now the memory pool assumes that each time step is going to be using
      // variables of the same size as the previous time step So for that reason
      // there should be 100% recycling after the 2nd timestep or so. If a task is
      // constantly using different memory sizes, this pool doesn't deallocate
      // memory yet, so it will fail.

      T *addr = nullptr;
      {
	pool_monitor pool_write_lock{Uintah::CrowdMonitor<pool_tag>::WRITER};

	gpuMemoryPoolDeviceSizeItem item(device_id, sizeof(T)*count);

	std::multimap<gpuMemoryPoolDeviceSizeItem, gpuMemoryPoolDeviceSizeValue<T>>::iterator ret = gpuMemoryPoolUnused->find(item);

	if (ret != gpuMemoryPoolUnused->end()) {
	  // we found one
	  addr = ret->second.ptr;
	  gpuMemoryPoolDevicePtrValue insertValue;
	  insertValue.timestep = 99999;
	  insertValue.size = sizeof(T)*count;
	  gpuMemoryPoolInUse->insert(
	    std::pair<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>(gpuMemoryPoolDevicePtrItem(device_id, addr), insertValue));
	  gpuMemoryPoolUnused->erase(ret);
	} else {
	  // There wasn't one, Set the device
	  OnDemandDataWarehouse::uintahSetGpuDevice(device_id);

	  // Allocate the memory.
#ifdef HAVE_CUDA
	  cudaError_t err;
	  err = cudaMalloc((void **)&addr, sizeof(T)*count);
	  if (err == cudaErrorMemoryAllocation) {
	    printf("The GPU memory pool is full.  Need to clear!\n");
	    exit(-1);
	  }
#elif defined(HAVE_HIP)
	  hipError_t err;
	  err = hipMalloc((void **)&addr, sizeof(T)*count);
	  if (err == hipErrorMemoryAllocation) {
	    printf("The GPU memory pool is full.  Need to clear!\n");
	    exit(-1);
	  }	  
#elif defined(HAVE_SYCL)
	  addr = sycl::malloc_device<T>(count, *(sycl_get_device(device_id)),
					*(sycl_get_context(device_id)));
	  if (addr == nullptr) {
	    printf("The SYCL GPU memory pool allocation failed!\n");
	    exit(-1);
	  }
#endif

	  gpuMemoryPoolDevicePtrValue insertValue;
	  insertValue.timestep = 99999;
	  insertValue.size = count * sizeof(T);
	  gpuMemoryPoolInUse->insert(
	    std::pair<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>(gpuMemoryPoolDevicePtrItem(device_id, addr), insertValue));
	}
      } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

      return addr;
    }

    template <typename T>
    static bool freeGpuSpaceFromPool(unsigned short device_id, T* addr) {
      {
	pool_monitor pool_write_lock{Uintah::CrowdMonitor<pool_tag>::WRITER};

	std::size_t memSize;
	gpuMemoryPoolDevicePtrItem item(device_id, addr);

	std::multimap<gpuMemoryPoolDevicePtrItem,
		      gpuMemoryPoolDevicePtrValue>::iterator ret =
	  gpuMemoryPoolInUse->find(item);

	if (ret != gpuMemoryPoolInUse->end()) {
	  // We found it
	  memSize = ret->second.size;

	  gpuMemoryPoolDeviceSizeItem insertItem(device_id, memSize);
	  gpuMemoryPoolDeviceSizeValue<T> insertValue;
	  insertValue.ptr = addr;

	  gpuMemoryPoolUnused->insert(
	    std::pair<gpuMemoryPoolDeviceSizeItem, gpuMemoryPoolDeviceSizeValue<T>>(
	      insertItem, insertValue));
	  gpuMemoryPoolInUse->erase(ret);

	} else {
	  printf("ERROR: GPUMemoryPool::freeGpuSpaceFromPool - No memory found at "
		 "pointer %p on device %u\n",
		 addr, device_id);
	  return false;
	}
      } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

      return true;

      // TODO: Actually deallocate!!!
    }

  private:

    //For a given device and address, holds the timestep
    static std::multimap<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue> *gpuMemoryPoolInUse;
    static std::multimap<gpuMemoryPoolDeviceSizeItem, gpuMemoryPoolDeviceSizeValue<T>> *gpuMemoryPoolUnused;

  };

} //end namespace
