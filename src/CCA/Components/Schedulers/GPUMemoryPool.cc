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

#include <CCA/Components/Schedulers/GPUMemoryPool.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>

#ifdef HAVE_SYCL
// std::vector<std::pair<sycl::device *, sycl::context *>> OnDemandDataWarehouse::syclDevices;

auto sycl_asynchandler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &ex) {
      std::cout << "Caught asynchronous SYCL exception:" << std::endl
                << ex.what() << ", SYCL code: " << ex.code() << std::endl;
    }
  }
};
#endif

std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem,
              Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>
    *Uintah::GPUMemoryPool::gpuMemoryPoolInUse =
        new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem,
                          Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>;

std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem,
              Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>
    *Uintah::GPUMemoryPool::gpuMemoryPoolUnused =
        new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem,
                          Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>;

std::map<unsigned int, std::queue<gpuStream_t *>>
    *Uintah::GPUMemoryPool::s_idle_streams =
        new std::map<unsigned int, std::queue<gpuStream_t *>>;

extern Uintah::MasterLock cerrLock;

namespace {
Uintah::MasterLock idle_streams_mutex{};

struct pool_tag {};
using pool_monitor = Uintah::CrowdMonitor<pool_tag>;
} // namespace

namespace Uintah {

//______________________________________________________________________
//
void *GPUMemoryPool::allocateGpuSpaceFromPool(unsigned int device_id,
                                              size_t memSize) {

  // Right now the memory pool assumes that each time step is going to be using
  // variables of the same size as the previous time step So for that reason
  // there should be 100% recycling after the 2nd timestep or so. If a task is
  // constantly using different memory sizes, this pool doesn't deallocate
  // memory yet, so it will fail.

  void *addr = nullptr;
  {
    pool_monitor pool_write_lock{Uintah::CrowdMonitor<pool_tag>::WRITER};

    gpuMemoryPoolDeviceSizeItem item(device_id, memSize);

    std::multimap<gpuMemoryPoolDeviceSizeItem,
                  gpuMemoryPoolDeviceSizeValue>::iterator ret =
        gpuMemoryPoolUnused->find(item);

    if (ret != gpuMemoryPoolUnused->end()) {
      // we found one
      addr = ret->second.ptr;
      gpuMemoryPoolDevicePtrValue insertValue;
      insertValue.timestep = 99999;
      insertValue.size = memSize;
      gpuMemoryPoolInUse->insert(
          std::pair<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>(
              gpuMemoryPoolDevicePtrItem(device_id, addr), insertValue));
      gpuMemoryPoolUnused->erase(ret);
    } else {
      // There wasn't one
      // Set the device
      OnDemandDataWarehouse::uintahSetGpuDevice(device_id);

      // Allocate the memory.
#ifdef HAVE_CUDA
      cudaError_t err;
      err = cudaMalloc(&addr, memSize);
      if (err == cudaErrorMemoryAllocation) {
        printf("The GPU memory pool is full.  Need to clear!\n");
        exit(-1);
      }
#elif defined(HAVE_SYCL)
      addr = sycl::malloc_device(memSize,
                                 *(sycl_get_device(device_id)),
                                 *(sycl_get_context(device_id)));
      if (addr == nullptr) {
        printf("The SYCL GPU memory pool allocation failed!\n");
        exit(-1);
      }
#endif

      gpuMemoryPoolDevicePtrValue insertValue;
      insertValue.timestep = 99999;
      insertValue.size = memSize;
      gpuMemoryPoolInUse->insert(
          std::pair<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>(
              gpuMemoryPoolDevicePtrItem(device_id, addr), insertValue));
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return addr;
}

//______________________________________________________________________
//
bool GPUMemoryPool::freeGpuSpaceFromPool(unsigned int device_id, void *addr) {

  {
    pool_monitor pool_write_lock{Uintah::CrowdMonitor<pool_tag>::WRITER};

    size_t memSize;
    gpuMemoryPoolDevicePtrItem item(device_id, addr);

    std::multimap<gpuMemoryPoolDevicePtrItem,
                  gpuMemoryPoolDevicePtrValue>::iterator ret =
        gpuMemoryPoolInUse->find(item);

    if (ret != gpuMemoryPoolInUse->end()) {
      // We found it
      memSize = ret->second.size;

      gpuMemoryPoolDeviceSizeItem insertItem(device_id, memSize);
      gpuMemoryPoolDeviceSizeValue insertValue;
      insertValue.ptr = addr;

      gpuMemoryPoolUnused->insert(
          std::pair<gpuMemoryPoolDeviceSizeItem, gpuMemoryPoolDeviceSizeValue>(
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

//______________________________________________________________________
//
void GPUMemoryPool::freeGpuStreamsFromPool() {

  idle_streams_mutex.lock();
  {
    unsigned int totalStreams = 0;
    for (std::map<unsigned int, std::queue<gpuStream_t *>>::const_iterator it =
             s_idle_streams->begin();
         it != s_idle_streams->end(); ++it) {
      totalStreams += it->second.size();
    }

    for (std::map<unsigned int, std::queue<gpuStream_t *>>::const_iterator it =
             s_idle_streams->begin();
         it != s_idle_streams->end(); ++it) {
      unsigned int device = it->first;
      OnDemandDataWarehouse::uintahSetGpuDevice(device);

      while (!s_idle_streams->operator[](device).empty()) {
        gpuStream_t *stream = s_idle_streams->operator[](device).front();
        s_idle_streams->operator[](device).pop();
#ifdef HAVE_CUDA
        cudaError_t retVal;
        CUDA_RT_SAFE_CALL(retVal = cudaStreamDestroy(*stream));
        free(stream);
#elif defined(HAVE_SYCL)
        delete stream;
#endif
      }
    }

  }
  idle_streams_mutex.unlock();
}

//______________________________________________________________________
//
gpuStream_t *GPUMemoryPool::getGpuStreamFromPool(int device) {
  gpuStream_t *stream = nullptr;

  idle_streams_mutex.lock();
  {
    if (s_idle_streams->operator[](device).size() > 0) {
      stream = s_idle_streams->operator[](device).front();
      s_idle_streams->operator[](device).pop();
    } else { // shouldn't need any more than the queue capacity, but in case
      OnDemandDataWarehouse::uintahSetGpuDevice(device);

      // this will get put into idle stream queue and ultimately deallocated
      // after final timestep
#ifdef HAVE_CUDA
      cudaError_t retVal;
      stream = ((gpuStream_t *)malloc(sizeof(gpuStream_t)));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));
#elif defined(HAVE_SYCL)
      stream = new sycl::queue(*(sycl_get_context(device)),
                               *(sycl_get_device(device)),
                               sycl_asynchandler,
                               sycl::property_list{sycl::property::queue::in_order{}});
#endif
    }
  }
  idle_streams_mutex.unlock();
  return stream;
}

//______________________________________________________________________
//
// Operations within the same stream are ordered (FIFO) and cannot overlap.
// Operations in different streams are unordered and can overlap
// For this reason we let each task own a stream, as we want one task to be able
// to run if it is able to do so even if another task is not yet ready.
void GPUMemoryPool::reclaimGpuStreamsIntoPool(DetailedTask *dtask) {

  // reclaim DetailedTask streams
  std::set<unsigned int> deviceNums = dtask->getDeviceNums();
  for (std::set<unsigned int>::iterator iter = deviceNums.begin();
       iter != deviceNums.end(); ++iter) {
    gpuStream_t *stream = dtask->getGpuStreamForThisTask(*iter);
    if (stream != nullptr) {

      idle_streams_mutex.lock();
      { s_idle_streams->operator[](*iter).push(stream); }
      idle_streams_mutex.unlock();

      // It seems that task objects persist between timesteps.  So make sure we
      // remove all knowledge of any formerly used streams.
      dtask->clearGpuStreamsForThisTask();
    }
  }
}

} // end namespace Uintah
