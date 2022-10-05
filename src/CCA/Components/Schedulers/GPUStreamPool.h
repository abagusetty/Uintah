/*
 * The MIT License
 *
 * Copyright (c) 1997-2022 The University of Utah
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

#include <sci_defs/gpu_defs.h>

#include <map>
#include <vector>

#if defined(HAVE_SYCL)
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
#endif // HAVE_SYCL

namespace Uintah {

void gpuGetDeviceCount(int *ngpus) {
#if defined(HAVE_CUDA)
  cudaGetDeviceCount(ngpus);
#elif defined(HAVE_HIP)
  hipGetDeviceCount(ngpus);
#elif defined(HAVE_SYCL)
  syclGetDeviceCount(ngpus);
#else
  *ngpus = -1;
#endif
}

void gpuSetDevice(int devID) {
#if defined(HAVE_CUDA)
  cudaSetDevice(devID);
#elif defined(HAVE_HIP)
  hipSetDevice(devID);
#elif defined(HAVE_SYCL)
  syclSetDevice(devID);
#endif
}

/* MasterLock streampool_mutex{}; */
#if defined(HAVE_CUDA)
template <int N = 32>
#elif defined(HAVE_HIP)
template <int N = 4>
#elif defined(HAVE_SYCL)
template <int N = 1>
#endif
class GPUStreamPool {
protected:
  // thread shared data, needs lock protection when accessed

  //[CUDA] Operations within the same stream are ordered (FIFO) and cannot
  //overlap. [SYCL] Operations within the same queue are out-of-order and can be
  //overlapped. Operations in different streams are unordered and can overlap
  // For this reason we let each task own a stream, as we want one task to be
  // able to run if it is ready to do work even if another task is not yet
  // ready. It also enables us to easily determine when a computed variable is
  // "valid" because when that task's stream completes, then we can infer the
  // variable is ready to go.  More about how a task claims a stream can be
  // found in DetailedTasks.cc
  std::map<int, std::vector<gpuStream_t *>> s_idle_streams;

  // thread-safe, GPU-specific counter for getting a round-robin stream/queue
  // from pool
  unsigned int *s_count{nullptr};

private:
  GPUStreamPool() {
    int numGPUs_{};
    gpuGetDeviceCount(&numGPUs_);
    s_count = new unsigned int[numGPUs_];

    for (int devID = 0; devID < numGPUs_; devID++) { // # of GPUs per node
      gpuSetDevice(devID);
      s_count[devID] = 0;

      std::vector<gpuStream_t *> streamVec;
      for (int j = 0; j < N; j++) { // # of streams per GPU
        gpuStream_t *stream = nullptr;
#if defined(HAVE_CUDA)
        stream = new cudaStream_t;
        cudaStreamCreate(stream);
#elif defined(HAVE_HIP)
        stream = new hipStream_t;
        hipStreamCreate(stream);
#elif defined(HAVE_SYCL)
        stream = new sycl::queue(*sycl_get_context(devID),
                                 *sycl_get_device(devID), sycl_asynchandler);
#endif
        streamVec.push_back(stream);
      } // j
      s_idle_streams[devID] = std::move(streamVec);
    }
  }

  ~GPUStreamPool() {}

public:
  /// Returns a default/first GPU stream in a round-robin fashion
  gpuStream_t *getDefaultGpuStreamFromPool(int deviceID) {
    auto stream_iter = s_idle_streams.find(deviceID);
    if (stream_iter != s_idle_streams.end()) {
      return (stream_iter->second).front();
    } else {
      std::exit(-1);
    }
  }

  /// Returns a GPU stream in a round-robin fashion
  gpuStream_t *getGpuStreamFromPool(int deviceID) {
    unsigned int streamCounter = (s_count[deviceID])++ % N;
    auto stream_iter = s_idle_streams.find(deviceID);
    if (stream_iter != s_idle_streams.end()) {
      return (stream_iter->second)[streamCounter];
    } else {
      std::exit(-1);
    }
  }

  /// Returns the instance of device manager singleton.
  inline static GPUStreamPool &getInstance() {
    static GPUStreamPool d_m{};
    return d_m;
  }

  // eliminate copy, assignment and move
  GPUStreamPool(const GPUStreamPool &) = delete;
  GPUStreamPool &operator=(const GPUStreamPool &) = delete;
  GPUStreamPool(GPUStreamPool &&) = delete;
  GPUStreamPool &operator=(GPUStreamPool &&) = delete;
};

} // namespace Uintah

// #if defined(HAVE_CUDA)
// template class Uintah::GPUStreamPool<32>;
// #elif defined(HAVE_HIP)
// template class Uintah::GPUStreamPool<4>;
// #elif defined(HAVE_SYCL)
// template class Uintah::GPUStreamPool<1>;
// #endif
