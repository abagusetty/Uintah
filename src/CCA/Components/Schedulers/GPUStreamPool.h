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

  // [CUDA] Operations within the same stream are ordered (FIFO) and cannot
  // overlap. [SYCL] Operations within the same queue are out-of-order and can be
  // overlapped. Operations in different streams are unordered and can overlap
  // For this reason we let each task own a stream, as we want one task to be
  // able to run if it is ready to do work even if another task is not yet
  // ready. It also enables us to easily determine when a computed variable is
  // "valid" because when that task's stream completes, then we can infer the
  // variable is ready to go.  More about how a task claims a stream can be
  // found in DetailedTasks.cc
  gpuStream_t* s_idle_streams{nullptr};

  // thread-safe, GPU-specific counter for getting a round-robin stream/queue
  // from pool
  unsigned int *s_count{nullptr};
  // total number of GPUs on node
  int s_ngpus{0};

private:
  GPUStreamPool() {
    GPU_RT_SAFE_CALL( gpuGetDeviceCount(&s_ngpus) );
    s_count = new unsigned int[s_ngpus];

    for (int devID = 0; devID < s_ngpus; devID++) { // # of GPUs per node
      GPU_RT_SAFE_CALL( gpuSetDevice(devID) );
      s_count[devID] = 0;
      s_idle_streams = new gpuStream_t[N];
      for (int streamID = 0; streamID < N; streamID++) { // # of streams per GPU
#if defined(HAVE_CUDA)
        GPU_RT_SAFE_CALL(cudaStreamCreateWithFlags(&s_idle_streams[streamID], cudaStreamNonBlocking));
#elif defined(HAVE_HIP)
        GPU_RT_SAFE_CALL(hipStreamCreateWithFlags(&s_idle_streams[streamID], hipStreamNonBlocking));
#elif defined(HAVE_SYCL)
        s_idle_streams[streamID] = new sycl::queue(*sycl_get_context(devID),
                                                   *sycl_get_device(devID),
                                                   sycl_asynchandler);
#endif
      } // streamID
    } // devID
  }

  ~GPUStreamPool() {}

  void check_device(int deviceID) {
    if(deviceID > s_ngpus) {
      std::cout << "Caught INVALID deviceID passed to getStreams(): " << std::endl
                << deviceID << std::endl;
      std::exit(-1);
    }
  }

public:
  /// Returns a default/first GPU stream
  gpuStream_t* getDefaultGpuStreamFromPool(int deviceID) {
    check_device(deviceID);
    return &(s_idle_streams[deviceID*N + 0]);
  }

  /// Returns a GPU stream in a round-robin fashion
  gpuStream_t* getGpuStreamFromPool(int deviceID) {
    check_device(deviceID);
    unsigned short int counter = (s_count[deviceID])++ % N;
    return &(s_idle_streams[deviceID*N + counter]);
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
