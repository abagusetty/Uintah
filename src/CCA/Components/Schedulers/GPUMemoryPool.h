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

#include <CCA/Components/Schedulers/GPUStreamPool.h>

#include <CCA/Components/Schedulers/mr/device_memory_resource.hpp>
#include <CCA/Components/Schedulers/mr/gpu_memory_resource.hpp>
#include <CCA/Components/Schedulers/mr/per_device_resource.hpp>
#include <CCA/Components/Schedulers/mr/pool_memory_resource.hpp>

namespace Uintah {

class GPUMemoryPool {
protected:
  using pool_mr =
      rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::vector<std::unique_ptr<pool_mr>> per_device_mr_;

private:
  GPUMemoryPool() {
    // GPU Stream Pool as singleton object
    // It is important to set-device first before getting a stream handle from
    // pool
    auto &streamPool = GPUStreamPool<>::getInstance();

    int ngpus{};
    gpuGetDeviceCount(&ngpus);
    for (int devID = 0; devID < ngpus; devID++) {
      gpuSetDevice(devID);
      auto dev_stream = streamPool.getDefaultGpuStreamFromPool(devID);
      per_device_mr_.push_back(std::make_unique<pool_mr>(
          rmm::mr::get_per_device_resource(devID), dev_stream));
    }
  }

public:
  void *allocateGpuSpaceFromPool(int device_id, std::size_t memSize) {
    auto &streamPool = GPUStreamPool<>::getInstance();
    void *addr = per_device_mr_[device_id].get()->allocate(
        memSize, streamPool.getDefaultGpuStreamFromPool(device_id));
    return addr;
  }
  // TODO: ABB 08/27/22 check to ensure that the pointer, memSize returned to the
  // pool were obtained using the above API
  void freeGpuSpaceToPool(int device_id, void *addr, std::size_t memSize) {
    auto &streamPool = GPUStreamPool<>::getInstance();
    per_device_mr_[device_id].get()->deallocate(
        addr, memSize, streamPool.getDefaultGpuStreamFromPool(device_id));
  }

  /// Returns the instance of GPUMemoryPool singleton.
  inline static GPUMemoryPool &getInstance() {
    static GPUMemoryPool m_mempool_{};
    return m_mempool_;
  }

  GPUMemoryPool(const GPUMemoryPool &) = delete;
  GPUMemoryPool &operator=(const GPUMemoryPool &) = delete;
  GPUMemoryPool(GPUMemoryPool &&) = delete;
  GPUMemoryPool &operator=(GPUMemoryPool &&) = delete;
};

} // namespace Uintah
