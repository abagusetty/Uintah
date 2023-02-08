#pragma once

#include <CCA/Components/Schedulers/GPUStreamPool.h>

#include <cstddef>
#include <new>
#include <unordered_map>
#include <vector>

namespace Uintah {

class GPUMemoryPool {
protected:
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;

private:
  GPUMemoryPool() { reserve_ = 5; }
  ~GPUMemoryPool() { ReleaseAll(); }

public:
  void *allocate(int device_id, std::size_t size) {
    auto&& reuse_it = memory_pool_.find(size);
    if(reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
      size_t free{}, total{};

      GPU_RT_SAFE_CALL(gpuMemGetInfo(&free, &total));

      if(size > free - total * reserve_ / 100) ReleaseAll();

      void* ret = nullptr;

#if defined(HAVE_CUDA)
      GPU_RT_SAFE_CALL( cudaMalloc(&ret, size) );
#elif defined(HAVE_HIP)
      GPU_RT_SAFE_CALL( hipMalloc(&ret, size) );
#elif defined(HAVE_DPCPP)
      gpuStream_t& stream = Uintah::GPUStreamPool::getInstance().getStream();
      ret                 = sycl::malloc_device(size, stream);
#endif

      used_memory_ += size;
      return ret;
    }
    else {
      auto&& reuse_pool = reuse_it->second;
      auto   ret        = reuse_pool.back();
      reuse_pool.pop_back();
      return ret;
    }
  }
  
  void deallocate(int device_id, void* ptr, size_t size) {
    auto&& reuse_pool = memory_pool_[size];
    reuse_pool.push_back(ptr);
  }

  void gpuMemset(void** ptr, size_t sizeInBytes) {
    gpuStream_t& stream = tamm::GPUStreamPool::getInstance().getStream();

#if defined(USE_DPCPP)
    stream.memset(*ptr, 0, sizeInBytes);
#elif defined(USE_HIP)
    hipMemsetAsync(*ptr, 0, sizeInBytes);
#elif defined(USE_CUDA)
    cudaMemsetAsync(*ptr, 0, sizeInBytes, stream);
#endif
  }
  
  void ReleaseAll() {
    for(auto&& i: memory_pool_) {
      for(auto&& j: i.second) {
#if defined(HAVE_CUDA)
        GPU_RT_SAFE_CALL(cudaFree(j));
#elif defined(HAVE_HIP)
        GPU_RT_SAFE_CALL(hipFree(j));
#elif defined(HAVE_DPCPP)
        gpuStream_t&stream = Uintah::GPUStreamPool::getInstance().getStream();
        stream->wait();
        sycl::free(j, stream);
#endif
        used_memory_ -= i.first;
      }
    }
    memory_pool_.clear();
  }

  /// Returns the instance of device manager singleton.
  inline static GPUMemoryPool& getInstance() {
    static GPUMemoryPool d_m{};
    return d_m;
  }

  GPUMemoryPool(const GPUMemoryPool&)            = delete;
  GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;
  GPUMemoryPool(GPUMemoryPool&&)                 = delete;
  GPUMemoryPool& operator=(GPUMemoryPool&&)      = delete;

}; // class GPUMemoryPool

} // namespace Uintah
