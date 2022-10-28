#pragma once

#include "device_memory_resource.hpp"

#include <cstddef>

namespace rmm::mr {
/**
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation.
 */
class gpu_memory_resource final: public device_memory_resource {
public:
  gpu_memory_resource()                                      = default;
  ~gpu_memory_resource() override                            = default;
  gpu_memory_resource(gpu_memory_resource const&)            = default;
  gpu_memory_resource(gpu_memory_resource&&)                 = default;
  gpu_memory_resource& operator=(gpu_memory_resource const&) = default;
  gpu_memory_resource& operator=(gpu_memory_resource&&)      = default;

private:
  /**
   * @brief Allocates memory of size at least `bytes` using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @note Stream argument is ignored
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, gpuStream_t* stream) override {
    void* ptr{nullptr};
#if defined(HAVE_CUDA)
    GPU_RT_SAFE_CALL(cudaMalloc(&ptr, bytes));
#elif defined(HAVE_HIP)
    GPU_RT_SAFE_CALL(hipMalloc(&ptr, bytes));
#elif defined(HAVE_SYCL)
    ptr = sycl::malloc_device(bytes, *stream);
#endif

    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @note Stream argument is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t bytes, gpuStream_t* stream) override {
#if defined(HAVE_CUDA)
    GPU_RT_SAFE_CALL(cudaFree(ptr));
#elif defined(HAVE_HIP)
    GPU_RT_SAFE_CALL(hipFree(ptr));
#elif defined(HAVE_SYCL)
    sycl::free(ptr, *stream);
#endif
  }
};
} // namespace rmm::mr
