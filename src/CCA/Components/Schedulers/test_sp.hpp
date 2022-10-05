#pragma once

#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <iostream>

namespace Uintah {

  template <unsigned short N=16>
  class GPUStreamPool {
  protected:
    std::map <unsigned short, std::vector<cudaStream_t*>> s_idle_streams;
    // GPU device specific counters
    unsigned int *_count{nullptr};

  private:
    GPUStreamPool() {
      unsigned short numGPUs_{};
      cudaGetDeviceCount(&numGPUs_);
      std::cout << "numGPUs : " << numGPUs_ << std::endl;
      _count = new unsigned int[numGPUs_];
      for (int i=0; i<numGPUs_; i++) {
	cudaSetDevice(i);
	_count[i] = 0;

	std::vector<cudaStream_t*> streamVec(N);
	for (int j=0; j<N; j++) {
	  cudaStream_t* stream = new cudaStream_t;
	  cudaStreamCreate(stream);
	  streamVec[j] = stream;
	}
	std::cout << "deviceID : " << i << ", " << "size of vector : " << streamVec.size() << std::endl;
	s_idle_streams[i] = std::move(streamVec);	
      }
    }
    ~GPUStreamPool() {}

  public:
    /// Returns a GPU stream in a round-robin fashion
    cudaStream_t* getGpuStreamFromPool(int deviceID) {
      unsigned short streamCounter = (_count[deviceID])++ % N;
      std::cout << "values of counter : " << streamCounter << " / " << _count[deviceID] << std::endl;    
      return (s_idle_streams[deviceID])[streamCounter];
    }

    /// Returns the instance of device manager singleton.
    inline static GPUStreamPool& getInstance() {
      static GPUStreamPool d_m{};
      return d_m;
    }

    // eliminate copy, assignment and move
    GPUStreamPool(const GPUStreamPool&)            = delete;
    GPUStreamPool& operator=(const GPUStreamPool&) = delete;
    GPUStreamPool(GPUStreamPool&&)                 = delete;
    GPUStreamPool& operator=(GPUStreamPool&&)      = delete;
  };

} // namespace Uintah
