#include "test_sp.hpp"

int main() {

  // for (int j=0; j<4; j++) {
  //   std::cout << "==================device ID : " << j << std::endl;
  //   for (int i=0; i<100; i++) {
  //     auto& pool = Uintah::GPUStreamPool<>::getInstance();
  //     cudaStream_t* str = pool.getGpuStreamFromPool(j);
  //   }
  // }

  unsigned short numGPUs{};
  cudaGetDeviceCount(&numGPUs);
  std::cout << "numGPUs : " << numGPUs << std::endl;

  return 0;
}
