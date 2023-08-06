// kernel_example.cu
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_example.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template<typename T>
__global__ void ExampleCudaKernel(const int size, const T *in, T *out) {
  // printf("blockDim.x: %d, gridDim.x: %d, blockIdx.x: %d, threadIdx.x: %d\n", blockDim.x, gridDim.x, blockIdx.x, threadIdx.x);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * __ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template<typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice &d, int size, const T *in, T *out) {
  std::cout << "ExampleFunctor<GPUDevice, T>::operator()" << std::endl;
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int thread_per_block = 20;
  int block_count = (size + thread_per_block - 1) / thread_per_block;
  ExampleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32_t>;

#endif // GOOGLE_CUDA