
#include <iostream>

__global__ void simdtest(float* A) {
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if ( (threadIdx.x % 32) < 8 ) {
    A[globalID] = 1;
  } else {
    A[globalID] = 0;
  }
}

int main()
{
  int THREADS = 256;
  float* A;

  cudaMallocManaged(&A, THREADS*sizeof(float));
  
  std::cout << "launch" << std::endl;
  simdtest<<< THREADS/256, 256 >>>(A);
  cudaDeviceSynchronize();
  std::cout << "A: ";
  for ( int i = 0; i < THREADS; ++i) {
    std::cout << A[i] << ", ";
  }
  std::cout << std::endl;
}
