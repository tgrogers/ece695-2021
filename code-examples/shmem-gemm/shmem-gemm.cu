#define TILE_WIDTH 32

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float Pvalue = 0;

  // Loop over the M and N tiles required to compute the P element
  for (int p = 0; p < Width/TILE_WIDTH; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
    ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; ++i)
      Pvalue += ds_M[ty][i] * ds_N[i][tx];
    __syncthreads();
  }
  P[Row*Width+Col] = Pvalue;
}

#include <iostream>

int main()
{
  const int size = 1024;
  float* M;
  float* N;
  float* P;
  cudaMallocManaged(&M, size*size*sizeof(float));
  cudaMallocManaged(&N, size*size*sizeof(float));
  cudaMallocManaged(&P, size*size*sizeof(float));

  MatrixMulKernel<<<(size/TILE_WIDTH, size/TILE_WIDTH),
                    (TILE_WIDTH, TILE_WIDTH)>>>(M,N,P,size);
  cudaDeviceSynchronize();
  std::cout << P[0];

}

