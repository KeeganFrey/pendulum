#ifndef MATRIX_CU
#define MATRIX_CU

__global__ void mmul_c(const float *A, const float *B, float *C, const int block_size, const int M, const int N, const int K);
__global__ void mmul(const float *A, const float *B, float *C, const int block_size, const int M, const int N, const int K);
__global__ void transpose_c(float *matrix, int m, int n);

#endif