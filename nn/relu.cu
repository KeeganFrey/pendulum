#include "activations.h"

// A more robust implementation for applying relu to a matrix
__global__ void relu_all(float *matrix, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = col * m + row;

    if (row < m && col < n) {
        if (matrix[index] < 0.0f) {
            matrix[index] = 0.0f;
        }
    }
}

//performs the relu operation on the whole mxn matrix
//assume each block is launched as 32 by 32 threads
//assume enough blocks are launched to tile the whole matrix
__global__ void relu_all_1D(float *matrix, int m, int n){
    int col = blockIdx.x % n / 32;
    int row = blockIdx.x * 32 / n;

    matrix[n * (row * 32 + threadIdx.y) + col * 32 + threadIdx.x] = (col * 32 + threadIdx.x < n && row * 32 + threadIdx.y < m) ? ((matrix[n * (row * 32 + threadIdx.y) + col * 32 + threadIdx.x] > 0.0) ? matrix[n * (row * 32 + threadIdx.y) + col * 32 + threadIdx.x] : 0.0) : 0.0;
}

//we say a vector is positive if its dot product of the vector of all ones is positive
//say it is a column major vector with 1024 elements per column
//launch 1024 x threads per block
//launch as many blocks in x as columns
__global__ void relu_vector(float *matrix_in float *matrix_out){
    int vector = 1024 * blockIdx.x;
    int t = threadIdx.x;
    __shared__ temp[1024] = {0};
    temp[t] = matrix_in[vector + t];
    __syncthreads();
    for(int i = 1; i < 1024; i*=2){
        if((t + 1) * 2 * i <= 1024){
            temp[(t+1) * 2 * i - 1] = temp[(t+1) * 2 * i - 1 - i] + temp[(t+1) * 2 * i - 1]; 
        }
    }
    __syncthreads();
    matrix_out[vector + t] = (temp[1023] >= 0.0) ? matrix_in[vector + t] : 0.0f;
}