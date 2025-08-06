using namespace nvcuda;

#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, const int block_size, const int M, const int N, const int K)
{
    extern __shared__ float s[];
    float dot_product = 0.0f;
    for (int tile_l = 0; tile_l < (K + block_size -1) / block_size; tile_l++) {
        s[threadIdx.y*block_size + threadIdx.x] = (blockIdx.y * blockDim.y + threadIdx.y < M && tile_l * blockDim.x + threadIdx.x < K) ? A[(blockIdx.y * blockDim.y + threadIdx.y)*K + tile_l * blockDim.x + threadIdx.x] : 0.0f; //Loads the A matrix section to the first part of s
        s[block_size*block_size + threadIdx.y * block_size + threadIdx.x] = (tile_l * blockDim.y + threadIdx.y < K && blockIdx.x * blockDim.x + threadIdx.x < N) ? B[(tile_l * blockDim.y + threadIdx.y)*N + blockDim.x * blockIdx.x + threadIdx.x] : 0.0f; //Loads the B matrix section to the second part of s
        __syncthreads();
        for (int k = 0; k < block_size; ++k) {
            dot_product += s[threadIdx.y*block_size + k] * s[block_size*block_size + k * block_size + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x + blockIdx.x * blockDim.x < N && threadIdx.y + blockIdx.y * blockDim.y < M) {
        C[(blockIdx.y * blockDim.y + threadIdx.y) * N + blockDim.x * blockIdx.x + threadIdx.x] = dot_product;
    }
}

// matrix multiply (naive) kernel: C = A * B
//for column major
__global__ void mmul_c(const float *A, const float *B, float *C, const int block_size, const int M, const int N, const int K)
{
    extern __shared__ float s[];
    float dot_product = 0.0f;
    for (int tile_l = 0; tile_l < (K + block_size -1) / block_size; tile_l++) {
        s[threadIdx.y + block_size * threadIdx.x] = (blockIdx.y * blockDim.y + threadIdx.y < M && tile_l * blockDim.x + threadIdx.x < K) ? A[blockIdx.y * blockDim.y + threadIdx.y + (tile_l * blockDim.x + threadIdx.x) * M] : 0.0f; //Loads the A matrix section to the first part of s
        s[block_size*block_size + threadIdx.y * block_size + threadIdx.x] = (tile_l * blockDim.y + threadIdx.y < K && blockIdx.x * blockDim.x + threadIdx.x < N) ? B[tile_l * blockDim.y + threadIdx.y + (blockDim.x * blockIdx.x + threadIdx.x) * K] : 0.0f; //Loads the B matrix section to the second part of s
        __syncthreads();
        for (int k = 0; k < block_size; ++k) {
            dot_product += s[threadIdx.y + block_size * k] * s[block_size*block_size + k * block_size + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x + blockIdx.x * blockDim.x < N && threadIdx.y + blockIdx.y * blockDim.y < M) {
        C[blockIdx.y * blockDim.y + threadIdx.y + (blockDim.x * blockIdx.x + threadIdx.x) * M] = dot_product;
    }
}
/**
 * Assume the matrix is tiled
 * 
 */
__global__ void transpose_duse(float *matrix, const int m, const int n){
    extern __shared__ float s[];
    if(threadIdx.x + blockIdx.x * blockDim.x < n && threadIdx.y + blockIdx.y * blockDim.y < m){
        s[(threadIdx.y + blockIdx.y * blockDim.y) * n + blockIdx.x * blockDim.x + threadIdx.x] = matrix[(blockIdx.x * blockDim.x + threadIdx.x) * m + threadIdx.y + blockIdx.y * blockDim.y]
    }
    __syncthreads();
    if(threadIdx.x + blockIdx.x * blockDim.x < n && threadIdx.y + blockIdx.y * blockDim.y < m){
        matrix[(threadIdx.y + blockIdx.y * blockDim.y) * n + blockIdx.x * blockDim.x + threadIdx.x] = s[(threadIdx.y + blockIdx.y * blockDim.y) * n + blockIdx.x * blockDim.x + threadIdx.x]
    }
}

#define TILE_DIM 32

__global__ void transpose_col_major(float *out, const float *in, const int m, const int n) {
    // Statically allocated shared memory
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Get the global thread indices
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load a tile of the input matrix into shared memory
    // Ensure that we don't read out of bounds
    if (x < n && y < m) {
        tile[threadIdx.y][threadIdx.x] = in[x * m + y];
    }

    // Synchronize to make sure the tile is loaded before we proceed
    __syncthreads();

    // Get the new indices for the transposed matrix
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write the transposed tile from shared memory to the output matrix
    // Ensure that we don't write out of bounds
    if (x < m && y < n) {
        out[x * n + y] = tile[threadIdx.x][threadIdx.y];
    }
}

/*

#include <mma.h>
#include <cuda_bf16.h>

__global__ void matrixMultiplyKernel(half *A, half *B, half *C, int M, int N, int K) {
    // Tile sizes (example)
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;

    // Shared memory for tiles
    __shared__ half sharedA[TILE_M * TILE_K];
    __shared__ half sharedB[TILE_K * TILE_N];
    __shared__ half sharedC[TILE_M * TILE_N]; // Accumulate here first

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // WMMA fragment types
    using MatrixAFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_M, TILE_K, 16, nvcuda::wmma::row_major>;
    using MatrixBFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_K, TILE_N, 16, nvcuda::wmma::col_major>;
    using AccumulatorFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_M, TILE_N, 16>;

    // Initialize accumulator fragment
    AccumulatorFragment accumulatorFragment;
    nvcuda::wmma::fill_fragment(accumulatorFragment, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += TILE_K) {
        // Load tiles from global memory into shared memory
        int aRow = blockRow * TILE_M + threadRow;
        int aCol = k + threadCol;
        int bRow = k + threadRow;
        int bCol = blockCol * TILE_N + threadCol;

        if (aRow < M && aCol < K) {
            sharedA[threadRow * TILE_K + threadCol] = A[aRow * K + aCol];
        } else {
            sharedA[threadRow * TILE_K + threadCol] = 0.0f; // Pad if out of bounds
        }
        if (bRow < K && bCol < N) {
            sharedB[threadRow * TILE_N + threadCol] = B[bRow * N + bCol];
        } else {
            sharedB[threadRow * TILE_N + threadCol] = 0.0f; // Pad if out of bounds
        }
        __syncthreads(); // Ensure all threads load before proceeding

        // Load matrix fragments
        MatrixAFragment matrixAFragment;
        MatrixBFragment matrixBFragment;
        nvcuda::wmma::load_matrix_sync(matrixAFragment, sharedA + threadRow * TILE_K, TILE_K);
        nvcuda::wmma::load_matrix_sync(matrixBFragment, sharedB + threadRow * TILE_N, TILE_N);

        // Perform WMMA operation
        nvcuda::wmma::mma_sync(accumulatorFragment, matrixAFragment, matrixBFragment, accumulatorFragment);

        __syncthreads(); // Ensure all WMMA operations finish
    }

    // Store results back to shared memory
    nvcuda::wmma::store_matrix_sync(sharedC + threadRow * TILE_N + threadCol, accumulatorFragment, TILE_N, nvcuda::wmma::mem_row_major);

    __syncthreads();

    // Store shared memory tile to global memory
    int cRow = blockRow * TILE_M + threadRow;
    int cCol = blockCol * TILE_N + threadCol;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sharedC[threadRow * TILE_N + threadCol];
    }
}

// Define the matrix dimensions for a warp-level operation
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int B = 1;
//Two ways of loading memory:
//Move the 16xk and kx16 chunks to shared memory before the muliply
//Move the 16x16 and 16x16 chunks to shared memory during the muliply
//First I will try option 2
__global__ void matrix_multiply(const float *a, const float *b, float *c, const int K, const int R, const int C) {
    //A and B will both have row major ordering
    //A is a k by 16 matrix
    //B is a 16 by k matrix
    __shared__ __nv_bfloat16 A_row[B * 16 * 16] = {0};
    __shared__ __nv_bfloat16 B_col[B * 16 * 16] = {0};

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    int A_row = blockIdx.x % C / 16;
    int B_col = blockIdx.x * 16 / C;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    for(int i = 0; i < K; i += 16){
        for(int j = 0; j < B; j++){
            A_row[B * (16 * threadIdx.y + threadIdx.x) + j] = __float2bfloat16(a[B * K * 16 * (A_row + threadIdx.y) + B * threadIdx.x + j]);
            B_col[B * (16 * threadIdx.y + threadIdx.x) + j] = __float2bfloat16(b[B * C * 16 * (A_row + threadIdx.y) + B * threadIdx.x + j]);
        }
        __syncthreads(); // Synchronize before loading the next tile

        // Load sub-tiles from shared memory into fragments
        nvcuda::wmma::load_matrix_sync(a_frag, &A_row[0][0], 16 * 16 * 16);
        nvcuda::wmma::load_matrix_sync(b_frag, &B_col[0][0], 16 * 16 * 16);

        // Perform the matrix multiply-accumulate
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads(); // Synchronize before loading the next tile
   }
   // Store the output
   float c_ptr = &(c[B * (C * A_val + B_col * 16)]);
   wmma::store_matrix_sync(c_ptr, c_frag, B*C, wmma::mem_row_major);
}



  dim3 blockmm(8, 4);
  dim3 gridmm(C/16 * R/16);
  matrix_multiply<<<gridmm, blockmm>>>(d_A, d_B, d_C2, K, R, C);
  cudaCheckErrors("kernel launch failure");



*/